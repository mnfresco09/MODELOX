"""
Indicador PCI v2.1 independiente del tracker.

Fuentes:
- Precio: Binance (close/trade price)
- Open Interest: Coinalyze
- CVD live: buy/sell taker volume de Binance aggTrade

El pipeline implementa:
1. retorno precio, delta relativo CVD, CVD rolling 14 y cambio % OI normalizado por tiempo
2. EWM de señal (halflife = n)
3. clip robusto sobre estadísticos EWM crudos (halflife = 3n)
4. Z-scores finales sobre la señal suavizada
5. dirección continua con CVD primario y acuerdo tanh(precio_Z * CVD_Z)
6. convicción con gate adaptativo + amplificador saturado del OI
7. score final acotado con tanh
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

PCI_DEFAULT_HALFLIFE_BARS = 15
PCI_CLIP_SIGMAS = 2.5
PCI_GATE_SLOPE = 6.0
PCI_GATE_THRESHOLD_FLOOR = 0.3
PCI_GATE_THRESHOLD_STD_FACTOR = 0.25
PCI_OI_AMPLIFIER_DENOMINATOR = math.sqrt(3.0)
PCI_SCORE_SCALE_FLOOR = 0.1
PCI_SLOPE_WINDOW = 7
PCI_CVD_ROLLING_WINDOW = 14

PCI_EXTRA_FIELDS = (
    "pci_price_z",
    "pci_cvd_z",
    "pci_oi_z",
    "pci_cvd_rolling_14",
    "pci_direction",
    "pci_gate_threshold",
    "pci_gate",
    "pci_amplifier",
    "pci_conviction",
    "pci_score_raw",
    "pci_score",
    "pci_slope",
)

PCI_RAW_FIELDS = (
    "pci_buy_volume_raw",
    "pci_sell_volume_raw",
    "pci_total_volume_raw",
    "pci_open_interest_raw",
    "pci_volume_mean_ewm",
    "pci_volume_threshold",
    "pci_price_return_raw",
    "pci_cvd_delta_raw",
    "pci_oi_change_raw",
)

PCI_ALL_FIELDS = PCI_RAW_FIELDS + PCI_EXTRA_FIELDS


def _safe_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


def _alpha_from_halflife(halflife: int) -> float:
    return 1.0 - math.exp(-math.log(2.0) / max(1.0, float(halflife or 1)))


@dataclass
class _EwmMeanState:
    alpha: float
    value: float | None = None

    def project(self, sample: float) -> float:
        value = _safe_float(sample)
        if self.value is None:
            return value
        return ((1.0 - self.alpha) * self.value) + (self.alpha * value)

    def update(self, sample: float) -> float:
        self.value = self.project(sample)
        return float(self.value or 0.0)


@dataclass
class _EwmMomentsState:
    alpha: float
    mean: float | None = None
    mean_sq: float | None = None

    def project(self, sample: float) -> tuple[float, float]:
        value = _safe_float(sample)
        if self.mean is None or self.mean_sq is None:
            mean = value
            mean_sq = value * value
        else:
            mean = ((1.0 - self.alpha) * self.mean) + (self.alpha * value)
            mean_sq = ((1.0 - self.alpha) * self.mean_sq) + (self.alpha * value * value)
        variance = max(0.0, mean_sq - (mean * mean))
        return float(mean), math.sqrt(variance)

    def update(self, sample: float) -> tuple[float, float]:
        value = _safe_float(sample)
        if self.mean is None or self.mean_sq is None:
            self.mean = value
            self.mean_sq = value * value
        else:
            self.mean = ((1.0 - self.alpha) * self.mean) + (self.alpha * value)
            self.mean_sq = ((1.0 - self.alpha) * self.mean_sq) + (self.alpha * value * value)
        variance = max(0.0, float(self.mean_sq or 0.0) - (float(self.mean or 0.0) ** 2))
        return float(self.mean or 0.0), math.sqrt(variance)


def _clip_with_sigmas(value: float, mean: float, std: float, sigmas: float = PCI_CLIP_SIGMAS) -> float:
    if std <= 0.0:
        return float(value or 0.0)
    lower = mean - (sigmas * std)
    upper = mean + (sigmas * std)
    return min(max(float(value or 0.0), lower), upper)


def _gate_value(abs_direction: float, threshold: float) -> float:
    exponent = -PCI_GATE_SLOPE * (float(abs_direction or 0.0) - float(threshold or 0.0))
    exponent = max(-60.0, min(60.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def _direction_from_price_and_cvd_z(price_z: float, cvd_z: float) -> float:
    agreement = math.tanh(float(price_z or 0.0) * float(cvd_z or 0.0))
    return float(cvd_z or 0.0) * (0.5 + (0.5 * agreement))


def _compute_window_slope(history: deque[float], current_value: float) -> float:
    sample_count = len(history) + 1
    if sample_count < 2:
        return 0.0

    x_mean = (sample_count - 1) / 2.0
    numerator = 0.0
    denominator = 0.0

    index = 0
    for value in history:
        dx = float(index) - x_mean
        numerator += dx * float(value or 0.0)
        denominator += dx * dx
        index += 1

    dx = float(index) - x_mean
    numerator += dx * float(current_value or 0.0)
    denominator += dx * dx

    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _build_output_dict(
    *,
    buy_volume_raw: float,
    sell_volume_raw: float,
    total_volume_raw: float,
    open_interest_raw: float,
    volume_mean_ewm: float,
    volume_threshold: float,
    price_return_raw: float,
    cvd_delta_raw: float,
    cvd_rolling_14: float,
    oi_change_raw: float,
    price_z: float,
    cvd_z: float,
    oi_z: float,
    direction: float,
    gate_threshold: float,
    gate: float,
    amplifier: float,
    conviction: float,
    score_raw: float,
    score: float,
    slope: float,
) -> dict[str, float]:
    return {
        "pci_buy_volume_raw": float(buy_volume_raw),
        "pci_sell_volume_raw": float(sell_volume_raw),
        "pci_total_volume_raw": float(total_volume_raw),
        "pci_open_interest_raw": float(open_interest_raw),
        "pci_volume_mean_ewm": float(volume_mean_ewm),
        "pci_volume_threshold": float(volume_threshold),
        "pci_price_return_raw": float(price_return_raw),
        "pci_cvd_delta_raw": float(cvd_delta_raw),
        "pci_cvd_rolling_14": float(cvd_rolling_14),
        "pci_oi_change_raw": float(oi_change_raw),
        "pci_price_z": float(price_z),
        "pci_cvd_z": float(cvd_z),
        "pci_oi_z": float(oi_z),
        "pci_direction": float(direction),
        "pci_gate_threshold": float(gate_threshold),
        "pci_gate": float(gate),
        "pci_amplifier": float(amplifier),
        "pci_conviction": float(conviction),
        "pci_score_raw": float(score_raw),
        "pci_score": float(score),
        "pci_slope": float(slope),
        "cvd": float(cvd_delta_raw),
        "cvd_rolling_14": float(cvd_rolling_14),
        "absorcion_score": float(score),
        "absorcion_slope": float(slope),
        "open_interest": float(open_interest_raw),
    }


class _PciSymbolState:
    def __init__(self, halflife_bars: int, *, timeframe_ms: int = 300_000):
        self.halflife_bars = max(1, int(halflife_bars or PCI_DEFAULT_HALFLIFE_BARS))
        self.normalization_halflife = max(1, self.halflife_bars * 3)
        self.signal_alpha = _alpha_from_halflife(self.halflife_bars)
        self.normalization_alpha = _alpha_from_halflife(self.normalization_halflife)
        self.timeframe_ms = max(1, int(timeframe_ms or 300_000))

        self.last_price_close: float | None = None
        self.last_open_interest: float | None = None

        self.price_signal = _EwmMeanState(self.signal_alpha)
        self.price_raw_stats = _EwmMomentsState(self.normalization_alpha)
        self.price_final_stats = _EwmMomentsState(self.normalization_alpha)

        self.cvd_signal = _EwmMeanState(self.signal_alpha)
        self.cvd_raw_stats = _EwmMomentsState(self.normalization_alpha)
        self.cvd_final_stats = _EwmMomentsState(self.normalization_alpha)

        self.oi_signal = _EwmMeanState(self.signal_alpha)
        self.oi_raw_stats = _EwmMomentsState(self.normalization_alpha)
        self.oi_final_stats = _EwmMomentsState(self.normalization_alpha)

        self.abs_direction_stats = _EwmMomentsState(self.normalization_alpha)
        self.score_raw_stats = _EwmMomentsState(self.normalization_alpha)
        self.score_history: deque[float] = deque(maxlen=PCI_SLOPE_WINDOW)
        self.cvd_history: deque[float] = deque(maxlen=PCI_CVD_ROLLING_WINDOW)

    @staticmethod
    def _sanitize_snapshot(
        buy_volume_raw: float,
        sell_volume_raw: float,
        total_volume_raw: float,
        open_interest_raw: float,
    ) -> tuple[float, float, float, float]:
        buy = max(0.0, _safe_float(buy_volume_raw))
        sell = max(0.0, _safe_float(sell_volume_raw))
        volume = max(0.0, _safe_float(total_volume_raw))
        total = max(volume, buy + sell)
        oi = max(0.0, _safe_float(open_interest_raw))
        return buy, sell, total, oi

    @staticmethod
    def _current_mean_std(state: _EwmMomentsState) -> tuple[float, float]:
        if state.mean is None or state.mean_sq is None:
            return 0.0, 0.0
        mean = float(state.mean or 0.0)
        variance = max(0.0, float(state.mean_sq or 0.0) - (mean * mean))
        return mean, math.sqrt(variance)

    def _process_metric(
        self,
        raw_value: float,
        signal_state: _EwmMeanState,
        raw_stats_state: _EwmMomentsState,
        final_stats_state: _EwmMomentsState,
        *,
        commit: bool,
    ) -> float:
        signal = signal_state.update(raw_value) if commit else signal_state.project(raw_value)
        raw_mean, raw_std = (
            raw_stats_state.update(signal)
            if commit
            else raw_stats_state.project(signal)
        )
        clipped_signal = _clip_with_sigmas(signal, raw_mean, raw_std)
        final_mean, final_std = (
            final_stats_state.update(clipped_signal)
            if commit
            else final_stats_state.project(clipped_signal)
        )
        if final_std <= 0.0:
            return 0.0
        return float((signal - final_mean) / final_std)

    def _project_cvd_rolling(self, current_value: float) -> float:
        projected = deque(self.cvd_history, maxlen=PCI_CVD_ROLLING_WINDOW)
        projected.append(float(current_value or 0.0))
        return float(sum(projected))

    def _compute_metrics(
        self,
        price: float,
        *,
        buy_volume_raw: float,
        sell_volume_raw: float,
        total_volume_raw: float,
        open_interest_raw: float,
        commit: bool,
    ) -> dict[str, float]:
        buy, sell, total, oi = self._sanitize_snapshot(
            buy_volume_raw,
            sell_volume_raw,
            total_volume_raw,
            open_interest_raw,
        )
        price_value = max(0.0, _safe_float(price))

        # Campo retenido solo por compatibilidad del contrato del runtime/UI.
        volume_mean_ewm = 0.0
        volume_threshold = 0.0

        price_return_raw = 0.0
        if price_value > 0.0 and self.last_price_close is not None and self.last_price_close > 0.0:
            price_return_raw = (price_value - self.last_price_close) / self.last_price_close

        if total > 0.0:
            cvd_delta_raw = (buy - sell) / total
        else:
            cvd_delta_raw = 0.0
        cvd_delta_raw = max(-1.0, min(1.0, cvd_delta_raw))
        cvd_rolling_14 = self._project_cvd_rolling(cvd_delta_raw)

        oi_change_raw = 0.0
        if oi > 0.0 and self.last_open_interest is not None and self.last_open_interest > 0.0:
            timeframe_hours = max(float(self.timeframe_ms) / 3_600_000.0, 1e-9)
            oi_change_raw = ((oi - self.last_open_interest) / self.last_open_interest) / timeframe_hours

        price_z = self._process_metric(
            price_return_raw,
            self.price_signal,
            self.price_raw_stats,
            self.price_final_stats,
            commit=commit,
        )
        cvd_z = self._process_metric(
            cvd_delta_raw,
            self.cvd_signal,
            self.cvd_raw_stats,
            self.cvd_final_stats,
            commit=commit,
        )
        oi_z = self._process_metric(
            oi_change_raw,
            self.oi_signal,
            self.oi_raw_stats,
            self.oi_final_stats,
            commit=commit,
        )

        direction = _direction_from_price_and_cvd_z(price_z, cvd_z)

        abs_direction = abs(direction)
        mean_abs_direction, std_abs_direction = self._current_mean_std(self.abs_direction_stats)
        gate_threshold = max(
            PCI_GATE_THRESHOLD_FLOOR,
            mean_abs_direction + (PCI_GATE_THRESHOLD_STD_FACTOR * std_abs_direction),
        )
        gate = _gate_value(abs_direction, gate_threshold)
        amplifier = math.sqrt(abs(oi_z)) / PCI_OI_AMPLIFIER_DENOMINATOR if abs(oi_z) > 0.0 else 0.0
        conviction = 1.0 + (gate * amplifier)
        score_raw = direction * conviction

        _score_mean, score_scale = self._current_mean_std(self.score_raw_stats)
        score_scale = max(PCI_SCORE_SCALE_FLOOR, score_scale)
        score = math.tanh(score_raw / score_scale) * 3.0
        slope = _compute_window_slope(self.score_history, score)

        if commit:
            self.abs_direction_stats.update(abs_direction)
            self.score_raw_stats.update(score_raw)
            self.score_history.append(float(score))
            self.cvd_history.append(float(cvd_delta_raw))
            if price_value > 0.0:
                self.last_price_close = price_value
            if oi > 0.0:
                self.last_open_interest = oi

        return _build_output_dict(
            buy_volume_raw=buy,
            sell_volume_raw=sell,
            total_volume_raw=total,
            open_interest_raw=oi,
            volume_mean_ewm=volume_mean_ewm,
            volume_threshold=volume_threshold,
            price_return_raw=price_return_raw,
            cvd_delta_raw=cvd_delta_raw,
            cvd_rolling_14=cvd_rolling_14,
            oi_change_raw=oi_change_raw,
            price_z=price_z,
            cvd_z=cvd_z,
            oi_z=oi_z,
            direction=direction,
            gate_threshold=gate_threshold,
            gate=gate,
            amplifier=amplifier,
            conviction=conviction,
            score_raw=score_raw,
            score=score,
            slope=slope,
        )

    def compute(
        self,
        price: float,
        *,
        buy_volume_raw: float,
        sell_volume_raw: float,
        total_volume_raw: float,
        open_interest_raw: float,
    ) -> dict[str, float]:
        return self._compute_metrics(
            price,
            buy_volume_raw=buy_volume_raw,
            sell_volume_raw=sell_volume_raw,
            total_volume_raw=total_volume_raw,
            open_interest_raw=open_interest_raw,
            commit=False,
        )

    def commit(
        self,
        price: float,
        *,
        buy_volume_raw: float,
        sell_volume_raw: float,
        total_volume_raw: float,
        open_interest_raw: float,
    ) -> dict[str, float]:
        return self._compute_metrics(
            price,
            buy_volume_raw=buy_volume_raw,
            sell_volume_raw=sell_volume_raw,
            total_volume_raw=total_volume_raw,
            open_interest_raw=open_interest_raw,
            commit=True,
        )


def calculate_pci_frame(
    rows: list[dict],
    *,
    halflife_bars: int = PCI_DEFAULT_HALFLIFE_BARS,
    timeframe_ms: int = 300_000,
) -> pl.DataFrame:
    import polars as pl

    if not rows:
        schema = {"timestamp": pl.Int64}
        for field in PCI_ALL_FIELDS:
            schema[field] = pl.Float64
        return pl.DataFrame(schema=schema)

    frame = pl.DataFrame(rows).sort("timestamp")
    missing_columns = {
        "close": 0.0,
        "open_interest_raw": 0.0,
        "pci_buy_volume_raw": 0.0,
        "pci_sell_volume_raw": 0.0,
        "pci_total_volume_raw": 0.0,
        "pci_open_interest_raw": 0.0,
        "is_closed": True,
    }
    missing_exprs = [
        pl.lit(default_value).alias(column_name)
        for column_name, default_value in missing_columns.items()
        if column_name not in frame.columns
    ]
    if missing_exprs:
        frame = frame.with_columns(missing_exprs)

    frame = (
        frame
        .with_columns(
            [
                pl.col("timestamp").cast(pl.Int64),
                pl.col("close").cast(pl.Float64),
                pl.coalesce(
                    [
                        pl.col("pci_buy_volume_raw"),
                        pl.lit(0.0),
                    ]
                ).cast(pl.Float64).alias("_pci_buy_volume_raw"),
                pl.coalesce(
                    [
                        pl.col("pci_sell_volume_raw"),
                        pl.lit(0.0),
                    ]
                ).cast(pl.Float64).alias("_pci_sell_volume_raw"),
                pl.coalesce(
                    [
                        pl.col("pci_open_interest_raw"),
                        pl.col("open_interest_raw"),
                        pl.lit(0.0),
                    ]
                ).cast(pl.Float64).alias("_pci_open_interest_raw"),
            ]
        )
        .with_columns(
            [
                pl.max_horizontal(
                    [
                        pl.coalesce(
                            [
                                pl.col("pci_total_volume_raw"),
                                pl.lit(0.0),
                            ]
                        ).cast(pl.Float64),
                        (pl.col("_pci_buy_volume_raw") + pl.col("_pci_sell_volume_raw")).cast(pl.Float64),
                    ]
                ).alias("_pci_total_volume_raw"),
                pl.coalesce(
                    [
                        pl.col("is_closed"),
                        pl.lit(True),
                    ]
                ).cast(pl.Boolean).alias("_pci_is_closed"),
            ]
        )
        .select(
            [
                "timestamp",
                "close",
                "_pci_buy_volume_raw",
                "_pci_sell_volume_raw",
                "_pci_total_volume_raw",
                "_pci_open_interest_raw",
                "_pci_is_closed",
            ]
        )
    )

    columns = frame.to_dict(as_series=False)
    state = _PciSymbolState(halflife_bars, timeframe_ms=timeframe_ms)
    metrics: list[dict[str, float | int]] = []
    row_count = len(columns.get("timestamp", []))
    for index in range(row_count):
        price = _safe_float(columns["close"][index])
        buy = _safe_float(columns["_pci_buy_volume_raw"][index])
        sell = _safe_float(columns["_pci_sell_volume_raw"][index])
        total = _safe_float(columns["_pci_total_volume_raw"][index])
        oi = _safe_float(columns["_pci_open_interest_raw"][index])
        is_closed = bool(columns["_pci_is_closed"][index])

        if is_closed:
            pci = state.commit(
                price,
                buy_volume_raw=buy,
                sell_volume_raw=sell,
                total_volume_raw=total,
                open_interest_raw=oi,
            )
        else:
            pci = state.compute(
                price,
                buy_volume_raw=buy,
                sell_volume_raw=sell,
                total_volume_raw=total,
                open_interest_raw=oi,
            )
        metrics.append({
            "timestamp": int(columns["timestamp"][index]),
            **pci,
        })

    return pl.DataFrame(metrics)


def calculate_pci_rows(
    rows: list[dict],
    *,
    halflife_bars: int = PCI_DEFAULT_HALFLIFE_BARS,
    timeframe_ms: int = 300_000,
) -> list[dict]:
    frame = calculate_pci_frame(rows, halflife_bars=halflife_bars, timeframe_ms=timeframe_ms)
    return frame.to_dicts()


class PciCalculator:
    HALFLIFE_BARS = PCI_DEFAULT_HALFLIFE_BARS

    def __init__(self, halflife_bars: int | None = None, *, timeframe_ms: int = 300_000):
        self.halflife_bars = max(1, int(halflife_bars or self.HALFLIFE_BARS))
        self.normalization_halflife = max(1, self.halflife_bars * 3)
        self.warmup_bars = max(1, self.halflife_bars * 9)
        self.timeframe_ms = max(1, int(timeframe_ms or 300_000))
        self._states: dict[str, _PciSymbolState] = {}
        self._current_open_interest: dict[str, float | None] = {}
        self._current_buy_volume: dict[str, float] = {}
        self._current_sell_volume: dict[str, float] = {}
        self._current_total_volume: dict[str, float] = {}
        self._current_last_trade_timestamp_ms: dict[str, int] = {}
        self._current_last_snapshot_timestamp_ms: dict[str, int] = {}

    def _init_symbol(self, symbol: str) -> str:
        normalized = str(symbol).upper()
        if normalized not in self._states:
            self._states[normalized] = _PciSymbolState(
                self.halflife_bars,
                timeframe_ms=self.timeframe_ms,
            )
            self._current_open_interest[normalized] = None
            self._current_buy_volume[normalized] = 0.0
            self._current_sell_volume[normalized] = 0.0
            self._current_total_volume[normalized] = 0.0
            self._current_last_trade_timestamp_ms[normalized] = 0
            self._current_last_snapshot_timestamp_ms[normalized] = 0
        return normalized

    def actualizar_snapshot(
        self,
        symbol: str,
        open_interest_raw: float | None,
        buy_volume_raw: float | None,
        sell_volume_raw: float | None,
        total_volume_raw: float | None = None,
        *,
        snapshot_timestamp_ms: int | None = None,
    ) -> bool:
        normalized = self._init_symbol(symbol)
        incoming_snapshot_ts = max(0, int(snapshot_timestamp_ms or 0))
        latest_volume_ts = max(
            int(self._current_last_trade_timestamp_ms.get(normalized, 0) or 0),
            int(self._current_last_snapshot_timestamp_ms.get(normalized, 0) or 0),
        )
        if incoming_snapshot_ts > 0 and incoming_snapshot_ts < latest_volume_ts:
            return False
        buy = max(0.0, _safe_float(buy_volume_raw))
        sell = max(0.0, _safe_float(sell_volume_raw))
        total = max(
            max(0.0, _safe_float(total_volume_raw)),
            buy + sell,
        )
        # Los snapshots parciales de kline pueden venir sin OI; en ese caso
        # preservamos el último valor conocido en vez de borrarlo.
        oi = open_interest_raw
        if oi is not None:
            self._current_open_interest[normalized] = max(0.0, _safe_float(oi))
        self._current_buy_volume[normalized] = buy
        self._current_sell_volume[normalized] = sell
        self._current_total_volume[normalized] = total
        if incoming_snapshot_ts > 0:
            self._current_last_snapshot_timestamp_ms[normalized] = incoming_snapshot_ts
        return True

    def actualizar_open_interest(
        self,
        symbol: str,
        open_interest_raw: float | None,
    ) -> None:
        normalized = self._init_symbol(symbol)
        if open_interest_raw is None:
            self._current_open_interest[normalized] = None
            return
        self._current_open_interest[normalized] = max(0.0, _safe_float(open_interest_raw))

    def registrar_trade(
        self,
        symbol: str,
        quantity: float | None,
        *,
        buyer_is_maker: bool,
        trade_timestamp_ms: int | None = None,
    ) -> None:
        normalized = self._init_symbol(symbol)
        incoming_trade_ts = max(0, int(trade_timestamp_ms or 0))
        latest_snapshot_ts = int(self._current_last_snapshot_timestamp_ms.get(normalized, 0) or 0)
        if incoming_trade_ts > 0 and latest_snapshot_ts > 0 and incoming_trade_ts < latest_snapshot_ts:
            return
        base_volume = max(0.0, _safe_float(quantity))
        if base_volume <= 0.0:
            return
        if buyer_is_maker:
            self._current_sell_volume[normalized] += base_volume
        else:
            self._current_buy_volume[normalized] += base_volume
        self._current_total_volume[normalized] = (
            self._current_buy_volume[normalized] + self._current_sell_volume[normalized]
        )
        if incoming_trade_ts > 0:
            self._current_last_trade_timestamp_ms[normalized] = max(
                int(self._current_last_trade_timestamp_ms.get(normalized, 0) or 0),
                incoming_trade_ts,
            )

    def _snapshot_actual(self, symbol: str) -> tuple[float, float, float, float]:
        normalized = self._init_symbol(symbol)
        state = self._states[normalized]
        oi = self._current_open_interest.get(normalized)
        open_interest_raw = state.last_open_interest if oi is None else float(oi or 0.0)
        return (
            float(self._current_buy_volume.get(normalized, 0.0) or 0.0),
            float(self._current_sell_volume.get(normalized, 0.0) or 0.0),
            float(self._current_total_volume.get(normalized, 0.0) or 0.0),
            max(0.0, float(open_interest_raw or 0.0)),
        )

    def obtener_metricas_actuales(self, symbol: str, precio_actual: float) -> dict[str, float]:
        normalized = self._init_symbol(symbol)
        buy, sell, total, oi = self._snapshot_actual(normalized)
        return self._states[normalized].compute(
            precio_actual,
            buy_volume_raw=buy,
            sell_volume_raw=sell,
            total_volume_raw=total,
            open_interest_raw=oi,
        )

    def obtener_metricas_desde_candle(
        self,
        symbol: str,
        precio_actual: float,
        open_interest_raw: float,
        buy_volume_raw: float,
        sell_volume_raw: float,
        total_volume_raw: float | None = None,
    ) -> dict[str, float]:
        normalized = self._init_symbol(symbol)
        return self._states[normalized].compute(
            precio_actual,
            buy_volume_raw=buy_volume_raw,
            sell_volume_raw=sell_volume_raw,
            total_volume_raw=total_volume_raw if total_volume_raw is not None else (buy_volume_raw + sell_volume_raw),
            open_interest_raw=open_interest_raw,
        )

    def cerrar_vela(
        self,
        symbol: str,
        precio_cierre: float,
        *,
        open_interest_raw_cierre: float | None = None,
        buy_volume_raw_cierre: float | None = None,
        sell_volume_raw_cierre: float | None = None,
        total_volume_raw_cierre: float | None = None,
    ) -> dict[str, float]:
        normalized = self._init_symbol(symbol)
        current_buy, current_sell, current_total, current_oi = self._snapshot_actual(normalized)
        buy = current_buy if buy_volume_raw_cierre is None else max(0.0, _safe_float(buy_volume_raw_cierre))
        sell = current_sell if sell_volume_raw_cierre is None else max(0.0, _safe_float(sell_volume_raw_cierre))
        total = (
            current_total
            if total_volume_raw_cierre is None
            else max(max(0.0, _safe_float(total_volume_raw_cierre)), buy + sell)
        )
        oi = current_oi if open_interest_raw_cierre is None else max(0.0, _safe_float(open_interest_raw_cierre))

        metrics = self._states[normalized].commit(
            precio_cierre,
            buy_volume_raw=buy,
            sell_volume_raw=sell,
            total_volume_raw=total,
            open_interest_raw=oi,
        )
        self.reiniciar(normalized)
        return metrics

    def reiniciar(self, symbol: str) -> None:
        normalized = self._init_symbol(symbol)
        self._current_open_interest[normalized] = None
        self._current_buy_volume[normalized] = 0.0
        self._current_sell_volume[normalized] = 0.0
        self._current_total_volume[normalized] = 0.0
        self._current_last_trade_timestamp_ms[normalized] = 0
        self._current_last_snapshot_timestamp_ms[normalized] = 0

    def reiniciar_todos(self) -> None:
        for symbol in list(self._states):
            self.reiniciar(symbol)

    def exportar_estado_runtime(self) -> dict:
        return {
            symbol: {
                "current_open_interest": self._current_open_interest.get(symbol),
                "current_buy_volume": float(self._current_buy_volume.get(symbol, 0.0) or 0.0),
                "current_sell_volume": float(self._current_sell_volume.get(symbol, 0.0) or 0.0),
                "current_total_volume": float(self._current_total_volume.get(symbol, 0.0) or 0.0),
                "current_last_trade_timestamp_ms": int(self._current_last_trade_timestamp_ms.get(symbol, 0) or 0),
                "current_last_snapshot_timestamp_ms": int(self._current_last_snapshot_timestamp_ms.get(symbol, 0) or 0),
            }
            for symbol in self._states
            if (
                self._current_open_interest.get(symbol) is not None
                or float(self._current_buy_volume.get(symbol, 0.0) or 0.0) > 0.0
                or float(self._current_sell_volume.get(symbol, 0.0) or 0.0) > 0.0
                or float(self._current_total_volume.get(symbol, 0.0) or 0.0) > 0.0
                or int(self._current_last_trade_timestamp_ms.get(symbol, 0) or 0) > 0
                or int(self._current_last_snapshot_timestamp_ms.get(symbol, 0) or 0) > 0
            )
        }

    def importar_estado_runtime(self, estado) -> None:
        if not isinstance(estado, dict):
            return

        for symbol, valores in estado.items():
            normalized = self._init_symbol(symbol)
            if not isinstance(valores, dict):
                continue
            current_open_interest = valores.get("current_open_interest")
            self._current_open_interest[normalized] = (
                None if current_open_interest is None else max(0.0, _safe_float(current_open_interest))
            )
            self._current_buy_volume[normalized] = max(0.0, _safe_float(valores.get("current_buy_volume", 0.0)))
            self._current_sell_volume[normalized] = max(0.0, _safe_float(valores.get("current_sell_volume", 0.0)))
            self._current_total_volume[normalized] = max(
                max(0.0, _safe_float(valores.get("current_total_volume", 0.0))),
                self._current_buy_volume[normalized] + self._current_sell_volume[normalized],
            )
            self._current_last_trade_timestamp_ms[normalized] = max(
                0,
                int(valores.get("current_last_trade_timestamp_ms", 0) or 0),
            )
            self._current_last_snapshot_timestamp_ms[normalized] = max(
                0,
                int(valores.get("current_last_snapshot_timestamp_ms", 0) or 0),
            )

    def importar_historico(self, estado) -> None:
        self._states = {}
        self._current_open_interest = {}
        self._current_buy_volume = {}
        self._current_sell_volume = {}
        self._current_total_volume = {}
        self._current_last_trade_timestamp_ms = {}
        self._current_last_snapshot_timestamp_ms = {}
        if not isinstance(estado, dict):
            return

        for symbol, valores in estado.items():
            normalized = self._init_symbol(symbol)
            candles = valores.get("candles", valores.get("history", [])) if isinstance(valores, dict) else valores
            if not isinstance(candles, list):
                continue
            for candle in candles:
                if not isinstance(candle, dict):
                    continue
                close = _safe_float(candle.get("close", 0.0))
                oi = _safe_float(
                    candle.get(
                        "pci_open_interest_raw",
                        candle.get("open_interest_raw", 0.0),
                    )
                )
                buy = _safe_float(candle.get("pci_buy_volume_raw", candle.get("buy_volume", 0.0)))
                sell = _safe_float(candle.get("pci_sell_volume_raw", candle.get("sell_volume", 0.0)))
                total = _safe_float(
                    candle.get(
                        "pci_total_volume_raw",
                        candle.get("volume", buy + sell),
                    )
                )
                self._states[normalized].commit(
                    close,
                    buy_volume_raw=buy,
                    sell_volume_raw=sell,
                    total_volume_raw=max(total, buy + sell),
                    open_interest_raw=oi,
                )
