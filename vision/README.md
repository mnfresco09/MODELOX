# MODELOX Vision Data Store

`vision/` es el nuevo almacen de datos historicos del sistema.

Por ahora se descarga solo BTC futures desde Binance Data Vision (`BTCUSDT`,
timeframe `1m`). El script genera un unico dataset canonico y lo guarda en los
formatos elegidos:

- `vision/BTC_ohlcv_1m.csv`
- `vision/BTC_ohlcv_1m.parquet`
- `vision/BTC_ohlcv_1m.feather`

No se guardan zips raw ni archivos auxiliares como `premium_index`, `metrics`,
`funding_rate` o `aggtrades`. Esas fuentes se descargan temporalmente solo para
construir las columnas del dataset final.

Columnas:

`timestamp`, `open`, `high`, `low`, `close`, `volume`, `num_trades`,
`taker_buy_volume`, `taker_sell_volume`, `taker_buy_quote_volume`,
`taker_sell_quote_volume`, `premium_index_close`, `open_interest_5m`,
`open_interest_value_5m`, `toptrader_count_long_short_ratio_5m`,
`toptrader_sum_long_short_ratio_5m`, `global_long_short_ratio_5m`,
`taker_long_short_ratio_5m`, `predicted_funding_rate_1m`.

Comando base:

```bash
.venv/bin/python descargar_vision.py
```

Rango de fechas:

```bash
.venv/bin/python descargar_vision.py --start 2021-01-01 --end 2025-12-31
```

Formatos:

```bash
.venv/bin/python descargar_vision.py --formats all
.venv/bin/python descargar_vision.py --formats parquet
.venv/bin/python descargar_vision.py --formats csv,parquet
.venv/bin/python descargar_vision.py --formats csv,parquet,feather
```

Por defecto se generan los tres formatos: `csv`, `parquet` y `feather`.
