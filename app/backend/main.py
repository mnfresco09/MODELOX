"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MODELOX QUANT STATION - API BACKEND v3.0                  ║
║                     Professional Trading Analytics Platform                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import psutil
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy import stats as scipy_stats

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("modelox")

# ============================================================================
# PATHS - Compatible with Docker and Local
# ============================================================================
if os.path.exists("/app/modelox"):
    BASE_DIR = Path("/app")
    IN_DOCKER = True
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    IN_DOCKER = False

DATA_DIR = BASE_DIR / "data" / "ohlcv"
RESULTS_DIR = BASE_DIR / "resultados"
STRATEGIES_DIR = BASE_DIR / "modelox" / "strategies"
EJECUTAR_SCRIPT = BASE_DIR / "ejecutar.py"

logger.info(f"🚀 MODELOX Backend Starting")
logger.info(f"   BASE_DIR: {BASE_DIR}")
logger.info(f"   IN_DOCKER: {IN_DOCKER}")
logger.info(f"   DATA_DIR: {DATA_DIR}")
logger.info(f"   RESULTS_DIR: {RESULTS_DIR}")


# ============================================================================
# SERIALIZATION HELPER
# ============================================================================
def serialize_value(val):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if val is None:
        return None
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (np.ndarray, list)):
        return [serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    if hasattr(val, 'item'):  # numpy scalar
        return val.item()
    return str(val) if not isinstance(val, (str, type(None))) else val


# ============================================================================
# METRIC/PARAMETER CLASSIFIER - ROBUST
# ============================================================================
class MetricParameterClassifier:
    """Clasificador robusto para distinguir métricas de parámetros."""
    
    KNOWN_METRICS = {
        # Identificadores
        'TRIAL', 'TRIAL_NUMBER', 'TRIAL_NUM', 'N_TRIAL',
        'ESTRATEGIA', 'STRATEGY', 'STRATEGY_NAME', 'COMBO', 'COMBINACION',
        
        # Scores
        'SCORE', 'SCORE_FINAL', 'TOTAL_SCORE', 'RANKING',
        
        # Retornos
        'ROI', 'ROI_PCT', 'ROI_PERCENT', 'RETURN', 'RET', 'RET_PCT',
        'PNL', 'PNL_NETO', 'NET_PNL', 'PROFIT', 'LOSS', 'NET_PROFIT',
        'GROSS_PROFIT', 'GROSS_LOSS', 'TOTAL_RETURN',
        
        # Ratios
        'PROFIT_FACTOR', 'PF', 'PAYOFF', 'PAYOFF_RATIO', 'RISK_REWARD',
        'RIESGO_BENEFICIO', 'WIN_LOSS_RATIO',
        
        # Drawdown
        'DRAWDOWN', 'MAX_DD', 'MAX_DD_PCT', 'MAX_DRAWDOWN', 'DD', 'DD_PCT',
        'DRAWDOWN_PCT', 'WORST_DRAWDOWN',
        
        # Trades
        'N_TRADES', 'TOTAL_TRADES', 'NUM_TRADES', 'TRADES', 'TRADES_DIA',
        'TRADES_POR_DIA', 'WINS', 'LOSSES', 'WIN_TRADES', 'LOSS_TRADES',
        'GANADORES', 'PERDEDORES', 'NUM_LONGS', 'NUM_SHORTS', 'COUNT_LONGS',
        'COUNT_SHORTS', 'N_TRADES_LONG', 'N_TRADES_SHORT',
        
        # Win Rate
        'WINRATE', 'WIN_RATE', 'WINRATE_PCT', 'WIN_PCT', 'HIT_RATE',
        'PORC_GANADORAS', 'PORC_PERDEDORAS', 'WIN_RATIO',
        
        # Risk Metrics
        'SQN', 'SHARPE', 'SHARPE_RATIO', 'SORTINO', 'CALMAR', 'OMEGA',
        'EXPECTANCY', 'EXPECTATIVA', 'ESTABILIDAD', 'STABILITY', 'VAR',
        
        # Balance
        'SALDO', 'SALDO_ACTUAL', 'SALDO_FINAL', 'SALDO_INICIAL', 'SALDO_MIN',
        'SALDO_MAX', 'SALDO_MEAN', 'BALANCE', 'EQUITY', 'CAPITAL',
        'SALDO_SIN_COMISIONES',
        
        # Other metrics
        'FECHA', 'DATE', 'DATETIME', 'TIEMPO', 'DURATION', 'DURATION_MEAN_MIN',
        'MAX_GANANCIA', 'MAX_PERDIDA', 'AVG_WIN', 'AVG_LOSS', 'RETORNO_PROMEDIO',
        'MARKET_EXPOSURE', 'FEES', 'COMMISSIONS', 'COMISIONES_TOTAL',
        'RACHA_GANADORA', 'RACHA_PERDEDORA', 'NOMBRE_COMBO',
        'PNL_NETO_POR_DIA_OPERADO',
    }
    
    PARAM_PATTERNS = [
        # Technical Indicators
        r'^(RSI|EMA|SMA|BB|ATR|ADX|MACD|STOCH|CCI|MFI|VWAP|OBV|DMI|ZLEMA|KELTNER)[\s_-]',
        r'(RSI|EMA|SMA|BB|ATR|ADX|MACD|STOCH|CCI|MFI)$',
        
        # Periods/Windows
        r'_PERIOD$', r'_LENGTH$', r'_WINDOW$', r'_LOOKBACK$', r'_LEN$',
        r'^PERIOD', r'^VENTANA', r'^LOOKBACK', r'^LOOKBAR',
        r'FAST_?LEN', r'SLOW_?LEN', r'SIGNAL_?LEN',
        
        # Thresholds
        r'_THRESHOLD$', r'_LEVEL$', r'_MULT$', r'_FACTOR$', r'_COEF$',
        r'^THRESHOLD', r'^UMBRAL', r'^NIVEL',
        
        # Entry/Exit
        r'^ENTRY_', r'^EXIT_', r'^SL[_%]?', r'^TP[_%]?', r'^TRAIL',
        r'^STOP_', r'^TAKE_', r'^RISK_', r'^QTY', r'^SIZE', r'^CANTIDAD',
        r'SL_PCT', r'TP_PCT', r'STOP_LOSS', r'TAKE_PROFIT',
        
        # Directional
        r'_FAST$', r'_SLOW$', r'_SIGNAL$', r'_SHORT$', r'_LONG$',
        
        # Common parameter names
        r'^N_', r'^NUM_(?!TRADES|LONGS|SHORTS)', r'_N$', r'_K$', r'_M$',
        r'^REQ_', r'^MIN_', r'^MAX_(?!DD|DRAWDOWN|GANANCIA|PERDIDA)',
        r'DIST', r'GAP', r'OFFSET', r'SHIFT',
    ]
    
    @classmethod
    def classify_columns(cls, df: pd.DataFrame) -> Dict[str, List[str]]:
        metrics, parameters, unknown = [], [], []
        
        for col in df.columns:
            col_upper = col.upper().strip()
            
            # Skip empty or unnamed columns
            if not col_upper or col_upper.startswith('UNNAMED'):
                continue
                
            if cls._is_metric(col_upper):
                metrics.append(col)
            elif cls._is_parameter(col_upper):
                parameters.append(col)
            elif cls._heuristic_is_parameter(df[col], col_upper):
                parameters.append(col)
            else:
                unknown.append(col)
        
        return {'metrics': metrics, 'parameters': parameters, 'unknown': unknown}
    
    @classmethod
    def _is_metric(cls, col: str) -> bool:
        if col in cls.KNOWN_METRICS:
            return True
        for metric in cls.KNOWN_METRICS:
            if metric in col and len(metric) >= 3:
                return True
        return False
    
    @classmethod
    def _is_parameter(cls, col: str) -> bool:
        for pattern in cls.PARAM_PATTERNS:
            if re.search(pattern, col, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def _heuristic_is_parameter(cls, series: pd.Series, col_name: str) -> bool:
        try:
            numeric = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric) < 5:
                return False
            
            unique_ratio = len(numeric.unique()) / len(numeric)
            if unique_ratio < 0.3 and len(numeric.unique()) > 1:
                return True
            
            if numeric.dtype in [np.int64, np.int32, np.float64]:
                if all(numeric == numeric.astype(int)):
                    if numeric.max() < 1000 and numeric.min() >= 0:
                        return True
            
            return False
        except Exception:
            return False


# ============================================================================
# ADVANCED ANALYSIS ENGINE
# ============================================================================
class AnalysisEngine:
    """Motor de análisis avanzado con detección de ruido y zonas robustas."""
    
    @staticmethod
    def load_file(filepath: str) -> Optional[pd.DataFrame]:
        try:
            path = Path(filepath)
            if not path.exists():
                return None
            
            if path.suffix.lower() == '.csv':
                preview = pd.read_csv(filepath, nrows=20)
                header_row = AnalysisEngine._find_header_row(preview)
                df = pd.read_csv(filepath, header=header_row)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                preview = pd.read_excel(filepath, nrows=20)
                header_row = AnalysisEngine._find_header_row(preview)
                df = pd.read_excel(filepath, header=header_row)
            else:
                return None
            
            df.columns = [str(c).strip().upper().replace(' ', '_') for c in df.columns]
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            return df
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    @staticmethod
    def _find_header_row(df_preview: pd.DataFrame) -> int:
        """Find the header row - checks if current columns look like valid headers."""
        # Check if current columns (row 0 of file) contain expected header keywords
        col_str = " ".join([str(c).upper() for c in df_preview.columns])
        if any(kw in col_str for kw in ['ROI', 'SCORE', 'TRIAL', 'PROFIT', 'PNL', 'WINRATE', 'DRAWDOWN']):
            return 0  # Row 0 is already the header
        
        # Otherwise, search in data rows for a header row
        for i, row in df_preview.iterrows():
            row_str = " ".join([str(x).upper() for x in row.values if pd.notna(x)])
            if any(kw in row_str for kw in ['ROI', 'SCORE', 'TRIAL', 'PROFIT', 'PNL', 'WINRATE']):
                return i + 1
        return 0
    
    @staticmethod
    def analyze_parameter(df: pd.DataFrame, param: str, metric: str = 'SCORE', top_percentile: float = 0.2) -> Dict[str, Any]:
        param_col = None
        metric_col = None
        for col in df.columns:
            if col.upper() == param.upper():
                param_col = col
            if col.upper() == metric.upper():
                metric_col = col
        
        if not param_col or not metric_col:
            return {'error': f'Column not found: param={param}, metric={metric}'}
        
        param_data = pd.to_numeric(df[param_col], errors='coerce')
        metric_data = pd.to_numeric(df[metric_col], errors='coerce')
        
        mask = ~(param_data.isna() | metric_data.isna())
        param_data = param_data[mask]
        metric_data = metric_data[mask]
        
        if len(param_data) < 10:
            return {'error': 'Insufficient data (need at least 10 samples)'}
        
        correlation = param_data.corr(metric_data)
        
        top_threshold = metric_data.quantile(1 - top_percentile)
        top_mask = metric_data >= top_threshold
        
        optimal_range = {
            'min': float(param_data[top_mask].min()),
            'max': float(param_data[top_mask].max()),
            'mean': float(param_data[top_mask].mean()),
            'std': float(param_data[top_mask].std()),
            'median': float(param_data[top_mask].median()),
        }
        
        try:
            n_bins = min(10, len(param_data.unique()))
            if n_bins >= 3:
                param_bins = pd.qcut(param_data, q=n_bins, duplicates='drop')
                bin_stats = metric_data.groupby(param_bins).agg(['mean', 'std', 'count'])
                bin_stats['stability'] = bin_stats['mean'] / (bin_stats['std'] + 1e-6)
                best_bin_idx = bin_stats['stability'].idxmax()
                robustness = {
                    'best_bin': str(best_bin_idx),
                    'stability_score': float(bin_stats['stability'].max()),
                }
            else:
                robustness = {'error': 'Not enough unique values'}
        except Exception as e:
            robustness = {'error': str(e)}
        
        try:
            top_param = param_data[top_mask]
            rest_param = param_data[~top_mask]
            if len(top_param) >= 2 and len(rest_param) >= 2:
                t_stat, p_value = scipy_stats.ttest_ind(top_param, rest_param)
                significance = {'t_statistic': float(t_stat), 'p_value': float(p_value), 'significant': bool(p_value < 0.05)}
            else:
                significance = {'error': 'Not enough samples'}
        except Exception as e:
            significance = {'error': str(e)}
        
        return serialize_value({
            'parameter': param,
            'metric': metric,
            'n_samples': int(len(param_data)),
            'correlation': float(correlation) if not pd.isna(correlation) else 0,
            'correlation_strength': AnalysisEngine._correlation_strength(correlation),
            'param_range': {'min': float(param_data.min()), 'max': float(param_data.max()), 'mean': float(param_data.mean()), 'std': float(param_data.std())},
            'optimal_range': optimal_range,
            'robustness': robustness,
            'significance': significance,
        })
    
    @staticmethod
    def _correlation_strength(corr: float) -> str:
        if pd.isna(corr):
            return 'undefined'
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.4:
            return 'moderate'
        elif abs_corr >= 0.2:
            return 'weak'
        return 'negligible'
    
    @staticmethod
    def noise_analysis(df: pd.DataFrame, metric: str = 'SCORE') -> Dict[str, Any]:
        metric_col = None
        for col in df.columns:
            if col.upper() == metric.upper():
                metric_col = col
                break
        
        if not metric_col:
            return {'error': f'Metric {metric} not found'}
        
        metric_data = pd.to_numeric(df[metric_col], errors='coerce').dropna()
        
        if len(metric_data) < 10:
            return {'error': 'Insufficient data'}
        
        mean = float(metric_data.mean())
        std = float(metric_data.std())
        cv = std / abs(mean) if mean != 0 else float('inf')
        
        q1, q3 = metric_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = metric_data[(metric_data < lower_bound) | (metric_data > upper_bound)]
        
        if cv < 0.1:
            noise_level = 'very_low'
        elif cv < 0.3:
            noise_level = 'low'
        elif cv < 0.5:
            noise_level = 'moderate'
        elif cv < 1.0:
            noise_level = 'high'
        else:
            noise_level = 'very_high'
        
        return {
            'metric': metric,
            'n_samples': int(len(metric_data)),
            'distribution': {'mean': mean, 'std': std, 'cv': float(cv), 'min': float(metric_data.min()), 'max': float(metric_data.max()), 'median': float(metric_data.median()), 'q1': float(q1), 'q3': float(q3)},
            'outliers': {'count': int(len(outliers)), 'percentage': float(len(outliers) / len(metric_data) * 100), 'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}},
            'noise_level': noise_level,
        }


# ============================================================================
# PROCESS MANAGER
# ============================================================================
class ProcessManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.log_queue: Queue = Queue(maxsize=10000)
        self.log_history: List[str] = []  # Keep logs in history
        self.is_running: bool = False
        self.current_config: Optional[Dict] = None
        self.start_time: Optional[datetime] = None
        self.progress: Dict = {'trial': 0, 'total': 0, 'strategy': '', 'asset': '', 'best_score': 0.0, 'status': 'idle', 'eta': None}
        self._lock = threading.Lock()
    
    def start(self, config: Dict) -> bool:
        with self._lock:
            if self.is_running:
                return False
            
            env = os.environ.copy()
            env.update({
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "MODELOX_WEB_MODE": "1",
                "MODELOX_ACTIVO": config.get("asset", "BTC"),
                "MODELOX_TIMEFRAME": config.get("timeframe", "1m"),
                "MODELOX_N_TRIALS": str(config.get("n_trials", 100)),
                "MODELOX_STRATEGY_IDS": ",".join(map(str, config.get("strategy_ids", []))),
            })
            
            script = EJECUTAR_SCRIPT
            if not script.exists():
                logger.error(f"Script not found: {script}")
                return False
            
            try:
                logger.info(f"Starting optimization: {config}")
                
                self.process = subprocess.Popen(
                    [sys.executable, "-u", str(script)],
                    env=env, cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                
                self.is_running = True
                self.current_config = config
                self.start_time = datetime.now()
                self.progress = {'trial': 0, 'total': config.get('n_trials', 100), 'strategy': '', 'asset': config.get('asset', ''), 'best_score': 0.0, 'status': 'running', 'eta': None}
                
                while not self.log_queue.empty():
                    try:
                        self.log_queue.get_nowait()
                    except Empty:
                        break
                
                # Clear log history for new run
                self.log_history = []
                
                threading.Thread(target=self._read_output, daemon=True).start()
                return True
                
            except Exception as e:
                logger.error(f"Failed to start: {e}")
                self.is_running = False
                return False
    
    def _read_output(self):
        try:
            trial_times = []
            
            # Log first line for debugging
            logger.info("Starting to read process output")
            
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                
                line = line.rstrip()
                if not line:
                    continue
                
                # Log important lines
                if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower():
                    logger.error(f"Process error: {line}")
                
                try:
                    self.log_queue.put_nowait(line)
                    # Also add to history (keep last 1000 lines)
                    with self._lock:
                        self.log_history.append(line)
                        if len(self.log_history) > 1000:
                            self.log_history = self.log_history[-1000:]
                except:
                    pass
                
                self._parse_progress(line, trial_times)
            
            self.process.wait()
            exit_code = self.process.returncode
            
            logger.info(f"Process finished with exit code: {exit_code}")
            
            with self._lock:
                self.is_running = False
                self.progress['status'] = 'completed' if exit_code == 0 else 'failed'
            
            self.log_queue.put(f"[FINISHED] Exit code: {exit_code}")
            
        except Exception as e:
            logger.error(f"Output reader error: {e}")
            logger.error(traceback.format_exc())
            with self._lock:
                self.is_running = False
                self.progress['status'] = 'error'
    
    def _parse_progress(self, line: str, trial_times: list):
        try:
            if match := re.search(r'TRIAL\s+(\d+)', line, re.I):
                trial_num = int(match.group(1))
                with self._lock:
                    self.progress['trial'] = trial_num
                
                now = datetime.now()
                trial_times.append(now)
                if len(trial_times) > 1:
                    avg_time = (trial_times[-1] - trial_times[0]).total_seconds() / len(trial_times)
                    remaining = self.progress['total'] - trial_num
                    self.progress['eta'] = avg_time * remaining
            
            if match := re.search(r'SCORE[:\s]+(-?\d+\.?\d*)', line, re.I):
                score = float(match.group(1))
                with self._lock:
                    if score > self.progress['best_score']:
                        self.progress['best_score'] = score
            
            if match := re.search(r'BEST[:\s]+(-?\d+\.?\d*)', line, re.I):
                score = float(match.group(1))
                with self._lock:
                    if score > self.progress['best_score']:
                        self.progress['best_score'] = score
                        
        except Exception:
            pass
    
    def stop(self) -> bool:
        with self._lock:
            if not self.is_running or not self.process:
                return False
            
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.error(f"Stop error: {e}")
            
            self.is_running = False
            self.progress['status'] = 'stopped'
            return True
    
    def get_logs(self, n: int = 100) -> List[str]:
        """Get last n logs from history without consuming them."""
        with self._lock:
            return self.log_history[-n:]
    
    def get_status(self) -> Dict:
        with self._lock:
            elapsed = None
            if self.start_time and self.is_running:
                elapsed = (datetime.now() - self.start_time).total_seconds()
            
            return {'is_running': self.is_running, 'progress': self.progress.copy(), 'config': self.current_config, 'elapsed_seconds': elapsed}


process_mgr = ProcessManager()


# ============================================================================
# FASTAPI APP
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MODELOX Quant Station v3.0 Starting")
    yield
    if process_mgr.is_running:
        process_mgr.stop()

app = FastAPI(title="MODELOX Quant Station", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ============================================================================
# MODELS
# ============================================================================
class OptimizationConfig(BaseModel):
    asset: str = "BTC"
    timeframe: str = "1m"
    n_trials: int = Field(default=100, ge=1, le=100000)
    strategy_ids: List[int] = []

class AnalysisRequest(BaseModel):
    file_path: str
    target_metric: str = "SCORE"
    parameters: Optional[List[str]] = None
    top_percentile: float = Field(default=0.2, ge=0.05, le=0.5)


# ============================================================================
# HELPERS
# ============================================================================
def scan_strategies() -> List[Dict]:
    strategies = []
    if not STRATEGIES_DIR.exists():
        return strategies
    
    for f in STRATEGIES_DIR.glob("*.py"):
        if f.name.startswith("_") or f.name in ("registry.py", "ESTRATEGIA_BASE.py", "__init__.py"):
            continue
        try:
            content = f.read_text(errors='ignore')
            id_match = re.search(r'combinacion_id\s*[=:]\s*(\d+)', content)
            name_match = re.search(r'name\s*[=:]\s*["\']([^"\']+)["\']', content)
            
            sid = int(id_match.group(1)) if id_match else 0
            sname = name_match.group(1) if name_match else f.stem
            
            if sid > 0:
                strategies.append({'id': sid, 'name': sname, 'filename': f.name})
        except Exception:
            pass
    
    return sorted(strategies, key=lambda x: x['id'])


def scan_assets() -> List[Dict]:
    assets = defaultdict(set)
    if not DATA_DIR.exists():
        return []
    
    for ext in ["feather", "csv", "parquet"]:
        for f in DATA_DIR.glob(f"*_ohlcv_*.{ext}"):
            parts = f.stem.split("_ohlcv_")
            if len(parts) == 2:
                asset, tf = parts[0].upper(), parts[1]
                assets[asset].add(tf)
    
    return [{'name': n, 'timeframes': sorted(list(t))} for n, t in sorted(assets.items())]


def scan_results_tree() -> Dict:
    tree = {'strategies': {}, 'total_files': 0}
    if not RESULTS_DIR.exists():
        return tree
    
    for sdir in RESULTS_DIR.iterdir():
        if not sdir.is_dir():
            continue
        
        sname = sdir.name
        tree['strategies'][sname] = {'timeframes': {}, 'total': 0}
        
        for tfdir in sdir.iterdir():
            if not tfdir.is_dir():
                continue
            
            tfname = tfdir.name
            tree['strategies'][sname]['timeframes'][tfname] = {'charts': [], 'csv': []}
            
            graficos_dir = tfdir / "graficos"
            if graficos_dir.exists():
                for asset_dir in graficos_dir.iterdir():
                    if asset_dir.is_dir():
                        for html in asset_dir.glob("*.html"):
                            tree['strategies'][sname]['timeframes'][tfname]['charts'].append({
                                'name': html.name,
                                'path': str(html.relative_to(RESULTS_DIR)),
                                'asset': asset_dir.name.upper(),
                            })
                            tree['strategies'][sname]['total'] += 1
                            tree['total_files'] += 1
            
            excel_dir = tfdir / "excel"
            if excel_dir.exists():
                for csv_file in excel_dir.rglob("*.csv"):
                    tree['strategies'][sname]['timeframes'][tfname]['csv'].append({
                        'name': csv_file.name,
                        'path': str(csv_file.relative_to(RESULTS_DIR)),
                    })
                    tree['strategies'][sname]['total'] += 1
                    tree['total_files'] += 1
    
    return tree


# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/health")
async def health():
    return {"status": "healthy", "version": "3.0.0"}


@app.get("/system/status")
async def system_status():
    mem = psutil.virtual_memory()
    opt_status = process_mgr.get_status()
    
    return {
        "cpu": psutil.cpu_percent(interval=0.1),
        "ram": mem.percent,
        "is_running": opt_status['is_running'],
        "progress": opt_status['progress'],
    }


@app.get("/strategies")
async def get_strategies():
    return scan_strategies()


@app.get("/assets")
async def get_assets():
    return scan_assets()


@app.get("/results/tree")
async def get_results_tree():
    return scan_results_tree()


@app.get("/results/charts")
async def get_charts(strategy: Optional[str] = None, asset: Optional[str] = None, limit: int = 100):
    charts = []
    
    for html in RESULTS_DIR.rglob("*.html"):
        try:
            rel = html.relative_to(RESULTS_DIR)
            parts = str(rel).split("/")
            
            s = parts[0] if parts else None
            a = None
            
            for p in parts:
                if p.upper() in ['BTC', 'GOLD', 'SP500', 'NASDAQ', 'ETH']:
                    a = p.upper()
                    break
            
            if strategy and s and strategy.lower() not in s.lower():
                continue
            if asset and a and asset.upper() != a.upper():
                continue
            
            score = 0.0
            if m := re.search(r'SCORE[_-]?(-?\d+\.?\d*)', html.name, re.I):
                score = float(m.group(1))
            
            charts.append({'name': html.name, 'path': f"/results/{rel}", 'strategy': s, 'asset': a, 'score': score})
        except Exception:
            pass
    
    charts.sort(key=lambda x: x['score'], reverse=True)
    return charts[:limit]


@app.get("/results/summaries")
async def get_summaries():
    summaries = []
    for f in RESULTS_DIR.rglob("*RESUMEN*.csv"):
        summaries.append({'name': f.name, 'path': str(f.relative_to(RESULTS_DIR)), 'full_path': str(f)})
    return summaries


@app.post("/run")
async def run_optimization(config: OptimizationConfig):
    if process_mgr.is_running:
        raise HTTPException(400, "Optimization already running")
    
    if not process_mgr.start(config.model_dump()):
        raise HTTPException(500, "Failed to start optimization")
    
    return {"status": "started", "config": config.model_dump()}


@app.post("/stop")
async def stop_optimization():
    if not process_mgr.is_running:
        raise HTTPException(400, "No optimization running")
    
    process_mgr.stop()
    return {"status": "stopped"}


@app.get("/logs")
async def get_logs(limit: int = 200):
    return {"logs": process_mgr.get_logs(limit), "is_running": process_mgr.is_running, "progress": process_mgr.progress}


@app.get("/progress")
async def get_progress():
    return process_mgr.get_status()


# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================
@app.post("/analysis/classify")
async def classify_columns(req: AnalysisRequest):
    full_path = req.file_path
    if not Path(full_path).is_absolute():
        full_path = str(RESULTS_DIR / req.file_path)
    
    df = AnalysisEngine.load_file(full_path)
    if df is None:
        raise HTTPException(404, f"File not found: {req.file_path}")
    
    return {'file': req.file_path, 'rows': len(df), 'classification': MetricParameterClassifier.classify_columns(df)}


@app.post("/analysis/parameters")
async def analyze_parameters(req: AnalysisRequest):
    full_path = req.file_path
    if not Path(full_path).is_absolute():
        full_path = str(RESULTS_DIR / req.file_path)
    
    df = AnalysisEngine.load_file(full_path)
    if df is None:
        raise HTTPException(404, "File not found")
    
    classification = MetricParameterClassifier.classify_columns(df)
    params = req.parameters or classification['parameters']
    
    results = {}
    for param in params[:30]:
        results[param] = AnalysisEngine.analyze_parameter(df, param, req.target_metric, req.top_percentile)
    
    sorted_results = dict(sorted(results.items(), key=lambda x: abs(x[1].get('correlation', 0)) if 'error' not in x[1] else 0, reverse=True))
    
    return {'file': req.file_path, 'target_metric': req.target_metric, 'impacts': sorted_results}


@app.post("/analysis/noise")
async def analyze_noise(req: AnalysisRequest):
    full_path = req.file_path
    if not Path(full_path).is_absolute():
        full_path = str(RESULTS_DIR / req.file_path)
    
    df = AnalysisEngine.load_file(full_path)
    if df is None:
        raise HTTPException(404, "File not found")
    
    return AnalysisEngine.noise_analysis(df, req.target_metric)


@app.get("/analysis/summary/{file_path:path}")
async def get_file_summary(file_path: str):
    full_path = RESULTS_DIR / file_path
    if not full_path.exists():
        raise HTTPException(404, "File not found")
    
    df = AnalysisEngine.load_file(str(full_path))
    if df is None:
        raise HTTPException(400, "Could not load file")
    
    classification = MetricParameterClassifier.classify_columns(df)
    
    stats = {}
    for col in classification['metrics'][:15]:
        try:
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.notna().sum() > 0:
                stats[col] = {'min': float(numeric.min()), 'max': float(numeric.max()), 'mean': float(numeric.mean())}
        except Exception:
            pass
    
    return {'file': file_path, 'rows': len(df), 'classification': classification, 'stats': stats, 'preview': df.head(10).fillna('').to_dict('records')}


# ============================================================================
# WEBSOCKETS
# ============================================================================
@app.websocket("/ws/logs")
async def websocket_logs(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected: logs")
    
    try:
        while True:
            logs = process_mgr.get_logs(20)
            status = process_mgr.get_status()
            
            await ws.send_json({"logs": logs, "progress": status['progress'], "is_running": status['is_running']})
            await asyncio.sleep(0.2)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: logs")


@app.websocket("/ws/status")
async def websocket_status(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected: status")
    
    try:
        while True:
            mem = psutil.virtual_memory()
            opt_status = process_mgr.get_status()
            
            await ws.send_json({"cpu": psutil.cpu_percent(interval=0.5), "ram": mem.percent, "is_running": opt_status['is_running'], "progress": opt_status['progress']})
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: status")


# ============================================================================
# STATIC FILES
# ============================================================================
if RESULTS_DIR.exists():
    app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
