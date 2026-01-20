import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import sys
import os
import scipy.stats as stats

# ==============================================================================
# 1. CONFIGURACIÓN VISUAL INSTITUCIONAL ("SOPHISTICATED QUANT")
# ==============================================================================

# CONFIGURACIÓN DE ESTILO BASE
plt.style.use('dark_background')
warnings.filterwarnings('ignore')

# PALETA DE COLORES PROFESIONAL (NEUTRA Y FINANCIERA)
COLORS = {
    'bg':       '#0E1117',    # Gunmetal Dark (Fondo General)
    'plot_bg':  '#161B22',    # Panel Background (Ligeramente más claro)
    'text':     '#C9D1D9',    # Soft White (Texto Principal)
    'subtext':  '#8B949E',    # Slate Grey (Texto Secundario)
    'grid':     '#21262D',    # Subtle Grid
    'trend':    '#FFFFFF',    # Línea Principal (Blanco Puro)
    'area':     '#79C0FF',    # Neutral Blue (Bandas)
    'up':       '#56A868',    # Sage Green (Profit - No Neón)
    'down':     '#E5534B',    # Brick Red (Risk - No Neón)
    'gold':     '#D29922',    # Muted Gold (Highlights)
    'table_h':  '#1F2428',    # Table Header
    'table_r':  '#0D1117'     # Table Row
}

# ACTUALIZACIÓN DE PARÁMETROS GLOBALES DE MATPLOTLIB
plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor':   COLORS['plot_bg'],
    'axes.edgecolor':   '#30363D',
    'axes.linewidth':   1.0,
    'grid.color':       COLORS['grid'],
    'grid.linestyle':   '-',
    'grid.linewidth':   0.8,
    'grid.alpha':       0.5,
    'text.color':       COLORS['text'],
    'axes.labelcolor':  COLORS['subtext'],
    'xtick.color':      COLORS['subtext'],
    'ytick.color':      COLORS['subtext'],
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Segoe UI', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':        9,
    'axes.titleweight': 'medium', 
    'axes.titlesize':   11
})


# ==============================================================================
# 2. CLASE DE CARGA Y PROCESAMIENTO DE DATOS
# ==============================================================================

class QuantLoader:
    def __init__(self):
        # Mapeo de métricas clave
        self.TARGET_METRICS = {
            'ROI': 'ROI_PCT',
            'DD':  'MAX_DD_PCT',
            'PF':  'PROFIT_FACTOR',
            'SQN': 'SQN'
        }
        
        self.TRADE_COL = 'TOTAL_TRADES'
        
        # Columnas a ignorar (Metadatos que no son parámetros)
        self.IGNORE_COLS = [
            'TRIAL', 'ESTRATEGIA', 'SCORE', 'SALDO_ACTUAL', 'SALDO_INICIAL',
            'WINRATE', 'WINRATE_PCT', 'TOTAL_TRADES', 'TRADES_DIA', 'TRADES_POR_DIA',
            'SHARPE', 'SORTINO', 'ESTABILIDAD', 'EXPECTATIVA', 'NUM_LONGS', 'NUM_SHORTS',
            'FECHA', 'TIEMPO', 'DURATION', 'GROSS_PROFIT', 'GROSS_LOSS', 'RET_PCT', 
            'DRAWDOWN', 'DRAWDOWN_PCT', 'MARKET_EXPOSURE', 'FEES', 'COMMISSIONS',
            'RACHA_GANADORA', 'RACHA_PERDEDORA', 'PORC_GANADORAS', 'PORC_PERDEDORAS',
            'PNL_NETO_POR_DIA_OPERADO', 'N_TRADES', 'NUM_TRADES', 'N_TRADES_LONG',
            'COUNT_LONGS', 'N_TRADES_SHORT', 'COUNT_SHORTS', 'RIESGO_BENEFICIO',
            'PAYOFF_RATIO', 'CALMAR', 'SALDO_MIN', 'SALDO_MAX', 'SALDO_MEAN',
            'MAX_GANANCIA', 'MAX_PERDIDA', 'DURATION_MEAN_MIN', 'COMISIONES_TOTAL',
            'SALDO_SIN_COMISIONES', 'PNL_NETO', 'NET_PNL', 'RETORNO_PROMEDIO', 
            'NOMBRE_COMBO'
        ]

    def _find_header_row(self, df_preview):
        """Busca la fila real del encabezado."""
        for i, row in df_preview.iterrows():
            row_str = " ".join([str(x).upper() for x in row.values])
            if ('ROI' in row_str or 'ROI_PCT' in row_str) and 'TRIAL' in row_str:
                return i + 1
        return 0

    def load_data(self, file_path):
        """Carga y normaliza los datos con limpieza estricta de tipos numéricos."""
        print(f"--- 📡 READING DATASET: {os.path.basename(file_path)} ---")
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            # 1. Carga inicial
            if ext == '.csv':
                preview = pd.read_csv(file_path, nrows=20)
                header_idx = self._find_header_row(preview)
                df = pd.read_csv(file_path, header=header_idx if header_idx > 0 else 0)
            elif ext in ['.xlsx', '.xls']:
                preview = pd.read_excel(file_path, nrows=20)
                header_idx = self._find_header_row(preview)
                df = pd.read_excel(file_path, header=header_idx if header_idx > 0 else 0)
            else:
                return None

            # 2. Normalización de columnas
            df.columns = [str(c).strip().replace(' ', '_').upper() for c in df.columns]
            
            rename_map = {}
            for col in df.columns:
                if 'ROI' in col and 'PCT' not in col: rename_map[col] = 'ROI_PCT'
                if 'DRAWDOWN' in col: rename_map[col] = 'MAX_DD_PCT' 
                if 'PROFIT' in col and 'FACTOR' in col: rename_map[col] = 'PROFIT_FACTOR'
            
            if rename_map: 
                df.rename(columns=rename_map, inplace=True)
            
            # 3. CONVERSIÓN NUMÉRICA ESTRICTA (Soluciona el error de formato 'str')
            # Forzamos conversión en las métricas clave. Si falla, pone NaN.
            for metric in self.TARGET_METRICS.values():
                if metric in df.columns:
                    df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
            # Intentar convertir el resto de columnas numéricas (Parámetros)
            df = df.apply(pd.to_numeric, errors='ignore')
            
            # 4. Limpieza final
            if self.TARGET_METRICS['ROI'] not in df.columns:
                return None
            
            # Eliminar filas donde el ROI sea NaN (datos sucios o filas vacías)
            df = df.dropna(subset=[self.TARGET_METRICS['ROI']])
            
            # Detectar columna de Trades
            for c in df.columns:
                if 'TRADES' in c and 'TOTAL' in c: 
                    self.TRADE_COL = c
                    break
            
            return df
        except Exception as e:
            print(f"⚠️ Warning loading data: {e}")
            return None

    def get_params(self, df):
        """Identifica parámetros numéricos variables."""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        candidates = [c for c in numeric if c not in self.TARGET_METRICS.values() and c not in self.IGNORE_COLS]
        return sorted([c for c in candidates if df[c].nunique() > 1])


# ==============================================================================
# 3. MOTOR MATEMÁTICO: REGRESIÓN DE CUANTILES PONDERADA
# ==============================================================================

def weighted_quantile_smooth(x, y, weights, q=0.75, bandwidth=None, resolution=150):
    """Calcula Percentiles Ponderados (Suavizado No-Paramétrico)."""
    # Filtrado estricto de infinitos y NaNs
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
    x, y, weights = x[mask], y[mask], weights[mask]
    
    if len(x) < 10: 
        return None, None 

    x_min, x_max = x.min(), x.max()
    margin = (x_max - x_min) * 0.05
    x_grid = np.linspace(x_min - margin, x_max + margin, resolution)
    
    if bandwidth is None:
        bandwidth = 1.06 * x.std() * (len(x) ** (-1/5))

    y_smooth = []

    for x0 in x_grid:
        dists = (x - x0) / bandwidth
        kernel_weights = np.exp(-0.5 * dists**2)
        final_weights = kernel_weights * weights
        
        # Filtrar pesos insignificantes
        idx = final_weights > 1e-5
        if not np.any(idx):
            y_smooth.append(np.nan)
            continue
            
        y_local = y[idx]
        w_local = final_weights[idx]
        
        # Ordenamiento para cuantiles
        sorter = np.argsort(y_local)
        y_sorted = y_local[sorter]
        w_sorted = w_local[sorter]
        
        cum_weights = np.cumsum(w_sorted)
        cum_weights /= cum_weights[-1]
        
        idx_q = np.searchsorted(cum_weights, q)
        
        if idx_q == 0: val = y_sorted[0]
        elif idx_q >= len(y_sorted): val = y_sorted[-1]
        else: val = y_sorted[idx_q]
            
        y_smooth.append(val)

    return x_grid, np.array(y_smooth)


# ==============================================================================
# 4. VISUALIZADOR INSTITUCIONAL (CLASE PRINCIPAL)
# ==============================================================================

class InstitutionalVisualizer:
    def __init__(self, metrics_map, trade_col):
        self.metrics = metrics_map
        self.trade_col = trade_col

    def _robust_scale(self, ax, data, axis='y', padding=0.10):
        try:
            dmin, dmax = np.nanpercentile(data, [1, 99])
            span = dmax - dmin
            if span == 0: span = 1
            if axis == 'y': ax.set_ylim(dmin - span*padding, dmax + span*padding)
            elif axis == 'x': ax.set_xlim(dmin - span*padding, dmax + span*padding)
        except: pass

    # --------------------------------------------------------------------------
    # PÁGINA 1: PORTADA Y KPIs
    # --------------------------------------------------------------------------
    def plot_cover(self, pdf, filename, df):
        plt.figure(figsize=(11.69, 8.27))
        
        plt.text(0.5, 0.70, "ALGORITHMIC STRATEGY AUDIT", ha='center', fontsize=26, fontweight='light', color=COLORS['text'])
        plt.text(0.5, 0.65, "SENSITIVITY & ROBUSTNESS REPORT", ha='center', fontsize=10, color=COLORS['subtext'])
        
        # Extracción segura de valores máximos (ahora garantizados como float)
        best_roi = df[self.metrics['ROI']].max()
        best_dd = df[self.metrics['DD']].min() if self.metrics['DD'] in df.columns else 0.0
        
        # Formateo KPI (El error ocurría aquí si best_roi era str)
        plt.text(0.35, 0.5, f"{best_roi:+.2f}%", ha='center', fontsize=36, fontweight='medium', color=COLORS['up'])
        plt.text(0.35, 0.45, "PEAK RETURN (ROI)", ha='center', fontsize=9, color=COLORS['subtext'])
        
        plt.text(0.65, 0.5, f"{best_dd:.2f}%", ha='center', fontsize=36, fontweight='medium', color=COLORS['down'])
        plt.text(0.65, 0.45, "MIN DRAWDOWN", ha='center', fontsize=9, color=COLORS['subtext'])

        plt.text(0.5, 0.10, f"FILE: {filename}", ha='center', fontsize=8, color=COLORS['subtext'], style='italic')
        plt.axis('off')
        pdf.savefig()
        plt.close()

    # --------------------------------------------------------------------------
    # PÁGINA 2: DISTRIBUCIÓN
    # --------------------------------------------------------------------------
    def plot_market_regime_distribution(self, df, pdf):
        col = self.metrics['ROI']
        data = df[col]
        q1, q99 = data.quantile(0.01), data.quantile(0.99)
        data = data[(data >= q1) & (data <= q99)]
        
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fig.suptitle('PROFIT DISTRIBUTION DENSITY', fontsize=14, color=COLORS['text'], x=0.1, ha='left')
        
        sns.histplot(data, stat="density", bins=50, color=COLORS['area'], alpha=0.3, element="step", fill=True, edgecolor=None, ax=ax)
        
        mu, std = data.mean(), data.std()
        x = np.linspace(mu - 4*std, mu + 4*std, 1000)
        ax.plot(x, stats.norm.pdf(x, mu, std), color=COLORS['text'], linewidth=1, alpha=0.6, linestyle='--')
        
        ax.axvline(mu, color=COLORS['gold'], linewidth=1.5, label=f'Mean: {mu:.2f}%')
        
        ax.set_xlabel('Net Profit %')
        ax.set_ylabel('Probability')
        ax.legend(frameon=False, loc='upper right')
        ax.grid(True, which='major', color=COLORS['grid'], alpha=0.3)
        sns.despine(left=True, bottom=True)
        
        pdf.savefig(bbox_inches='tight')
        plt.close()

    # --------------------------------------------------------------------------
    # PÁGINA DE ANÁLISIS PROFUNDO (SCATTER FIX + QUANTILES)
    # --------------------------------------------------------------------------
    def plot_deep_dive_analysis(self, df, param, pdf):
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.2)
        
        fig.suptitle(f'PARAMETER SENSITIVITY: {param}', fontsize=16, color=COLORS['text'], x=0.5, y=0.95)
        fig.text(0.5, 0.92, 'White Line: 75th Percentile Potential | Band: Top-Tier Variance (50th-90th)', 
                 ha='center', fontsize=9, color=COLORS['subtext'])

        configs = [
            (0, 0, 'ROI', self.metrics['ROI'], 'Net Profit %', COLORS['up']),
            (0, 1, 'DD', self.metrics['DD'], 'Max Drawdown %', COLORS['down']),
            (1, 0, 'PF', self.metrics['PF'], 'Profit Factor', COLORS['area']),
            (1, 1, 'SQN', self.metrics['SQN'], 'SQN Score', COLORS['gold'])
        ]

        x = df[param].values
        trades = df[self.trade_col].values if self.trade_col in df.columns else np.ones_like(x)
        
        # Tamaños dinámicos logarítmicos
        sizes = np.log1p(trades)
        sizes = (sizes / (sizes.max() + 1e-9)) * 40 + 5 

        for r, c, key, col, label, color in configs:
            ax = fig.add_subplot(gs[r, c])
            if col not in df.columns: continue
            
            y = df[col].values
            
            # 1. SCATTER PLOT SEGURO
            # Máscara de validación total para alinear dimensiones X, Y, Sizes
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sizes)
            
            if np.sum(mask) > 0:
                ax.scatter(x[mask], y[mask], c=color, alpha=0.15, s=sizes[mask], edgecolors='none', zorder=1)

            # 2. CÁLCULO DE CURVAS
            xs, y_50 = weighted_quantile_smooth(x, y, trades, q=0.50)
            _, y_90 = weighted_quantile_smooth(x, y, trades, q=0.90)
            
            if xs is not None:
                # Banda Top Tier
                ax.fill_between(xs, y_50, y_90, color=color, alpha=0.1, linewidth=0, zorder=2)
                
                # 3. LÍNEA DE POTENCIAL & OPTIMIZACIÓN
                if key == 'DD':
                    # Para DD: buscamos percentil bajo (25)
                    _, y_75 = weighted_quantile_smooth(x, y, trades, q=0.25)
                    idx_best = np.argmin(y_75)
                else:
                    # Para Profit: percentil alto (75)
                    _, y_75 = weighted_quantile_smooth(x, y, trades, q=0.75)
                    # Score: Retorno alto pero estable (penaliza ancho de banda)
                    spread = y_90 - y_50
                    score = y_75 - (spread * 0.2) 
                    idx_best = np.argmax(score)

                ax.plot(xs, y_75, color=COLORS['trend'], linewidth=1.5, alpha=0.9, zorder=4)

                # 4. MARCADOR DE PUNTO ÓPTIMO
                bx, by = xs[idx_best], y_75[idx_best]
                ax.scatter(bx, by, color=COLORS['bg'], s=60, edgecolors=COLORS['text'], linewidth=1.5, zorder=5)
                ax.text(bx, by + (by*0.05 if key!='DD' else -by*0.05), f"{bx:.2f}", 
                        ha='center', va='bottom' if key!='DD' else 'top', 
                        fontsize=8, color=COLORS['text'], fontweight='bold')

            # Estética
            ax.set_title(label, color=COLORS['text'], fontsize=10, fontweight='medium', loc='left', pad=10)
            ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=0.5, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(COLORS['grid'])
            ax.spines['bottom'].set_color(COLORS['grid'])
            
            self._robust_scale(ax, df[param], 'x')
            self._robust_scale(ax, df[col], 'y')

        pdf.savefig(bbox_inches='tight')
        plt.close()

    # --------------------------------------------------------------------------
    # PÁGINA DE TABLA
    # --------------------------------------------------------------------------
    def plot_parameter_table(self, df, param, pdf):
        unique_vals = df[param].nunique()
        data = df.copy()
        x_col = param
        
        if unique_vals > 12:
            try: data[f'{param}_BIN'] = pd.qcut(data[param], q=8, duplicates='drop'); x_col = f'{param}_BIN'
            except: 
                try: data[f'{param}_BIN'] = pd.cut(data[param], bins=8); x_col = f'{param}_BIN'
                except: pass
        
        stats = data.groupby(x_col, observed=True).agg({
            self.metrics['ROI']: 'mean',
            self.metrics['PF']: 'mean',
            self.metrics['SQN']: 'mean'
        }).reset_index()
        
        if self.metrics['DD'] in df.columns:
            dd_stats = data.groupby(x_col, observed=True)[self.metrics['DD']].mean().reset_index()
            stats = pd.merge(stats, dd_stats, on=x_col)
        else: 
            stats['DD'] = 0

        table_vals = []
        for _, row in stats.iterrows():
            label = str(row[x_col]).replace('(','').replace(']','').replace(', ',' - ')
            roi = row[self.metrics['ROI']]
            dd = row[self.metrics['DD']] if self.metrics['DD'] in df.columns else 0
            table_vals.append([label, f"{roi:.2f}%", f"{dd:.2f}%", f"{row[self.metrics['PF']]:.2f}", f"{row[self.metrics['SQN']]:.2f}"])

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis('off')
        
        cols = ['Parameter Range', 'Avg ROI', 'Avg DD', 'Avg PF', 'Avg SQN']
        t = ax.table(cellText=table_vals, colLabels=cols, loc='center', cellLoc='center', 
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
        
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 2)
        
        for key, cell in t.get_celld().items():
            cell.set_linewidth(0)
            if key[0] == 0:
                cell.set_facecolor(COLORS['table_h'])
                cell.set_text_props(color=COLORS['text'], weight='bold')
            else:
                cell.set_facecolor(COLORS['table_r'] if key[0]%2 else COLORS['bg'])
                cell.set_text_props(color=COLORS['subtext'])
                if key[1] == 1: 
                    val = float(table_vals[key[0]-1][1].strip('%'))
                    if val > 0: cell.set_text_props(color=COLORS['up'])
                if key[1] == 2: 
                    cell.set_text_props(color=COLORS['down'])

        pdf.savefig(bbox_inches='tight')
        plt.close()

    # --------------------------------------------------------------------------
    # PÁGINA FINAL: CORRELACIÓN
    # --------------------------------------------------------------------------
    def plot_correlation_matrix(self, df, params, pdf):
        valid_metrics = [m for m in self.metrics.values() if m in df.columns]
        cols = params + valid_metrics
        corr = df[cols].corr()
        final_corr = corr.loc[params, valid_metrics]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title('FACTOR CORRELATION', fontsize=12, color=COLORS['text'], pad=20)
        
        sns.heatmap(final_corr, annot=True, cmap=sns.diverging_palette(240, 10, as_cmap=True), center=0, fmt=".2f",
                    linewidths=0.5, linecolor=COLORS['bg'], square=True,
                    cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
        
        pdf.savefig()
        plt.close()


# ==============================================================================
# 5. EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    print("\n" + "━"*60)
    print("   QUANTITATIVE STRATEGY AUDIT | v21.0")
    print("━"*60 + "\n")
    
    file_path = ""
    if len(sys.argv) > 1: 
        file_path = sys.argv[1]
    else:
        files = [f for f in os.listdir('.') if f.endswith('.csv') or f.endswith('.xlsx')]
        if files: 
            file_path = files[0]
            print(f"📂 SOURCE: {file_path}")
        else: 
            file_path = input(">> DATA FILE PATH: ").strip().strip('"').strip("'").strip()

    if not os.path.exists(file_path): 
        print("❌ FILE NOT FOUND.")
        return

    loader = QuantLoader()
    df = loader.load_data(file_path)
    if df is None: 
        print("❌ INVALID DATA OR MISSING CRITICAL METRICS.")
        return
    
    params = loader.get_params(df)
    print(f"\n📊 PARAMETERS DETECTED: {len(params)}")

    out_pdf = f"STRATEGY_AUDIT_{os.path.splitext(os.path.basename(file_path))[0]}.pdf"
    viz = InstitutionalVisualizer(loader.TARGET_METRICS, loader.TRADE_COL)
    
    print("\n⚙️  CALCULATING QUANTILE REGRESSIONS...")
    
    try:
        with PdfPages(out_pdf) as pdf:
            viz.plot_cover(pdf, os.path.basename(file_path), df)
            viz.plot_market_regime_distribution(df, pdf)
            viz.plot_correlation_matrix(df, params, pdf)
            
            total = len(params)
            for i, param in enumerate(params):
                sys.stdout.write(f"\r   🔹 Analyzing: {param} [{i+1}/{total}]")
                sys.stdout.flush()
                viz.plot_deep_dive_analysis(df, param, pdf)
                viz.plot_parameter_table(df, param, pdf)
                
        print(f"\n\n✅ REPORT GENERATED: {out_pdf}")
        
        try:
            if os.name == 'nt': os.startfile(out_pdf)
            elif sys.platform == 'darwin': os.system(f'open "{out_pdf}"')
            else:
                import subprocess
                subprocess.run(['xdg-open', out_pdf], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: 
            print(f"ℹ️  Open manually: {out_pdf}")
        
    except Exception as e:
        print(f"\n❌ EXECUTION ERROR: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()