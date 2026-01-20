import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import HistGradientBoostingRegressor
import os

# ==============================================================================
# 1. ESTÉTICA DE ALTA DENSIDAD (QUANT DARK)
# ==============================================================================
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#05070A',
    'axes.facecolor': '#0B0F14',
    'axes.edgecolor': '#1E2329',
    'grid.color': '#1E2329',
    'text.color': '#D1D5DB',
    'font.family': 'sans-serif',
    'font.size': 9
})

# ==============================================================================
# 2. CEREBRO DE DETECCIÓN DE ESTRUCTURA (ANTI-MÉTRICAS)
# ==============================================================================
class ModeloxCausalScanner:
    def __init__(self):
        # Lista negra definitiva de métricas que no deben analizarse como parámetros
        self.EXCLUDE = {
            'TRIAL', 'SCORE', 'ESTRATEGIA', 'ROI', 'ROI_PCT', 'WINRATE', 'WINRATE_PCT', 
            'DRAWDOWN', 'MAX_DD_PCT', 'EXPECTATIVA', 'EXPECTANCY', 'SQN', 'ESTABILIDAD', 
            'SHARPE', 'SORTINO', 'PROFIT_FACTOR', 'TRADES_DIA', 'PNL_NETO', 'NET_PNL',
            'NUM_LONGS', 'NUM_SHORTS', 'TOTAL_TRADES', 'N_TRADES', 'DURATION', 'FECHA',
            'SALDO', 'SALDO_ACTUAL', 'RACHA_GANADORA', 'RACHA_PERDEDORA'
        }

    def load_and_classify(self, path):
        path = path.strip().replace("'", "").replace('"', "")
        df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
        
        # Detectar cabecera real en Excels sucios
        if df.columns[0].startswith('Unnamed') or df.iloc[0].isnull().all():
            df = pd.read_excel(path, header=1)
            
        df.columns = [str(c).strip().upper().replace(' ', '_').replace('%', 'PCT') for c in df.columns]
        df = df.fillna(0)
        
        target = 'ROI_PCT' if 'ROI_PCT' in df.columns else ('ROI' if 'ROI' in df.columns else None)
        
        # Identificar solo parámetros de entrada (INPUTS)
        indicators = []
        for col in df.select_dtypes(include=[np.number]).columns:
            # Si el nombre de la columna contiene alguna palabra de la lista negra, fuera
            if any(metric in col for metric in self.EXCLUDE): continue
            if col.startswith('__'): continue
            if df[col].nunique() > 1:
                indicators.append(col)
        
        return df, indicators, target

# ==============================================================================
# 3. MOTOR DE INFERENCIA CAUSAL (ALE & ROBUSTNESS)
# ==============================================================================
class QuantumCausalEngine:
    @staticmethod
    def get_robust_zone(x, y, pct=0.10):
        """Busca la meseta de máxima estabilidad (Top performance + baja pendiente)"""
        threshold = np.max(y) - (np.max(y) - np.min(y)) * pct
        top_idx = np.where(y >= threshold)[0]
        if len(top_idx) < 2: return None, None
        return x[top_idx[0]], x[top_idx[-1]]

    def analyze(self, df, indicators, target, path):
        # Usamos HistGradientBoosting (la IA más rápida y precisa para datos tabulares)
        model = HistGradientBoostingRegressor(max_iter=300, max_depth=7, l2_regularization=1.5, random_state=42)
        model.fit(df[indicators], df[target])
        
        output_pdf = f"CAUSAL_AUDIT_{os.path.basename(path).split('.')[0]}.pdf"
        
        with PdfPages(output_pdf) as pdf:
            # PÁGINA DE PORTADA
            fig_cover = plt.figure(figsize=(11.69, 8.27))
            plt.axis('off')
            plt.text(0.5, 0.7, "MODELOX CAUSAL ANALYST", color='#58A6FF', fontsize=38, ha='center', weight='bold')
            plt.text(0.5, 0.62, "Desacoplo de Ruido Estocástico e Identificación de Mesetas de Robustez", color='white', fontsize=14, ha='center')
            plt.text(0.5, 0.4, f"Dataset: {os.path.basename(path)} | N: {len(df)}", color='#8B949E', fontsize=11, ha='center')
            pdf.savefig(fig_cover); plt.close()

            for ind in indicators:
                print(f"🧬 Desacoplando Causalidad: {ind}...")
                
                # Simulador de Impacto ALE (Accumulated Local Effects)
                x_grid = np.linspace(df[ind].min(), df[ind].max(), 100)
                temp_df = pd.DataFrame([df[indicators].median()] * 100, columns=indicators)
                temp_df[ind] = x_grid
                y_isolated = model.predict(temp_df)
                
                z_start, z_end = self.get_robust_zone(x_grid, y_isolated)
                
                # GRAFICACIÓN PROFESIONAL
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                
                # Datos crudos (Sutiles)
                ax.scatter(df[ind], df[target], color='#161B22', alpha=0.3, s=12, label='Ruido Estocástico (Trials)')
                
                # Sombreado de Zona Robusta
                if z_start is not None:
                    ax.axvspan(z_start, z_end, color='#238636', alpha=0.15, label='MESETA DE ROBUSTEZ (Safe Zone)')
                    ax.axvline(z_start, color='#238636', linestyle='--', alpha=0.4, linewidth=1)
                    ax.axvline(z_end, color='#238636', linestyle='--', alpha=0.4, linewidth=1)
                
                # Línea de Impacto Puro
                ax.plot(x_grid, y_isolated, color='#58A6FF', linewidth=3.5, label='Efecto Causal (IA Desacoplada)', zorder=10)
                
                # Decoración técnica
                ax.set_title(f"DETECCIÓN DE IMPACTO: {ind}", fontsize=20, color='#58A6FF', loc='left', pad=30, weight='bold')
                ax.set_xlabel(f"Configuración del Parámetro {ind}", color='#8B949E')
                ax.set_ylabel(f"Impacto Neto en {target}", color='#8B949E')
                ax.grid(True, alpha=0.05)
                
                # Cuadro de Análisis IA
                peak = x_grid[np.argmax(y_isolated)]
                coverage = (z_end - z_start) / (x_grid[-1] - x_grid[0]) if z_start is not None else 0
                
                diag = "DIAGNÓSTICO CAUSAL DE IA:\n"
                diag += "───────────────────────\n"
                diag += f"• Pico Teórico: {peak:.4g}\n"
                diag += f"• Rango de Robustez: [{z_start:.4g}, {z_end:.4g}]\n"
                diag += f"• Cobertura Segura: {coverage*100:.1f}%\n\n"
                
                if coverage > 0.35:
                    diag += "RESULTADO: ALTAMENTE ROBUSTO. El indicador permite errores de configuración sin destruir el ROI."
                elif coverage > 0:
                    diag += "RESULTADO: SENSIBLE. La zona de beneficio es clara pero requiere precisión."
                else:
                    diag += "RESULTADO: RUIDO / INESTABLE. No se recomienda optimizar este parámetro."

                plt.figtext(0.14, 0.05, diag, color='white', family='monospace', fontsize=11,
                            bbox=dict(facecolor='#0D1117', edgecolor='#58A6FF', boxstyle='round,pad=1', alpha=0.9))

                ax.legend(facecolor='#05070A', edgecolor='#1E2329', loc='upper right')
                pdf.savefig(fig); plt.close()

        print(f"\n✅ ANÁLISIS COMPLETADO: {output_pdf}")

if __name__ == "__main__":
    print("\n" + "="*60 + "\n   MODELOX CAUSAL ANALYST v7.0\n" + "="*60)
    archivo = input("\n👉 ARRASTRA EL ARCHIVO Y PULSA ENTER: ").strip().replace("'", "").replace('"', "")
    
    if os.path.exists(archivo):
        scanner = ModeloxCausalScanner()
        df, inds, target = scanner.load_and_classify(archivo)
        if target and inds:
            print(f"🚀 Detectados {len(inds)} parámetros reales. Entrenando modelo profundo...")
            QuantumCausalEngine().analyze(df, inds, target, archivo)
        else:
            print("❌ Error: No se pudo separar métricas de parámetros.")
    else:
        print("❌ Archivo no encontrado.")