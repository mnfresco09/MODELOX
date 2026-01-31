"""
================================================================================
VISUAL/CSV_A_EXCEL.PY - Conversor CSV a Excel con formato Dashboard
================================================================================
Script para convertir archivos CSV de resultados a Excel con el mismo formato
profesional que genera ejecutar.py al finalizar la optimización.

USO:
    python csv_a_excel.py
    (Luego arrastrar el CSV cuando se solicite)

    O directamente:
    python csv_a_excel.py /ruta/al/archivo.csv

Autor: Sistema MODELOX
================================================================================
"""

import os
import sys
import re
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en el path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual.excel import convertir_resumen_csv_a_excel


def extraer_info_nombre_csv(csv_path: str) -> dict:
    """
    Intenta extraer activo, estrategia y timeframe del nombre del archivo CSV.
    
    Patrones comunes:
    - RESUMEN.csv
    - RESUMEN_BTC_ID3_15m.csv
    - resultados_GOLD_ESTRATEGIA_1h.csv
    """
    filename = os.path.basename(csv_path).upper()
    filename_sin_ext = os.path.splitext(filename)[0]
    
    info = {
        "activo": None,
        "estrategia": "UNKNOWN",
        "timeframe": None,
    }
    
    # Lista de activos conocidos
    activos_conocidos = ["BTC", "GOLD", "NASDAQ", "SP500", "ETH", "EUR", "USD"]
    
    # Lista de timeframes conocidos
    timeframes_conocidos = ["1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W"]
    
    # Buscar activo
    for activo in activos_conocidos:
        if activo in filename_sin_ext:
            info["activo"] = activo
            break
    
    # Buscar timeframe
    for tf in timeframes_conocidos:
        if tf in filename_sin_ext or f"_{tf}" in filename_sin_ext:
            info["timeframe"] = tf.lower()
            break
    
    # Buscar estrategia (patrones como ID3, ID4, etc.)
    match_estrategia = re.search(r'(ID\d+[A-Z0-9_.-]*)', filename_sin_ext, re.IGNORECASE)
    if match_estrategia:
        info["estrategia"] = match_estrategia.group(1)
    else:
        # Intentar extraer nombre después del activo
        partes = filename_sin_ext.replace("RESUMEN", "").replace("_", " ").split()
        for parte in partes:
            if parte not in activos_conocidos and parte not in timeframes_conocidos:
                if len(parte) > 2:
                    info["estrategia"] = parte
                    break
    
    return info


def limpiar_ruta(ruta: str) -> str:
    """
    Limpia la ruta de caracteres especiales que pueden venir al arrastrar archivos.
    """
    ruta = ruta.strip()
    # Quitar comillas si las hay
    ruta = ruta.strip('"').strip("'")
    # Quitar espacios escapados en macOS
    ruta = ruta.replace("\\ ", " ")
    return ruta


def main():
    print("\n" + "=" * 70)
    print("   CSV A EXCEL - Conversor con formato Dashboard MODELOX")
    print("=" * 70 + "\n")
    
    # Obtener ruta del CSV
    if len(sys.argv) > 1:
        csv_path = " ".join(sys.argv[1:])
    else:
        print("📂 Arrastra aquí el archivo CSV que deseas convertir a Excel:")
        print("   (O escribe la ruta completa y presiona Enter)")
        print()
        csv_path = input(">>> ").strip()
    
    # Limpiar la ruta
    csv_path = limpiar_ruta(csv_path)
    
    # Validar que existe
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: No se encontró el archivo: {csv_path}")
        sys.exit(1)
    
    # Validar que es un CSV
    if not csv_path.lower().endswith(".csv"):
        print(f"\n❌ ERROR: El archivo debe ser .csv, recibido: {csv_path}")
        sys.exit(1)
    
    print(f"\n📄 Archivo CSV: {csv_path}")
    
    # Extraer información del nombre
    info = extraer_info_nombre_csv(csv_path)
    
    # Intentar detectar estrategia desde el contenido del CSV
    try:
        import pandas as pd
        df_preview = pd.read_csv(csv_path, nrows=1)
        cols_upper = [c.upper() for c in df_preview.columns]
        
        # Buscar columna strategy/estrategia
        for col in df_preview.columns:
            if col.upper() in ["STRATEGY", "ESTRATEGIA"]:
                val = str(df_preview[col].iloc[0]).strip()
                if val and val.upper() != "NAN":
                    info["estrategia"] = val.upper()
                    break
    except Exception:
        pass
    
    print(f"   • Activo:     {info['activo'] or 'UNKNOWN'}")
    print(f"   • Estrategia: {info['estrategia']}")
    print(f"   • Timeframe:  {info['timeframe'] or 'UNKNOWN'}")
    
    # El Excel se guardará en el mismo directorio que el CSV
    output_dir = os.path.dirname(csv_path) or "."
    
    print("\n" + "=" * 50)
    print("   Procesando conversión...")
    print("=" * 50 + "\n")
    
    try:
        excel_path = convertir_resumen_csv_a_excel(
            csv_path=csv_path,
            strategy_name=info["estrategia"],
            activo=info["activo"],
            timeframe=info["timeframe"],
            output_dir=output_dir,
        )
        
        print(f"✅ ÉXITO: Excel generado correctamente")
        print(f"\n📊 Archivo Excel: {excel_path}")
        print(f"📁 Ubicación: {os.path.dirname(excel_path) or '.'}")
        
    except Exception as e:
        print(f"\n❌ ERROR durante la conversión: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✨ Conversión completada.")


if __name__ == "__main__":
    main()
