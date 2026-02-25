import pandas as pd
import sys
import os

# Intentar importar rich para una interfaz más bonita
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import box
    from rich.columns import Columns
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

try:
    import xlsxwriter
    HAS_XLSX = True
except ImportError:
    HAS_XLSX = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.table import Table as MatproTable
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

def analizar_variacion():
    if HAS_RICH:
        console.print("\n[bold blue]=== ANALIZADOR DE VARIACIÓN OHLC PRO ===[/]", justify="center")
    else:
        print("\n=== ANALIZADOR DE VARIACIÓN OHLC PRO ===\n")
    
    # 1. Obtener ruta del archivo
    if len(sys.argv) > 1:
        path = sys.argv[1].strip().strip("'").strip('"')
    else:
        if HAS_RICH:
            path = Prompt.ask("\nArrastra el archivo aquí y pulsa Enter").strip().strip("'").strip('"')
        else:
            path = input("\nArrastra el archivo aquí y pulsa Enter: ").strip().strip("'").strip('"')
    
    if not os.path.exists(path):
        if HAS_RICH:
            console.print(f"[red]Error: El archivo no existe en {path}[/]")
        else:
            print(f"Error: El archivo no existe en {path}")
        return

    # 2. Cargar datos
    try:
        if HAS_RICH:
            with console.status("[bold green]Cargando datos..."):
                df = cargar_dataframe(path)
        else:
            print(f"Cargando {os.path.basename(path)}...")
            df = cargar_dataframe(path)
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]Error al cargar el archivo: {e}[/]")
        else:
            print(f"Error al cargar el archivo: {e}")
        return

    # 3. Preparar columnas
    df.columns = [c.upper() for c in df.columns]
    
    # Buscar tiempo para permitir resampling
    time_col = next((c for c in ['TIMESTAMP', 'DATETIME', 'TIME', 'DATE', 'FECHA'] if c in df.columns), None)
    if not time_col:
        time_col = next((c for c in df.columns if 'TIME' in c or 'DATE' in c), None)

    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col], dayfirst=True)
            df.set_index(time_col, inplace=True)
            df.sort_index(inplace=True)
        except:
            time_col = None

    # Validar columnas necesarias
    needed = ['OPEN', 'CLOSE']
    if not all(c in df.columns for c in needed):
        msg = f"Error: Faltan columnas OPEN o CLOSE. Encontradas: {list(df.columns)}"
        if HAS_RICH: console.print(f"[red]{msg}[/]")
        else: print(msg)
        return

    # 3b. Elegir Rango de Datos (Todo vs Reducido)
    if time_col:
        rango_op = "1"
        if HAS_RICH:
            rango_op = Prompt.ask(
                "\n[bold cyan]Rango de Datos:[/]\n  1. [bold yellow]TODO[/] (Incompleto + Completo)\n  2. [bold green]REDUCIDO[/] (Solo desde 2020)\nSelección",
                choices=["1", "2"], default="1"
            )
        else:
            print("\nRango de Datos:\n  1. TODO\n  2. REDUCIDO (desde 2020)")
            rango_op = input("Selección (1/2) [1]: ").strip() or "1"
        
        if rango_op == "2":
            df = df[df.index >= '2020-01-01']
            if HAS_RICH:
                console.print(f"[bold yellow]→ Filtro aplicado:[/] Datos desde {df.index.min().date()} hasta {df.index.max().date()}")
            else:
                print(f"→ Filtro aplicado: Datos desde {df.index.min().date()} hasta {df.index.max().date()}")

    # 3c. Verificador de Cálculos (Audit Log)
    show_audit = False
    if HAS_RICH:
        show_audit = Prompt.ask("\n¿Quieres ver una [bold cyan]AUDITORÍA DE CÁLCULOS[/] (verificación paso a paso)?", choices=["S", "N"], default="N").upper() == "S"
    else:
        show_audit = input("\n¿Quieres ver una AUDITORÍA DE CÁLCULOS (S/N) [N]? ").strip().upper() == "S"

    # 4. Elegir Timeframe o Modo Masivo
    timeframes_to_process = []
    if HAS_RICH:
        opcion = Prompt.ask(
            "Timeframe (ej: 1H, 15min) o escribe [bold yellow]'BATCH'[/] para analizar todos (5m, 15m, 30m, 45m, 1h, 4h)", 
            default=""
        ).strip().upper()
    else:
        opcion = input("Timeframe o 'BATCH' para todos (5m, 15m, 30m, 45m, 1h, 4h): ").strip().upper()

    is_batch = opcion == 'BATCH'
    if is_batch:
        timeframes_to_process = ['5min', '15min', '30min', '45min', '1h', '4h']
    else:
        timeframes_to_process = [opcion] if opcion else [None]

    results_all_tf = {}

    for tf in timeframes_to_process:
        df_tf = df.copy()
        current_tf_name = tf or 'Original'
        
        if tf and time_col:
            try:
                if HAS_RICH:
                    with console.status(f"[bold green]Procesando {current_tf_name}..."):
                        df_tf = resamplear_ohlc(df_tf, tf)
                else:
                    print(f"Procesando {current_tf_name}...")
                    df_tf = resamplear_ohlc(df_tf, tf)
            except Exception as e:
                msg = f"Error resampleando {current_tf_name}: {e}"
                if HAS_RICH: console.print(f"[red]{msg}[/]")
                else: print(msg)
                continue
        
        # Cálculos de variaciones
        df_tf['VAR_PCT'] = (df_tf['CLOSE'] - df_tf['OPEN']) / df_tf['OPEN'] * 100
        df_tf['VAR_ABS'] = (df_tf['CLOSE'] - df_tf['OPEN'])
        
        # Calcular estadísticas
        stats_gen = calcular_stats(df_tf)
        
        # Calcular estadísticas anuales
        stats_years = {}
        if time_col is not None:
            for year, group in df_tf.groupby(df_tf.index.year):
                stats_years[year] = calcular_stats(group)
        
        results_all_tf[current_tf_name] = {
            'general': stats_gen,
            'years': stats_years,
            'df': df_tf
        }

    # 5. Mostrar resultados en terminal
    if is_batch:
        for tf_name, data in results_all_tf.items():
            if HAS_RICH:
                mostrar_resultados_completo_rich(path, data['df'], tf_name, time_col is not None)
            else:
                mostrar_resultados_simple(path, data['df'], tf_name)
    else:
        # Recuperar el único resultado
        tf_name = next(iter(results_all_tf))
        data = results_all_tf[tf_name]
        if HAS_RICH:
            mostrar_resultados_completo_rich(path, df_tf, tf_name, time_col is not None)
        else:
            mostrar_resultados_simple(path, df_tf, tf_name)

    # 5b. Auditoría si se solicitó
    if show_audit and results_all_tf:
        # Tomar el primer DF disponible para auditar (o el único si no es batch)
        first_tf = next(iter(results_all_tf))
        target_df = results_all_tf[first_tf]['df']
        if target_df is None: # Si es batch no guardamos todos los DFs por memoria, pero para el audit necesitamos uno
             target_df = df_tf # El último procesado
        
        if HAS_RICH:
            verificar_muestras_rich(target_df, first_tf)

    # 6. Exportar a Excel
    if HAS_XLSX:
        # Guardar en la raíz (donde se ejecuta el script)
        export_path = os.path.join(os.getcwd(), f"Análisis_Variación_{os.path.basename(path).split('.')[0]}.xlsx")
        try:
            if HAS_RICH:
                with console.status("[bold blue]Generando Excel profesional..."):
                    exportar_a_excel(export_path, results_all_tf, os.path.basename(path))
                console.print(f"\n[bold green]✅ Excel generado con éxito:[/][white] {export_path}[/]\n")
            else:
                exportar_a_excel(export_path, results_all_tf, os.path.basename(path))
                print(f"\nExcel generado: {export_path}\n")
        except Exception as e:
            msg = f"Error al generar Excel: {e}"
            if HAS_RICH: console.print(f"[red]{msg}[/]")
            else: print(msg)
    
    # 7. Exportar a PDF (Opcional)
    if HAS_PDF:
        pdf_path = os.path.join(os.getcwd(), f"Análisis_Variación_{os.path.basename(path).split('.')[0]}.pdf")
        try:
            if HAS_RICH:
                with console.status("[bold magenta]Generando PDF institucional..."):
                    exportar_a_pdf(pdf_path, results_all_tf, os.path.basename(path))
                console.print(f"[bold magenta]✅ PDF generado con éxito:[/] [white]{pdf_path}[/]\n")
            else:
                exportar_a_pdf(pdf_path, results_all_tf, os.path.basename(path))
                print(f"PDF generado: {pdf_path}")
        except Exception as e:
            msg = f"Error al generar PDF: {e}"
            if HAS_RICH: console.print(f"[red]{msg}[/]")
            else: print(msg)
    else:
        if HAS_RICH: console.print("[yellow]Aviso: matplotlib no instalado. No se generará PDF.[/]")

def cargar_dataframe(path):
    if path.endswith('.csv'):
        with open(path, 'r') as f:
            first_line = f.readline()
            sep = ';' if ';' in first_line else ','
        return pd.read_csv(path, sep=sep)
    elif path.endswith(('.feather', '.ftr', '.arrow')):
        return pd.read_feather(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        raise ValueError("Formato no soportado")

def resamplear_ohlc(df, tf):
    agg_logic = {'OPEN': 'first', 'CLOSE': 'last'}
    if 'HIGH' in df.columns: agg_logic['HIGH'] = 'max'
    if 'LOW' in df.columns: agg_logic['LOW'] = 'min'
    if 'VOLUME' in df.columns: agg_logic['VOLUME'] = 'sum'
    return df.resample(tf).apply(agg_logic).dropna()

def detectar_swings(df, window=3):
    """Detecta pivotes High/Low (ZigZag) y extrae los impulsos (swings) entre ellos."""
    if len(df) < window * 2 + 1: return []
    highs = df['HIGH'].values
    lows = df['LOW'].values
    pivots = []
    
    for i in range(window, len(df) - window):
        # Pivot High
        if all(highs[i] > highs[j] for j in range(i-window, i+window+1) if i != j):
            pivots.append({'idx': i, 'price': highs[i], 'type': 'HIGH'})
        # Pivot Low
        elif all(lows[i] < lows[j] for j in range(i-window, i+window+1) if i != j):
            pivots.append({'idx': i, 'price': lows[i], 'type': 'LOW'})
            
    if not pivots: return []
    
    # Filtrar para que alternen HIGH-LOW-HIGH...
    clean = []
    curr = pivots[0]
    for nxt in pivots[1:]:
        if nxt['type'] == curr['type']:
            if (curr['type'] == 'HIGH' and nxt['price'] > curr['price']) or \
               (curr['type'] == 'LOW' and nxt['price'] < curr['price']):
                curr = nxt
        else:
            clean.append(curr)
            curr = nxt
    clean.append(curr)
    
    swings = []
    for i in range(1, len(clean)):
        p1, p2 = clean[i-1], clean[i]
        swings.append({
            'duration': p2['idx'] - p1['idx'],
            'pct': (p2['price']/p1['price'] - 1) * 100,
            'type': 'UP' if p2['type'] == 'HIGH' else 'DOWN'
        })
    return swings

def calcular_stats(sub_df):
    pos = sub_df[sub_df['VAR_PCT'] > 0]
    neg = sub_df[sub_df['VAR_PCT'] < 0]
    neu = sub_df[sub_df['VAR_PCT'] == 0]
    
    # Anatomía de vela
    body = (sub_df['CLOSE'] - sub_df['OPEN']).abs()
    
    stats = {
        'm_pos': pos['VAR_PCT'].mean() if not pos.empty else 0,
        'm_neg': neg['VAR_PCT'].mean() if not neg.empty else 0,
        'a_pos': pos['VAR_ABS'].mean() if not pos.empty else 0,
        'a_neg': neg['VAR_ABS'].mean() if not neg.empty else 0,
        'max_pos': pos['VAR_PCT'].max() if not pos.empty else 0,
        'max_neg': neg['VAR_PCT'].min() if not neg.empty else 0,
        'count_pos': len(pos),
        'count_neg': len(neg),
        'count_neu': len(neu),
        'total': len(sub_df),
        'body_mean': body.mean()
    }

    if 'HIGH' in sub_df.columns and 'LOW' in sub_df.columns:
        upper_wick = sub_df['HIGH'] - sub_df[['OPEN', 'CLOSE']].max(axis=1)
        lower_wick = sub_df[['OPEN', 'CLOSE']].min(axis=1) - sub_df['LOW']
        total_wick = upper_wick + lower_wick
        stats['wick_mean'] = total_wick.mean()
        stats['wick_up_mean'] = upper_wick.mean()
        stats['wick_low_mean'] = lower_wick.mean()
        stats['range_mean'] = (sub_df['HIGH'] - sub_df['LOW']).mean()
        stats['body_vs_wick'] = (stats['body_mean'] / stats['wick_mean'] * 100) if stats['wick_mean'] > 0 else 0
        
        # --- Análisis de Swings ---
        swings = detectar_swings(sub_df)
        up_s = [s for s in swings if s['type'] == 'UP']
        dn_s = [s for s in swings if s['type'] == 'DOWN']
        
        stats['swing_dur_up'] = sum(s['duration'] for s in up_s) / len(up_s) if up_s else 0
        stats['swing_dur_dn'] = sum(s['duration'] for s in dn_s) / len(dn_s) if dn_s else 0
        stats['swing_pct_up'] = sum(s['pct'] for s in up_s) / len(up_s) if up_s else 0
        stats['swing_pct_dn'] = sum(s['pct'] for s in dn_s) / len(dn_s) if dn_s else 0
    
    return stats

def mostrar_resultados_completo_rich(path, df, tf, has_time):
    # Stats Generales
    general = calcular_stats(df)
    
    # Crear Tabla General
    table_gen = Table(title=f"RESUMEN GENERAL", box=box.ROUNDED, header_style="bold steel_blue3")
    table_gen.add_column("Métrica", style="bold cyan")
    table_gen.add_column("Velas Verdes (+)", justify="right", style="green3")
    table_gen.add_column("Velas Rojas (-)", justify="right", style="red3")
    
    table_gen.add_row("Media Variación (%)", f"{general['m_pos']:.4f}%", f"{general['m_neg']:.4f}%")
    table_gen.add_row("Media Variación Abs.", f"{general['a_pos']:.4f}", f"{general['a_neg']:.4f}")
    table_gen.add_row("Rango Medio (H-L)", f"{general.get('range_mean', 0):.2f}", f"{general.get('range_mean', 0):.2f}")
    table_gen.add_row("Máxima Variación (%)", f"{general['max_pos']:.4f}%", f"{general['max_neg']:.4f}%")
    table_gen.add_row("Cantidad Velas", f"{general['count_pos']}", f"{general['count_neg']}")
    table_gen.add_row("Frecuencia", f"{(general['count_pos']/general['total']*100):.1f}%", f"{(general['count_neg']/general['total']*100):.1f}%")

    # Crear Tabla Anual si hay tiempo
    table_years = None
    if has_time:
        table_years = Table(title="DESGLOSE POR AÑOS", box=box.ROUNDED, header_style="bold gold3")
        table_years.add_column("Año", justify="center", style="bold white")
        table_years.add_column("Media % (+)", justify="right", style="green3")
        table_years.add_column("Media % (-)", justify="right", style="red3")
        table_years.add_column("Puntos (+)", justify="right", style="pale_green3")
        table_years.add_column("Puntos (-)", justify="right", style="orange3")
        table_years.add_column("Cuerpo", justify="right", style="orchid")
        table_years.add_column("Mechas", justify="right", style="orchid")
        table_years.add_column("Ratio", justify="right", style="bold orchid")
        table_years.add_column("Rango H-L", justify="right", style="bold white")
        table_years.add_column("Frec. (+)", justify="right", style="grey78")
        table_years.add_column("Velas", justify="right", style="grey50")
        
        for year, group in df.groupby(df.index.year):
            s = calcular_stats(group)
            table_years.add_row(
                str(year),
                f"{s['m_pos']:.3f}%",
                f"{s['m_neg']:.3f}%",
                f"{s['a_pos']:.2f}",
                f"{s['a_neg']:.2f}",
                f"{s['body_mean']:.2f}",
                f"{s['wick_mean']:.2f}",
                f"{(s['body_mean']/s['wick_mean']):.2f}x" if s.get('wick_mean', 0) > 0 else "N/A",
                f"{s.get('range_mean', 0):.2f}",
                f"{(s['count_pos']/s['total']*100):.1f}%",
                str(s['total'])
            )

    # Crear Tabla de Anatomía (Cuerpo vs Mechas)
    table_anatomy = None
    if 'wick_mean' in general:
        table_anatomy = Table(title="ANATOMÍA & SWINGS (PROMEDIOS)", box=box.ROUNDED, header_style="bold orchid")
        table_anatomy.add_column("Parte / Swing", style="bold cyan")
        table_anatomy.add_column("Valor", justify="right", style="white")
        table_anatomy.add_column("% / Notas", justify="right", style="grey78")
        
        total_size = general['body_mean'] + general['wick_mean']
        p_body = (general['body_mean'] / total_size * 100) if total_size > 0 else 0
        p_wick = (general['wick_mean'] / total_size * 100) if total_size > 0 else 0
        
        table_anatomy.add_row("CUERPO", f"{general['body_mean']:.2f}", f"{p_body:.1f}%")
        table_anatomy.add_row("MECHAS (TOTAL)", f"{general['wick_mean']:.2f}", f"{p_wick:.1f}%")
        table_anatomy.add_row("RATIO C/M", f"{(general['body_mean']/general['wick_mean']):.2f}x" if general['wick_mean'] > 0 else "N/A", "")
        table_anatomy.add_section()
        table_anatomy.add_row("Duración Swing UP", f"{general.get('swing_dur_up',0):.1f} velas", f"{general.get('swing_pct_up',0):.2f}%")
        table_anatomy.add_row("Duración Swing DOWN", f"{general.get('swing_dur_dn',0):.1f} velas", f"{general.get('swing_pct_dn',0):.2f}%")


    # Mostrar todo
    print("\n")
    info_panel = Panel(
        f"ARCHIVO: [bold]{os.path.basename(path)}[/]\nTIMEFRAME: [bold yellow]{tf or 'ORIGINAL'}[/]",
        border_style="grey42",
        title="[bold]DATOS DE ENTRADA[/]"
    )
    console.print(info_panel, justify="center")
    
    from rich.columns import Columns
    # Organizar tablas: Resumen + Anatomía arriba, Desglose abajo
    top_row = [table_gen]
    if table_anatomy:
        top_row.append(table_anatomy)
    
    console.print(Columns(top_row, align="center"))
    
    if table_years:
        console.print("\n")
        console.print(table_years, justify="center")
    
    console.print(f"\n[dim]Total velas analizadas: {general['total']} | Neutras: {general['count_neu']}[/]\n", justify="center")

def mostrar_resultados_simple(path, df, tf):
    s = calcular_stats(df)
    print(f"\nResultados para: {os.path.basename(path)}")
    print(f"Timeframe: {tf or 'Original'}")
    print("-" * 40)
    print(f"GENERAL:")
    print(f"  Media (+): {s['m_pos']:.4f}% (n={s['count_pos']})")
    print(f"  Media (-): {s['m_neg']:.4f}% (n={s['count_neg']})")
    
    if hasattr(df.index, 'year'):
        print("-" * 40)
        print("POR AÑOS:")
        for year, group in df.groupby(df.index.year):
            ys = calcular_stats(group)
            print(f"  {year}: Pos={ys['m_pos']:.3f}% | Neg={ys['m_neg']:.3f}% | n={ys['total']}")
    print("-" * 40)

def exportar_a_excel(path, results, filename):
    workbook = xlsxwriter.Workbook(path)
    
    # Formatos
    f_title = workbook.add_format({'bold': True, 'font_size': 14, 'bg_color': '#1A1A2E', 'font_color': '#FFFFFF', 'align': 'center', 'border': 1})
    f_tf = workbook.add_format({'bold': True, 'font_size': 12, 'bg_color': '#0F3460', 'font_color': '#FFFFFF', 'align': 'center', 'border': 1})
    f_header = workbook.add_format({'bold': True, 'bg_color': '#16213E', 'font_color': '#FFFFFF', 'align': 'center', 'border': 1})
    f_label = workbook.add_format({'bold': True, 'bg_color': '#E3EAF6', 'align': 'left', 'border': 1})
    f_num = workbook.add_format({'num_format': '#,##0.00', 'align': 'right', 'border': 1})
    f_pct = workbook.add_format({'num_format': '0.000"%"', 'align': 'right', 'border': 1})
    f_green = workbook.add_format({'bg_color': '#E8F5E9', 'font_color': '#00897B', 'num_format': '0.000"%"', 'align': 'right', 'border': 1, 'bold': True})
    f_red = workbook.add_format({'bg_color': '#FFEBEE', 'font_color': '#C62828', 'num_format': '0.000"%"', 'align': 'right', 'border': 1, 'bold': True})

    ws = workbook.add_worksheet("ANÁLISIS")
    ws.set_column('A:K', 16)
    
    curr_row = 0
    
    # 1. TÍTULO Y RESUMEN COMPARATIVO
    ws.merge_range(curr_row, 0, curr_row, 7, f"ANALISIS VARIACION: {filename}", f_title)
    curr_row += 2
    
    ws.merge_range(curr_row, 0, curr_row, 9, "RESUMEN COMPARATIVO POR TEMPORALIDADES", f_header)
    curr_row += 1
    headers = ["TIMEFRAME", "MEDIA % (+)", "MEDIA % (-)", "DUR. SWING UP", "DUR. SWING DOWN", "RANGO (H-L)", "FREQ (+)", "RATIO C/M"]
    for col, h in enumerate(headers):
        ws.write(curr_row, col, h, f_header)
    curr_row += 1
    
    for tf_name, data in results.items():
        g = data['general']
        ws.write(curr_row, 0, tf_name, f_label)
        ws.write(curr_row, 1, g['m_pos'], f_green) 
        ws.write(curr_row, 2, g['m_neg'], f_red)
        ws.write(curr_row, 3, g.get('swing_dur_up', 0), f_num)
        ws.write(curr_row, 4, g.get('swing_dur_dn', 0), f_num)
        ws.write(curr_row, 5, g.get('range_mean', 0), f_num)
        ws.write(curr_row, 6, (g['count_pos']/g['total'])*100, f_pct)
        ws.write(curr_row, 7, g.get('body_mean', 0) / g.get('wick_mean', 1) if g.get('wick_mean', 0) > 0 else 0, f_num)
        curr_row += 1

    
    curr_row += 3 # Espacio antes de los detalles
    
    # 2. DETALLES POR TEMPORALIDAD
    for tf_name, data in results.items():
        # Cabecera de temporalidad
        ws.merge_range(curr_row, 0, curr_row, 10, f"DETALLE TEMPORALIDAD: {tf_name}", f_tf)
        curr_row += 1
        
        # Reservar espacio para Stats/Anatomía (se llenarán después con fórmulas)
        stats_start_row = curr_row
        curr_row += 7 # 1 header + 5 metrics + 1 spacer
        
        # --- Tabla Anual ---
        ws.write(curr_row, 0, "DESGLOSE ANUAL", f_header)
        curr_row += 1
        y_headers = ["AÑO", "MEDIA % (+)", "MEDIA % (-)", "PUNTOS (+)", "PUNTOS (-)", "RANGO H-L", "CUERPO", "MECHAS", "RATIO C/M", "VELAS (+)", "VELAS (-)", "TOTAL"]
        for col, h in enumerate(y_headers):
            ws.write(curr_row, col, h, f_header)
        
        # Anotar el inicio de los datos para las fórmulas (1-indexed para Excel)
        data_start_idx = curr_row + 2
        curr_row += 1
        
        if data['years']:
            for year, s in data['years'].items():
                ws.write(curr_row, 0, year, f_label)
                ws.write(curr_row, 1, s['m_pos'], f_green)
                ws.write(curr_row, 2, s['m_neg'], f_red)
                ws.write(curr_row, 3, s['a_pos'], f_num)
                ws.write(curr_row, 4, s['a_neg'], f_num)
                ws.write(curr_row, 5, s.get('range_mean', 0), f_num)
                ws.write(curr_row, 6, s['body_mean'], f_num)
                ws.write(curr_row, 7, s['wick_mean'], f_num)
                ws.write(curr_row, 8, s['body_mean']/s['wick_mean'] if s.get('wick_mean', 0) > 0 else 0, f_num)
                ws.write(curr_row, 9, s['count_pos'], f_num)
                ws.write(curr_row, 10, s['count_neg'], f_num)
                ws.write(curr_row, 11, s['total'], f_num)
                curr_row += 1
        
        data_end_idx = curr_row # El índice actual es justo después del último dato
        
        # Ahora rellenamos las mini-tablas superiores con fórmulas
        r = stats_start_row
        start, end = data_start_idx, data_end_idx - 1
        
        # Sub-tabla General (MÉTRICA)
        ws.write(r, 0, "MÉTRICA", f_header)
        ws.write(r, 1, "VERDES (+)", f_header)
        ws.write(r, 2, "ROJAS (-)", f_header)
        
        # Media %
        ws.write(r+1, 0, "MEDIA %", f_label)
        ws.write_formula(r+1, 1, f"=AVERAGE(B{start}:B{end})", f_green)
        ws.write_formula(r+1, 2, f"=AVERAGE(C{start}:C{end})", f_red)
        # Puntos
        ws.write(r+2, 0, "PUNTOS", f_label)
        ws.write_formula(r+2, 1, f"=AVERAGE(D{start}:D{end})", f_num)
        ws.write_formula(r+2, 2, f"=AVERAGE(E{start}:E{end})", f_num)
        # Rango
        ws.write(r+3, 0, "RANGO H-L", f_label)
        ws.write_formula(r+3, 1, f"=AVERAGE(F{start}:F{end})", f_num)
        ws.write_formula(r+3, 2, f"=AVERAGE(F{start}:F{end})", f_num)
        # Cantidad
        ws.write(r+4, 0, "CANTIDAD", f_label)
        ws.write_formula(r+4, 1, f"=SUM(J{start}:J{end})", f_num) # J es Velas (+)
        ws.write_formula(r+4, 2, f"=SUM(K{start}:K{end})", f_num) # K es Velas (-)
        # Frecuencia
        ws.write(r+5, 0, "FRECUENCIA", f_label)
        ws.write_formula(r+5, 1, f"=SUM(J{start}:J{end})/SUM(L{start}:L{end})", f_pct) # J/L (Pos/Total)
        ws.write_formula(r+5, 2, f"=SUM(K{start}:K{end})/SUM(L{start}:L{end})", f_pct) # K/L (Neg/Total)
            
        # Sub-tabla Anatomía (a la derecha)
        ws.write(r, 4, "ANATOMÍA / SWINGS", f_header)
        ws.write(r, 5, "VALOR", f_header)
        ws.write(r+1, 4, "RATIO C/M", f_label)
        ws.write_formula(r+1, 5, f"=AVERAGE(G{start}:G{end})/AVERAGE(H{start}:H{end})", f_num)
        ws.write(r+2, 4, "DUR. SWING UP", f_label)
        ws.write(r+2, 5, g.get('swing_dur_up', 0), f_num)
        ws.write(r+3, 4, "DUR. SWING DOWN", f_label)
        ws.write(r+3, 5, g.get('swing_dur_dn', 0), f_num)
        ws.write(r+4, 4, "IMPULSO UP %", f_label)
        ws.write(r+4, 5, g.get('swing_pct_up', 0)/100, f_pct)
        
        curr_row += 3 # Espacio entre temporalidades

    workbook.close()

def exportar_a_pdf(path, results, filename):
    # Colores institucionales (basados en visual/report.py)
    C_TITLE = '#0B1D40'
    C_HEADER = '#1B3B6F'
    C_LABEL = '#E3EAF6'
    C_GREEN = '#E8F5E9'
    C_RED = '#FFEBEE'
    C_WHITE = '#FFFFFF'
    
    with PdfPages(path) as pdf:
        # PÁGINA 1: PORTADA Y RESUMEN COMPARATIVO
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        # Título
        plt.text(0.5, 0.95, "REPORTE DE VARIACIÓN Y VOLATILIDAD", ha='center', fontsize=22, fontweight='bold', color=C_TITLE)
        plt.text(0.5, 0.91, f"Archivo: {filename}", ha='center', fontsize=12, color='grey')
        
        plt.text(0.05, 0.85, "RESUMEN COMPARATIVO POR TEMPORALIDADES", fontsize=14, fontweight='bold', color=C_HEADER)
        
        # Tabla resumen
        headers = ["Timeframe", "Media % (+)", "Media % (-)", "Dur. Swing UP", "Dur. Swing DN", "Rango H-L", "Freq (+)", "Ratio C/M"]
        rows = []
        for tf, data in results.items():
            g = data['general']
            rows.append([
                tf,
                f"{g['m_pos']:.3f}%",
                f"{g['m_neg']:.3f}%",
                f"{g.get('swing_dur_up',0):.1f} v",
                f"{g.get('swing_dur_dn',0):.1f} v",
                f"{g.get('range_mean', 0):.2f}",
                f"{(g['count_pos']/g['total']*100):.1f}%",
                f"{g.get('body_mean',0)/g.get('wick_mean',1):.2f}x" if g.get('wick_mean',0)>0 else "N/A"
            ])

            
        table = ax.table(cellText=rows, colLabels=headers, loc='center', bbox=[0.05, 0.45, 0.9, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Estilo tabla
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(C_HEADER)
                cell.set_text_props(color=C_WHITE, weight='bold')
            else:
                cell.set_facecolor(C_WHITE if row % 2 == 0 else '#F8F9FA')
                if col == 1: cell.set_text_props(color='#00897B', weight='bold')
                if col == 2: cell.set_text_props(color='#C62828', weight='bold')

        # Footer página 1
        plt.text(0.5, 0.1, "Modelox - Quantitative Analysis Report", ha='center', fontsize=8, color='grey')
        
        pdf.savefig(fig)
        plt.close()
        
        # PÁGINAS DE DETALLES
        for tf, data in results.items():
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            plt.text(0.05, 0.95, f"DETALLE TEMPORALIDAD: {tf}", fontsize=16, fontweight='bold', color=C_TITLE)
            
            # Sub-tabla 1: Stats Generales
            g = data['general']
            plt.text(0.05, 0.88, "Estadísticas Generales", fontsize=11, fontweight='bold')
            gs_rows = [
                ["Media Variación (%)", f"{g['m_pos']:.4f}%", f"{g['m_neg']:.4f}%"],
                ["Media Variación Abs.", f"{g['a_pos']:.2f}", f"{g['a_neg']:.2f}"],
                ["Rango Medio (H-L)", f"{g.get('range_mean',0):.2f}", f"{g.get('range_mean',0):.2f}"],
                ["Máxima Variación (%)", f"{g['max_pos']:.2f}%", f"{g['max_neg']:.2f}%"],
                ["Velas Analizadas", str(g['count_pos']), str(g['count_neg'])],
                ["Frecuencia", f"{(g['count_pos']/g['total']*100):.1f}%", f"{(g['count_neg']/g['total']*100):.1f}%"]
            ]
            t_gs = ax.table(cellText=gs_rows, colLabels=["MÉTRICA", "VERDES (+)", "ROJAS (-)"], 
                            loc='upper left', bbox=[0.05, 0.65, 0.5, 0.2])
            t_gs.auto_set_font_size(False)
            t_gs.set_fontsize(8)
            for (row, col), cell in t_gs.get_celld().items():
                if row == 0:
                    cell.set_facecolor(C_HEADER); cell.set_text_props(color=C_WHITE, weight='bold')
                else: 
                    if col == 0: cell.set_facecolor(C_LABEL); cell.set_text_props(weight='bold')
            
            # Sub-tabla 2: Anatomía
            plt.text(0.6, 0.88, "Impulsos & Anatomía", fontsize=11, fontweight='bold')
            an_rows = [
                ["Dur. Swing UP", f"{g.get('swing_dur_up',0):.1f} velas", f"{g.get('swing_pct_up',0):.2f}%"],
                ["Dur. Swing DOWN", f"{g.get('swing_dur_dn',0):.1f} velas", f"{g.get('swing_pct_dn',0):.2f}%"],
                ["Ratio C/M", f"{g['body_mean']/g['wick_mean']:.2f}x" if g.get('wick_mean',0)>0 else "N/A", ""]
            ]
            t_an = ax.table(cellText=an_rows, colLabels=["MÉTRICA", "VALOR", "RETORNO %"], 
                            loc='upper right', bbox=[0.6, 0.72, 0.35, 0.13])
            t_an.auto_set_font_size(False); t_an.set_fontsize(8)
            for (row, col), cell in t_an.get_celld().items():
                if row == 0: cell.set_facecolor(C_HEADER); cell.set_text_props(color=C_WHITE, weight='bold')
                elif col == 0: cell.set_facecolor(C_LABEL); cell.set_text_props(weight='bold')

            # Tabla 3: Desglose Anual
            if data['years']:
                plt.text(0.05, 0.58, "Desglose Histórico Anual", fontsize=12, fontweight='bold', color=C_HEADER)
                y_hd = ["Año", "Media % (+)", "Media % (-)", "Pts (+)", "Pts (-)", "Rango H-L", "Cuerpo", "Mechas", "Ratio", "Freq (+)", "Velas"]
                y_rows = []
                for year, s in data['years'].items():
                    y_rows.append([
                        year, f"{s['m_pos']:.2f}%", f"{s['m_neg']:.2f}%", f"{s['a_pos']:.1f}", f"{s['a_neg']:.1f}", 
                        f"{s.get('range_mean',0):.1f}", f"{s['body_mean']:.1f}", f"{s['wick_mean']:.1f}", 
                        f"{s['body_mean']/s['wick_mean']:.1f}x" if s.get('wick_mean',0)>0 else "N/A",
                        f"{(s['count_pos']/s['total']*100):.1f}%", s['total']
                    ])
                
                t_y = ax.table(cellText=y_rows, colLabels=y_hd, loc='bottom', bbox=[0.02, 0.05, 0.96, 0.5])
                t_y.auto_set_font_size(False); t_y.set_fontsize(7)
                for (row, col), cell in t_y.get_celld().items():
                    if row == 0:
                        cell.set_facecolor(C_HEADER); cell.set_text_props(color=C_WHITE, weight='bold')
                    else:
                        cell.set_facecolor(C_WHITE if row % 2 == 0 else '#F8F9FA')
                        if col == 1: cell.set_text_props(color='#00897B', weight='bold')
                        if col == 2: cell.set_text_props(color='#C62828', weight='bold')

            pdf.savefig(fig)
            plt.close()

def verificar_muestras_rich(df, tf_name):
    """Muestra una auditoría detallada de 3 velas aleatorias para verificar cálculos."""
    console.print(f"\n[bold magenta]🔍 AUDITORÍA DE CÁLCULOS (Timeframe: {tf_name})[/]")
    
    # Elegir 3 muestras aleatorias
    muestras = df.sample(min(3, len(df)))
    
    for i, (idx, row) in enumerate(muestras.iterrows()):
        o, h, l, c = row['OPEN'], row['HIGH'], row['LOW'], row['CLOSE']
        v_pct = row['VAR_PCT']
        v_abs = row['VAR_ABS']
        
        # Cálculos manuales para la auditoría
        man_var_pct = (c - o) / o * 100
        man_var_abs = c - o
        man_range = h - l
        man_body = abs(c - o)
        man_up_wick = h - max(o, c)
        man_low_wick = min(o, c) - l
        man_total_wick = man_up_wick + man_low_wick
        
        # Verificar integridad (Rango = Cuerpo + Mechas)
        integridad = "OK" if abs(man_range - (man_body + man_total_wick)) < 0.0001 else "ERROR"
        
        # Crear Panel de Auditoría
        audit_text = (
            f"[bold yellow]VELA {i+1} - {idx if isinstance(idx, pd.Timestamp) else 'Sample'}[/]\n"
            f"⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n"
            f"[bold cyan]DATOS OHLC:[/] O: {o:.2f} | H: {h:.2f} | L: {l:.2f} | C: {c:.2f}\n\n"
            f"[bold cyan]1. VARIACIÓN:[/]\n"
            f"   • %: ({c:.2f} - {o:.2f}) / {o:.2f} * 100 = [bold]{man_var_pct:.4f}%[/] (Script: {v_pct:.4f}%)\n"
            f"   • Puntos: {c:.2f} - {o:.2f} = [bold]{man_var_abs:.2f}[/] (Script: {v_abs:.2f})\n\n"
            f"[bold cyan]2. ANATOMÍA:[/]\n"
            f"   • Rango Total (H-L): {h:.2f} - {l:.2f} = [bold]{man_range:.2f}[/]\n"
            f"   • Cuerpo (|C-O|): |{c:.2f} - {o:.2f}| = [bold]{man_body:.2f}[/]\n"
            f"   • Mecha Sup (H-max(O,C)): {h:.2f} - {max(o,c):.2f} = [bold]{man_up_wick:.2f}[/]\n"
            f"   • Mecha Inf (min(O,C)-L): {min(o,c):.2f} - {l:.2f} = [bold]{man_low_wick:.2f}[/]\n"
            f"   • Suma Mechas: {man_up_wick:.2f} + {man_low_wick:.2f} = [bold]{man_total_wick:.2f}[/]\n\n"
            f"[bold cyan]3. VERIFICACIÓN DE INTEGRIDAD:[/]\n"
            f"   • Cuerpo ({man_body:.2f}) + Mechas ({man_total_wick:.2f}) == Rango ({man_range:.2f})? [bold green]{integridad}[/]\n"
        )
        
        console.print(Panel(audit_text, border_style="magenta", expand=False))

if __name__ == "__main__":
    try:
        analizar_variacion()
    except KeyboardInterrupt:
        print("\nOperación cancelada.")
    except Exception as e:
        print(f"\nOcurrió un error: {e}")
        import traceback
        traceback.print_exc()
