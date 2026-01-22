"""
modelox/core/memory.py
Gestor de memoria de alto rendimiento para evitar OOM (Out Of Memory) Kills.
"""
import gc
import platform
import ctypes
import logging

logger = logging.getLogger(__name__)

def nuclear_cleanup():
    """
    Realiza una limpieza agresiva de memoria.
    1. Fuerza la recolección de basura de Python (Generaciones 0, 1 y 2).
    2. En Linux, fuerza a la librería C (malloc) a devolver la RAM al sistema operativo.
    """
    # 1. Recolección estándar de Python
    gc.collect()
    
    # 2. Truco para Linux (malloc_trim)
    # Python a veces libera memoria pero no se la devuelve al OS (fragmentación).
    # Esto fuerza la devolución.
    if platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            # 0 significa liberar todo el espacio posible del heap
            libc.malloc_trim(0)
        except Exception:
            pass

def clean_trial_variables(*vars_to_delete):
    """
    Borra variables específicas y ejecuta limpieza.
    Uso: clean_trial_variables(df, signals, trades)
    """
    for v in vars_to_delete:
        try:
            del v
        except UnboundLocalError:
            pass
        except Exception:
            pass
    
    nuclear_cleanup()