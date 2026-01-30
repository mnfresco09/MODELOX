"""
Setup para compilar las extensiones C de MODELOX.

USO:
    cd cp
    python setup.py build_ext --inplace
    
    O para compilación optimizada:
    python setup.py build_ext --inplace --define CYTHON_TRACE=0

REQUISITOS:
    - Cython >= 0.29
    - NumPy
    - Compilador C (gcc, clang, MSVC)
"""

import os
import sys
import numpy as np
from setuptools import setup, Extension

# Intentar importar Cython
try:
    from Cython.Build import cythonize
    from Cython.Compiler import Options
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("⚠️  Cython no está instalado. Instalando...")
    os.system(f"{sys.executable} -m pip install cython")
    from Cython.Build import cythonize
    from Cython.Compiler import Options


# Configuración de optimización del compilador
Options.annotate = False  # No generar HTML de anotaciones (más rápido)

# Flags de compilación según plataforma
if sys.platform == "darwin":
    # macOS con Apple Silicon o Intel
    import platform
    if platform.machine() == "arm64":
        # Apple Silicon (M1/M2/M3)
        extra_compile_args = [
            "-O3",           # Optimización máxima
            "-ffast-math",   # Matemáticas rápidas (no-strict IEEE)
            "-funroll-loops", # Desenrollar loops
            "-fno-strict-aliasing",
            "-Wno-unreachable-code",
        ]
    else:
        # Intel Mac
        extra_compile_args = [
            "-O3",
            "-ffast-math",
            "-funroll-loops",
            "-fno-strict-aliasing",
            "-Wno-unreachable-code",
        ]
    extra_link_args = ["-O3"]
elif sys.platform == "win32":
    # Windows con MSVC
    extra_compile_args = [
        "/O2",           # Optimización máxima
        "/fp:fast",      # Matemáticas rápidas
        "/arch:AVX2",    # Instrucciones SIMD
    ]
    extra_link_args = []
else:
    # Linux
    extra_compile_args = [
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-march=native",
        "-fopenmp",      # Soporte OpenMP para parallelización
        "-fno-strict-aliasing",
    ]
    extra_link_args = ["-O3", "-fopenmp"]


# Definir extensión
extensions = [
    Extension(
        name="nuclear_engine",
        sources=["nuclear_engine.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_TRACE", "0"),
        ],
        language="c",
    ),
]

# Configuración de Cython
compiler_directives = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "nonecheck": False,
    "overflowcheck": False,
    "embedsignature": True,
    "profile": False,
    "linetrace": False,
}

if __name__ == "__main__":
    setup(
        name="modelox_nuclear",
        version="2.0.0",
        description="MODELOX Nuclear Engine - Extensiones C de Alto Rendimiento",
        author="MODELOX",
        ext_modules=cythonize(
            extensions,
            compiler_directives=compiler_directives,
            annotate=False,  # No generar HTML
            nthreads=os.cpu_count() or 4,  # Compilación paralela
        ),
        zip_safe=False,
    )
    
    print("\n" + "="*60)
    print("✅ COMPILACIÓN EXITOSA")
    print("="*60)
    print("\nLas extensiones C están listas para usar.")
    print("Importa con: from cp import simulate_trades_c, C_AVAILABLE")
    print("="*60 + "\n")
