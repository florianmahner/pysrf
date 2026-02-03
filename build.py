#!/usr/bin/env python3
"""
Build script for compiling Cython extensions.

This script compiles the Cython extensions for pysrf during development
or installation. It should be run from the project root directory.

Usage:
    python build.py              # Direct execution
    poetry run pysrf-compile     # Via Poetry script
    make compile                 # Via Makefile
"""

from __future__ import annotations

import sys
import tempfile
import shutil
import subprocess
from pathlib import Path


def compile_single_extension(pyx_name: str, numpy_include: str, python_include: str, ext_suffix: str) -> bool:
    """Compile a single Cython extension.

    Args:
        pyx_name: Name of the pyx file (without extension)
        numpy_include: Path to numpy include directory
        python_include: Path to Python include directory
        ext_suffix: Extension suffix for shared objects

    Returns:
        True if compilation successful, False otherwise.
    """
    pyx_file: Path = Path(__file__).parent / "pysrf" / f"{pyx_name}.pyx"

    if not pyx_file.exists():
        print(f"Cython source not found: {pyx_file}")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path: Path = Path(temp_dir)
        temp_pyx: Path = temp_path / f"{pyx_name}.pyx"
        shutil.copy2(pyx_file, temp_pyx)

        print(f"Compiling {pyx_name}...")
        cmd_cython: list[str] = [
            sys.executable,
            "-m",
            "cython",
            "--cplus",
            "--3str",
            "-o",
            str(temp_path / f"{pyx_name}.cpp"),
            str(temp_pyx),
        ]

        result: subprocess.CompletedProcess[str] = subprocess.run(
            cmd_cython, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Cython compilation failed for {pyx_name}: {result.stderr}")
            return False

        output_file: Path = pyx_file.parent / f"{pyx_name}{ext_suffix}"

        cmd_compile: list[str] = [
            "g++",
            "-shared",
            "-fPIC",
            "-O3",
            "-w",
            f"-I{python_include}",
            f"-I{numpy_include}",
            "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
            str(temp_path / f"{pyx_name}.cpp"),
            "-o",
            str(output_file),
        ]

        result = subprocess.run(cmd_compile, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"C++ compilation failed for {pyx_name}: {result.stderr}")
            return False

        print(f"Successfully compiled: {output_file}")
        return True


def compile_extensions() -> bool:
    """Compile all Cython extensions.

    Returns:
        True if all compilations successful, False otherwise.
    """
    try:
        import numpy
    except ImportError:
        print("numpy not found, cannot compile Cython extensions")
        return False

    import sysconfig

    python_include: str | None = sysconfig.get_path("include")
    numpy_include: str = numpy.get_include()
    ext_suffix: str = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

    # List of Cython extensions to compile
    extensions = ["_bsum", "_vbsum"]

    all_success = True
    for ext in extensions:
        if not compile_single_extension(ext, numpy_include, python_include, ext_suffix):
            all_success = False

    return all_success


def main() -> None:
    """Main entry point for compilation script."""
    success: bool = compile_extensions()
    if success:
        print("✓ Cython compilation successful")
        sys.exit(0)
    else:
        print("✗ Cython compilation failed, will use Python fallback")
        sys.exit(1)


if __name__ == "__main__":
    main()
