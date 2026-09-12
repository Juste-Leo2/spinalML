# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT
"""
Build script to compile SpinalML CLI into a standalone, single-file executable using PyInstaller.
Can be invoked locally or within GitHub Actions CI.
"""

import os
import sys
import platform
import argparse
from pathlib import Path
import PyInstaller.__main__

def get_target_binary_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        return "spinalml-windows-x64.exe"
    elif system == "linux":
        if "arm" in machine or "aarch64" in machine:
            return "spinalml-linux-arm64"
        return "spinalml-linux-x64"
    elif system == "darwin":
        if "arm" in machine or "aarch64" in machine:
            return "spinalml-darwin-arm64"
        return "spinalml-darwin-x64"
    
    return "spinalml.exe" if system == "windows" else "spinalml"

def build(custom_name: str = None):
    repo_root = Path(__file__).parent.parent.resolve()
    cli_dir = repo_root / "cli"
    entrypoint = cli_dir / "main.py"
    dist_dir = repo_root / "dist"
    build_dir = repo_root / "build"

    binary_name = custom_name or get_target_binary_name()
    print(f"=== Building SpinalML CLI Standalone Binary: {binary_name} ===")
    print(f"Repository Root : {repo_root}")
    print(f"Entrypoint      : {entrypoint}")
    print(f"Target Output   : {dist_dir / binary_name}")

    sep = os.pathsep

    # Static data bundles
    data_args = [
        f"{cli_dir / 'config.json'}{sep}.",
        f"{repo_root / 'boards'}{sep}boards",
        f"{repo_root / 'build.mill'}{sep}.",
        f"{repo_root / 'spinalML'}{sep}spinalML",
    ]

    pyinstaller_args = [
        str(entrypoint),
        "--onefile",
        "--clean",
        "--noconfirm",
        f"--name={binary_name}",
        f"--distpath={dist_dir}",
        f"--workpath={build_dir}",
        "--hidden-import=typer",
        "--hidden-import=rich",
        "--hidden-import=requests",
        "--collect-submodules=spinalml_cli",
        "--exclude-module=numpy",
        "--exclude-module=PIL",
        "--exclude-module=pytest",
        "--exclude-module=scipy",
        "--exclude-module=matplotlib",
    ]


    for data in data_args:
        pyinstaller_args.extend(["--add-data", data])

    print(f"Invoking PyInstaller with args:\n" + " \n".join(pyinstaller_args))
    PyInstaller.__main__.run(pyinstaller_args)

    output_file = dist_dir / binary_name
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n[SUCCESS] Binary created successfully: {output_file} ({size_mb:.2f} MB)")
    else:
        print(f"\n[ERROR] Output binary was not found at {output_file}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SpinalML CLI standalone executable")
    parser.add_argument("--name", type=str, default=None, help="Custom output binary name")
    args = parser.parse_args()
    build(args.name)
