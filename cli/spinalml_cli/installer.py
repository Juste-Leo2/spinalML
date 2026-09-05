# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import os
import tarfile
import urllib.request
import sys
import subprocess
import json
from pathlib import Path

from .config import TOOLS_DIR, get_bin_path

MANIFEST_FILE = TOOLS_DIR / ".installed_manifest.json"

def _load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_manifest(manifest: dict):
    try:
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        pass

def clean_coursier_cache(console=None, debug: bool = False):
    """Removes cached Coursier / Ivy artifacts to avoid SHA-1 checksum corruption."""
    import shutil
    home = Path.home()
    dirs = [
        home / ".cache" / "coursier",
        home / ".coursier",
        home / ".ivy2" / "cache"
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        dirs.append(Path(local_app_data) / "Coursier" / "Cache")
    app_data = os.environ.get("APPDATA")
    if app_data:
        dirs.append(Path(app_data) / "Coursier")

    cleaned = False
    for d in dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            cleaned = True
    if cleaned:
        msg = "Cleared Coursier & Ivy cache to prevent checksum corruption."
        if debug:
            print(msg)
        elif console:
            console.print(f"[dim cyan]{msg}[/dim cyan]")

def setup_tools(config: dict, debug: bool = False, force: bool = False, clean_cache: bool = False):
    from .config import get_oss_cad_suite_url, get_mill_url, get_w64devkit_url, get_os_arch
    
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    
    if force or clean_cache:
        clean_coursier_cache(debug=debug)

    is_win = "windows" in get_os_arch()
    
    if debug:
        print(f"Setting up tools in {TOOLS_DIR}...")
        install_oss_cad_suite(get_oss_cad_suite_url(config), debug=True, force=force)
        if is_win:
            install_w64devkit(get_w64devkit_url(config), debug=True, force=force)
        install_mill(get_mill_url(config), debug=True, force=force)
        print("Setup completed successfully!")
    else:
        from rich.console import Console
        console = Console()
        console.print(f"[bold blue]Checking tools in {TOOLS_DIR}...[/bold blue]")
        try:
            install_oss_cad_suite(get_oss_cad_suite_url(config), debug=False, console=console, force=force)
            if is_win:
                install_w64devkit(get_w64devkit_url(config), debug=False, console=console, force=force)
            install_mill(get_mill_url(config), debug=False, console=console, force=force)
            console.print("[bold green]Tools are verified and up to date![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error during setup:[/bold red] {e}")
            sys.exit(1)

def download_file(url: str, dest_path: Path, debug: bool = False, console=None):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if debug:
        print(f"Downloading {url}...")
        def report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded * 100 / total_size
                sys.stdout.write(f"\rDownloaded: {downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB ({percent:.1f}%)")
                sys.stdout.flush()
        urllib.request.urlretrieve(url, str(dest_path), reporthook=report)
        print("\nDownload complete.")
    else:
        from rich.progress import Progress
        with Progress(console=console) as progress:
            task = progress.add_task(f"[cyan]Downloading {dest_path.name}...", total=100)
            def report(block_num, block_size, total_size):
                if total_size > 0:
                    progress.update(task, completed=(block_num * block_size * 100 / total_size))
            urllib.request.urlretrieve(url, str(dest_path), reporthook=report)

def extract_tgz(tgz_path: Path, dest_dir: Path, debug: bool = False, console=None):
    if debug:
        print(f"Extracting {tgz_path.name} to {dest_dir}...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("Extraction complete.")
    else:
        if console:
            console.print(f"[cyan]Extracting {tgz_path.name}...[/cyan]")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)

def install_oss_cad_suite(url: str, debug: bool, console=None, force: bool = False):
    oss_dir = TOOLS_DIR / "oss-cad-suite"
    manifest = _load_manifest()

    # If directory exists and already matches URL, skip
    if oss_dir.exists() and not force:
        if manifest.get("oss-cad-suite") == url or "oss-cad-suite" not in manifest:
            manifest["oss-cad-suite"] = url
            _save_manifest(manifest)
            msg = "OSS CAD Suite is up to date (skipping download)."
            if debug:
                print(msg)
            elif console:
                console.print(f"[green]{msg}[/green]")
            return

    msg = "Updating OSS CAD Suite to new version..." if oss_dir.exists() else "Installing OSS CAD Suite..."
    if debug:
        print(msg)
    elif console:
        console.print(f"[yellow]{msg}[/yellow]")

    if oss_dir.exists():
        import shutil
        shutil.rmtree(oss_dir, ignore_errors=True)

    tgz_path = TOOLS_DIR / "oss-cad-suite.tgz"
    download_file(url, tgz_path, debug=debug, console=console)
    extract_tgz(tgz_path, TOOLS_DIR, debug=debug, console=console)
    try:
        tgz_path.unlink()
    except Exception:
        pass
    manifest["oss-cad-suite"] = url
    _save_manifest(manifest)

def install_mill(url: str, debug: bool, console=None, force: bool = False):
    is_win = os.name == "nt"
    mill_bin = TOOLS_DIR / ("mill.bat" if is_win else "mill")
    manifest = _load_manifest()

    if mill_bin.exists() and not force:
        if manifest.get("mill") == url:
            msg = "Mill is up to date (skipping download)."
            if debug:
                print(msg)
            elif console:
                console.print(f"[green]{msg}[/green]")
            return

    msg = "Updating Mill to new version..." if mill_bin.exists() else "Installing Mill..."
    if debug:
        print(msg)
    elif console:
        console.print(f"[yellow]{msg}[/yellow]")

    download_file(url, mill_bin, debug=debug, console=console)
    
    if os.name != "nt":
        mill_bin.chmod(0o755)
        
    if debug:
        print("Initializing Mill...")
    else:
        if console:
            console.print("[cyan]Initializing Mill...[/cyan]")
        
    try:
        subprocess.run([str(mill_bin), "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        if debug:
            print(f"Warning: Failed to initialize Mill: {e}")
        elif console:
            console.print(f"[yellow]Warning: Failed to initialize Mill automatically: {e}[/yellow]")

    manifest["mill"] = url
    _save_manifest(manifest)

def extract_zip(zip_path: Path, dest_dir: Path, debug: bool = False, console=None):
    import zipfile
    if debug:
        print(f"Extracting {zip_path.name} to {dest_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete.")
    else:
        if console:
            console.print(f"[cyan]Extracting {zip_path.name}...[/cyan]")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

def extract_sfx(exe_path: Path, dest_dir: Path, debug: bool = False, console=None):
    if debug:
        print(f"Extracting {exe_path.name} to {dest_dir}...")
    else:
        if console:
            console.print(f"[cyan]Extracting {exe_path.name}...[/cyan]")
    
    try:
        subprocess.run([str(exe_path), "-y", f"-o{dest_dir}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if debug:
            print("Extraction complete.")
    except Exception as e:
        if console:
            console.print(f"[red]Failed to extract {exe_path.name}: {e}[/red]")
        else:
            print(f"Failed to extract {exe_path.name}: {e}")
        raise

def install_w64devkit(url: str, debug: bool, console=None, force: bool = False):
    w64_dir = TOOLS_DIR / "w64devkit"
    manifest = _load_manifest()

    if w64_dir.exists() and not force:
        if manifest.get("w64devkit") == url or "w64devkit" not in manifest:
            manifest["w64devkit"] = url
            _save_manifest(manifest)
            msg = "w64devkit is up to date (skipping download)."
            if debug:
                print(msg)
            elif console:
                console.print(f"[green]{msg}[/green]")
            return

    exe_path = TOOLS_DIR / "w64devkit.exe"
    download_file(url, exe_path, debug=debug, console=console)
    extract_sfx(exe_path, TOOLS_DIR, debug=debug, console=console)
    try:
        exe_path.unlink()
    except Exception:
        pass
    manifest["w64devkit"] = url
    _save_manifest(manifest)
