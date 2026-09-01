import os
import tarfile
import urllib.request
import sys
import subprocess
from pathlib import Path

from .config import TOOLS_DIR, get_bin_path

def setup_tools(config: dict, debug: bool = False):
    from .config import get_oss_cad_suite_url, get_mill_url, get_w64devkit_url, get_os_arch
    
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    
    is_win = "windows" in get_os_arch()
    
    if debug:
        print(f"Setting up tools in {TOOLS_DIR}...")
        install_oss_cad_suite(get_oss_cad_suite_url(config), debug=True)
        patch_verilator_headers()
        if is_win:
            install_w64devkit(get_w64devkit_url(config), debug=True)
        install_mill(get_mill_url(config), debug=True)
        print("Setup completed successfully!")
    else:
        from rich.console import Console
        console = Console()
        console.print(f"[bold blue]Setting up tools in {TOOLS_DIR}...[/bold blue]")
        try:
            install_oss_cad_suite(get_oss_cad_suite_url(config), debug=False, console=console)
            patch_verilator_headers()
            if is_win:
                install_w64devkit(get_w64devkit_url(config), debug=False, console=console)
            install_mill(get_mill_url(config), debug=False, console=console)
            console.print("[bold green]Setup completed successfully![/bold green] 🎉")
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

def install_oss_cad_suite(url: str, debug: bool, console=None):
    oss_dir = TOOLS_DIR / "oss-cad-suite"
    if oss_dir.exists():
        if debug:
            print("OSS CAD Suite is already installed.")
        else:
            console.print("[green]OSS CAD Suite is already installed.[/green]")
        return

    tgz_path = TOOLS_DIR / "oss-cad-suite.tgz"
    download_file(url, tgz_path, debug=debug, console=console)
    extract_tgz(tgz_path, TOOLS_DIR, debug=debug, console=console)
    tgz_path.unlink() # Clean up
    patch_verilator_headers()

def patch_verilator_headers():
    """Ensure Verilator headers provide WData typedef for SpinalHDL compatibility."""
    verilated_h = TOOLS_DIR / "oss-cad-suite" / "share" / "verilator" / "include" / "verilated.h"
    if verilated_h.exists():
        content = verilated_h.read_text(encoding="utf-8")
        if "using WData = uint32_t;" not in content and "typedef uint32_t WData;" not in content:
            if "using EData = uint32_t;" in content:
                content = content.replace("using EData = uint32_t;", "using EData = uint32_t;\nusing WData = uint32_t;")
                verilated_h.write_text(content, encoding="utf-8")

def install_mill(url: str, debug: bool, console=None):
    mill_bin = get_bin_path("mill")
    if mill_bin.exists():
        if debug:
            print("Mill is already installed.")
        else:
            console.print("[green]Mill is already installed.[/green]")
        return

    download_file(url, mill_bin, debug=debug, console=console)
    
    if os.name != "nt":
        mill_bin.chmod(0o755)
        
    # Initialize mill by running it once to download its JVM and dependencies
    if debug:
        print("Initializing Mill (downloading internal JVM and dependencies)...")
    else:
        console.print("[cyan]Initializing Mill (downloading internal dependencies)...[/cyan]")
        
    try:
        subprocess.run([str(mill_bin), "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        if debug:
            print(f"Warning: Failed to initialize Mill: {e}")
        else:
            console.print(f"[yellow]Warning: Failed to initialize Mill automatically: {e}[/yellow]")

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
        # 7-Zip SFX accepts -y (yes to all) and -o<dir> (output directory)
        subprocess.run([str(exe_path), "-y", f"-o{dest_dir}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if debug:
            print("Extraction complete.")
    except Exception as e:
        if console:
            console.print(f"[red]Failed to extract {exe_path.name}: {e}[/red]")
        else:
            print(f"Failed to extract {exe_path.name}: {e}")
        raise

def install_w64devkit(url: str, debug: bool, console=None):
    w64_dir = TOOLS_DIR / "w64devkit"
    if w64_dir.exists():
        if debug:
            print("w64devkit is already installed.")
        else:
            console.print("[green]w64devkit is already installed.[/green]")
        return

    exe_path = TOOLS_DIR / "w64devkit.exe"
    download_file(url, exe_path, debug=debug, console=console)
    extract_sfx(exe_path, TOOLS_DIR, debug=debug, console=console)
    exe_path.unlink() # Clean up
