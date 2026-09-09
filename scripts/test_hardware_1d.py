#!/usr/bin/env python3
# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""
SpinalML Hardware Bit-Exact Inference Verification Script
Tests the Universal1DDemo accelerator deployed on Tang Primer 20K over UART.
"""

import sys
import time
import json
import struct
import argparse
from pathlib import Path

try:
    import serial
except ImportError:
    print("[ERROR] pyserial is required. Run 'uv pip install pyserial' first.")
    sys.exit(1)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

CMD_CSR_WRITE   = 0x43  # 'C'
CMD_MEM_WRITE   = 0x57  # 'W'
CMD_READ_LOGITS = 0x52  # 'R'
CMD_STATUS      = 0x53  # 'S'
CMD_VERSION     = 0x56  # 'V'

def run_hardware_test(port: str = "COM8", baud: int = 115200, vectors_path: Path = None) -> bool:
    console.print(Panel(
        f"[bold cyan]SpinalML Hardware In-Circuit Bit-Exact Tester[/bold cyan]\n"
        f"Target Port : [yellow]{port}[/yellow] @ [yellow]{baud}[/yellow] baud\n"
        f"Test Model  : [bold green]Universal1DDemo[/bold green] (Tang Primer 20K)",
        border_style="cyan"
    ))

    # 1. Load test vectors
    if vectors_path is None:
        vectors_path = Path("hw_build/tang-primer-20k/test_vectors.json")

    if not vectors_path.exists():
        console.print(f"[bold red]Error: Test vectors file not found at {vectors_path}[/bold red]")
        console.print("[yellow]Please generate it first via Mill Export1DVectors.[/yellow]")
        return False

    with open(vectors_path, "r", encoding="utf-8") as f:
        vectors = json.load(f)

    img_base = vectors.get("img_base", 0x10000)
    weight_base = vectors.get("weight_base", 0x20000)
    out_count = vectors.get("out_count", 2)
    img_data = bytes.fromhex(vectors["image_bytes_hex"])
    weight_data = bytes.fromhex(vectors["weight_bytes_hex"])
    expected_logits = vectors["expected_logits"]
    expected_hex = vectors["expected_bytes_hex"]

    console.print(f"Loaded test vectors: [cyan]{len(img_data)} image bytes[/cyan], [cyan]{len(weight_data)} weight bytes[/cyan], expected: [magenta]{expected_logits}[/magenta] (hex: [bold]{expected_hex}[/bold])")

    # 2. Connect to Serial Port
    console.print(f"\n[bold]1. Connecting to {port}...[/bold]", end=" ")
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=2.0, write_timeout=2.0)
    except Exception as e:
        console.print(f"[bold red]FAILED[/bold red]\n[red]{e}[/red]")
        return False

    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    console.print("[bold green]CONNECTED[/bold green]")

    try:
        # 3. Handshake: Ping Version ('V') with up to 3 attempts
        console.print("[bold]2. Pinging SoC Protocol Version ('V')...[/bold]", end=" ")
        proto_ver = None
        for attempt in range(1, 4):
            ser.reset_input_buffer()
            ser.write(bytes([CMD_VERSION]))
            ser.flush()
            v_resp = ser.read(1)
            if len(v_resp) == 1:
                proto_ver = v_resp[0]
                break
            time.sleep(0.2)

        if proto_ver is None:
            console.print(f"[bold red]TIMEOUT[/bold red] (No response from FPGA on {port} after 3 attempts)")
            console.print("[yellow]Tip: Make sure the bitstream is flashed and the board is powered on.[/yellow]")
            return False
        if proto_ver != 0x01:
            console.print(f"[bold red]UNEXPECTED[/bold red] (Expected 0x01, got 0x{proto_ver:02X})")
            return False
        console.print(f"[bold green]OK[/bold green] (Protocol v{proto_ver})")

        # 4. Status Check ('S')
        console.print("[bold]3. Reading SoC Status ('S')...[/bold]", end=" ")
        ser.write(bytes([CMD_STATUS]))
        ser.flush()
        s_resp = ser.read(1)
        if len(s_resp) != 1:
            console.print("[bold red]TIMEOUT[/bold red]")
            return False
        status_byte = s_resp[0]
        busy = (status_byte >> 1) & 1
        console.print(f"[bold green]OK[/bold green] (Status: 0x{status_byte:02X}, Busy={busy})")

        # 5. Write Image Data ('W')
        console.print(f"[bold]4. Uploading Input Image ({len(img_data)} bytes) to 0x{img_base:05X}...[/bold]", end=" ")
        w_img_cmd = bytes([CMD_MEM_WRITE]) + struct.pack("<II", img_base, len(img_data)) + img_data
        ser.write(w_img_cmd)
        ser.flush()
        time.sleep(0.01)
        console.print("[bold green]DONE[/bold green]")

        # 6. Write Weights Data ('W')
        console.print(f"[bold]5. Uploading Model Weights ({len(weight_data)} bytes) to 0x{weight_base:05X}...[/bold]", end=" ")
        w_weight_cmd = bytes([CMD_MEM_WRITE]) + struct.pack("<II", weight_base, len(weight_data)) + weight_data
        ser.write(w_weight_cmd)
        ser.flush()
        time.sleep(0.01)
        console.print("[bold green]DONE[/bold green]")

        # 7. Configure CSR Base Addresses ('C')
        console.print(f"[bold]6. Programming CSRs (Image Base 0x{img_base:05X}, Weight Base 0x{weight_base:05X})...[/bold]", end=" ")
        # CSR 0x08: Image Base Address
        ser.write(bytes([CMD_CSR_WRITE]) + struct.pack("<II", 0x08, img_base))
        ser.flush()
        time.sleep(0.005)
        # CSR 0x0C: Weights Base Address
        ser.write(bytes([CMD_CSR_WRITE]) + struct.pack("<II", 0x0C, weight_base))
        ser.flush()
        time.sleep(0.005)
        console.print("[bold green]CONFIGURED[/bold green]")

        # 8. Start Inference via CSR 0x00 ('C')
        console.print("[bold]7. Triggering Hardware Inference (CSR 0x00 = 1)...[/bold]", end=" ")
        csr_cmd = bytes([CMD_CSR_WRITE]) + struct.pack("<II", 0x00, 0x01)
        ser.write(csr_cmd)
        ser.flush()
        time.sleep(0.02)  # Hardware inference takes ~5 microseconds at 27 MHz
        console.print("[bold green]TRIGGERED[/bold green]")

        # 8. Read Logits ('R')
        console.print(f"[bold]7. Reading {out_count} Logits from Output Stream ('R')...[/bold]", end=" ")
        ser.write(bytes([CMD_READ_LOGITS]))
        ser.flush()
        rx_logits = ser.read(out_count)
        if len(rx_logits) != out_count:
            console.print(f"[bold red]INCOMPLETE[/bold red] (Expected {out_count} bytes, received {len(rx_logits)})")
            return False
        console.print(f"[bold green]RECEIVED[/bold green] ({len(rx_logits)} bytes)")

        # 9. Format & Compare Results
        rx_hex = rx_logits.hex()
        # Convert bytes to signed integers (I8 format)
        rx_ints = [b - 256 if b > 127 else b for b in rx_logits]

        table = Table(title="Hardware vs Golden Oracle Inference Comparison", border_style="cyan")
        table.add_column("Logit Index", justify="center", style="bold")
        table.add_column("Hardware Output (Hex)", justify="center")
        table.add_column("Hardware Output (Signed I8)", justify="center", style="cyan")
        table.add_column("Golden Oracle (Signed I8)", justify="center", style="magenta")
        table.add_column("Bit-Exact Match", justify="center")

        all_match = True
        for i in range(out_count):
            hw_val = rx_ints[i]
            hw_hex = f"0x{rx_logits[i]:02X}"
            gold_val = expected_logits[i]
            match = (hw_val == gold_val)
            if not match:
                all_match = False
            status_text = "[bold green]EXACT MATCH[/bold green]" if match else "[bold red]MISMATCH[/bold red]"
            table.add_row(f"Logit [{i}]", hw_hex, str(hw_val), str(gold_val), status_text)

        console.print("\n", table)

        if all_match:
            console.print(Panel(
                f"[bold green]✓ 100% BIT-EXACT VERIFICATION SUCCESSFUL![/bold green]\n"
                f"The Tang Primer 20K FPGA hardware output matches the software oracle perfectly bit-for-bit.",
                border_style="green"
            ))
            return True
        else:
            console.print(Panel(
                f"[bold red]✗ VERIFICATION FAILED[/bold red]\n"
                f"Discrepancy detected between hardware output ({rx_ints}) and golden oracle ({expected_logits}).",
                border_style="red"
            ))
            return False

    finally:
        ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpinalML In-Circuit Hardware Tester")
    parser.add_argument("--port", default="COM8", help="Serial port of Tang Primer 20K (default: COM8)")
    parser.add_argument("--baud", type=int, default=115200, help="UART Baud rate (default: 115200)")
    parser.add_argument("--vectors", type=Path, default=Path("hw_build/tang-primer-20k/test_vectors.json"), help="Test vectors JSON")
    args = parser.parse_args()

    success = run_hardware_test(port=args.port, baud=args.baud, vectors_path=args.vectors)
    sys.exit(0 if success else 1)
