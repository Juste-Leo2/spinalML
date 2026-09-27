#!/usr/bin/env python3
# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""
Silicon validation for the DRAM-backed UniversalScaleDemo bitstream
(Tang Primer 20K, Gowin EDA flow). Terminal-only, no Gradio.

Protocol layer (opcodes, chunked 'W' upload, CSR map) reused from
examples/Mnist/inference.py; the model here is I8 Linear(1024 -> 64)
with Ks=256 spill (P=4), NOT the MNIST W4A8 net.

Deterministic vectors (mirror the Verilator harness
cli/spinalml_cli/cli.py + WeightMemoryLayout/ModelReplica bit-exactly):
  input   : x[i]   = ((i*7+3) % 15) - 7            (1024 int8, [-7,7])
  weights : W[o,k] = ((o*1024+k) % 7) + 1          (logical [o][k], 1..7)
  bias    : b[o]   = (o % 5) + 1                   (1..5)
  output  : y[o]   = wrap8(b[o] + sum_k x[k]*W[o][k])
DRAM layout at WEIGHT_BASE: weights slice-transposed
  phys[p*Ks*N + o*Ks + kl] = W[o, p*Ks+kl], then 64 bias bytes.

Usage (Windows):
  python examples\\UniversalScale\\silicon_validate.py --port COM8 [--ping-only]
"""

import argparse
import json
import struct
import sys
import time

try:
    import serial
    from serial.tools import list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# --- Protocol (docs/uart_bridge.md, cf. examples/Mnist/inference.py) ---
CMD_CSR_WRITE   = 0x43  # 'C' + u32 addr + u32 value (LE)
CMD_MEM_WRITE   = 0x57  # 'W' + u32 addr + u32 len (LE) + bytes
CMD_READ_LOGITS = 0x52  # 'R' -> outCount bytes
CMD_STATUS      = 0x53  # 'S' -> 1 status byte
CMD_VERSION     = 0x56  # 'V' -> 1 version byte (0x01)

DEFAULT_BAUD = 115200
IMG_BASE     = 0x10000
WEIGHT_BASE  = 0x20000
OUT_COUNT    = 64

K, N, KS = 1024, 64, 256


def build_vectors():
    """Returns (input_bytes, dram_weight_bytes, expected_out_bytes)."""
    x = [((i * 7 + 3) % 15) - 7 for i in range(K)]
    w_log = [[((o * K + k) % 7) + 1 for k in range(K)] for o in range(N)]
    b = [(o % 5) + 1 for o in range(N)]

    def wrap8(v):
        v %= 256
        return v - 256 if v >= 128 else v

    expected = bytes(wrap8(b[o] + sum(x[k] * w_log[o][k] for k in range(K)))
                     & 0xFF for o in range(N))

    # Slice-transposed physical order: p*Ks*N + o*Ks + kl
    passes = K // KS
    phys = bytearray()
    for p in range(passes):
        for o in range(N):
            for kl in range(KS):
                phys.append(w_log[o][p * KS + kl] & 0xFF)
    phys += bytes(v & 0xFF for v in b)  # bias region right after weights

    in_bytes = bytes(v & 0xFF for v in x)
    return in_bytes, bytes(phys), expected


def find_port(explicit=None):
    cands = [explicit] if explicit else []
    if not explicit:
        for info in list_ports.comports():
            dev = info.device
            if dev.startswith("COM") or "USB" in dev or "ACM" in dev.upper():
                cands.append(dev)
    for port in cands:
        try:
            ser = serial.Serial(port=port, baudrate=DEFAULT_BAUD,
                                timeout=0.5, write_timeout=0.5)
            ser.reset_input_buffer()
            ser.write(bytes([CMD_VERSION]))
            ser.flush()
            resp = ser.read(1)
            ser.close()
            if len(resp) == 1 and resp[0] == 0x01:
                return port
        except Exception:
            continue
    return None


def upload(ser, base_addr, data):
    ser.write(bytes([CMD_MEM_WRITE]) + struct.pack("<II", base_addr, len(data)))
    ser.flush()
    for i in range(0, len(data), 256):
        ser.write(data[i:i + 256])
        ser.flush()
        time.sleep(0.002)


def csr(ser, addr, value):
    ser.write(bytes([CMD_CSR_WRITE]) + struct.pack("<II", addr, value))
    ser.flush()
    time.sleep(0.002)


def main():
    ap = argparse.ArgumentParser(description="UniversalScaleDemo DRAM silicon check")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--ping-only", action="store_true")
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    if not HAS_SERIAL:
        print("FAIL: pyserial not installed (pip install pyserial)")
        return 1

    port = find_port(args.port)
    if port is None:
        print(f"FAIL: no UART port answering protocol v1 (tried {args.port or 'auto'})")
        print("  -> board flashed? USB-UART enumerated? right port?")
        return 1
    print(f"ping V: OK on {port} (SoC alive: clk + reset release + init_done)")

    ser = serial.Serial(port=port, baudrate=args.baud, timeout=5.0, write_timeout=2.0)
    try:
        ser.write(bytes([CMD_STATUS]))
        ser.flush()
        st = ser.read(1)
        print(f"status S: 0x{st.hex() if st else 'TIMEOUT'}")
        if args.ping_only:
            return 0

        in_bytes, weight_bytes, expected = build_vectors()
        print(f"upload: {len(in_bytes)}B input @0x{IMG_BASE:05X} ...")
        upload(ser, IMG_BASE, in_bytes)
        print(f"upload: {len(weight_bytes)}B weights+bias @0x{WEIGHT_BASE:05X} ...")
        upload(ser, WEIGHT_BASE, weight_bytes)
        csr(ser, 0x08, IMG_BASE)
        csr(ser, 0x0C, WEIGHT_BASE)
        csr(ser, 0x00, 0x01)
        print("triggered, waiting for spill inference (DRAM P=4)...")
        time.sleep(2.0)
        for i in range(3):
            ser.write(bytes([CMD_STATUS]))
            ser.flush()
            st = ser.read(1)
            if st:
                b = st[0]
                print(f"poll S[{i}]: 0x{b:02X} outValid={bool(b&8)} "
                      f"accDone={bool(b&4)} accBusy={bool(b&2)}")
            else:
                print(f"poll S[{i}]: TIMEOUT")
            time.sleep(0.5)
        ser.write(bytes([CMD_READ_LOGITS]))
        ser.flush()
        got = ser.read(OUT_COUNT)
    finally:
        ser.close()

    row = {"port": port, "expected": list(expected), "got": list(got)}
    ok = len(got) == OUT_COUNT and bytes(got) == expected
    if len(got) != OUT_COUNT:
        print(f"FAIL: got {len(got)}/{OUT_COUNT} bytes (timeout? inference not done?)")
    else:
        nbad = sum(1 for a, e in zip(got, expected) if a != e)
        bad = [(i, got[i], expected[i]) for i in range(OUT_COUNT) if got[i] != expected[i]]
        print(f"bytes: {OUT_COUNT - nbad}/{OUT_COUNT} match"
              + ("" if nbad == 0 else "  first bad: "
                 + ", ".join(f"[{i}] got {g:#04x} exp {e:#04x}" for i, g, e in bad[:5])))
        print("expected:", " ".join(f"{v:02X}" for v in expected))
        print("got     :", " ".join(f"{v:02X}" for v in got))
        print("BIT-EXACT PASS" if ok else "MISMATCH")
    row["pass"] = ok
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        print(f"results -> {args.save_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
