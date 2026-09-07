# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""
Bridge L2 protocol scenario test on UartBridgeTestComp.

Covers, in one long session (reference protocol, see docs/uart_bridge.md):
  - 'V' version reply
  - 'S' status byte (idle + after idle)
  - 'C' CSR writes (0x08 imgBase / 0x0C weightBase / 0x1C run) verified on
    the stub regs
  - 'W' memory writes (image region 0x10000, weights region 0x20000 with the
    +0x800 mapping) verified by AXI reads back through the BRAM slave port
  - 'W' with len=0 (no-op, bridge stays idle)
  - 'R' 10 logits from the stub output stream (0..9)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run

from utils.tb_utils import run_mill

CLK_FREQ = 27000000
BAUD_RATE = 115200
CLK_PER_BIT = CLK_FREQ // BAUD_RATE  # 234
HALF = CLK_PER_BIT // 2


def frame_bits(byte: int):
    return [0] + [(byte >> i) & 1 for i in range(8)] + [1]


def decode_frame(samples, start):
    """Decode one frame starting at `start` (index of the falling edge)."""
    bits = [samples[start + k * CLK_PER_BIT + HALF] for k in range(10)]
    if bits[0] != 0 or bits[9] != 1:
        return None
    val = 0
    for k, b in enumerate(bits[1:9]):
        val |= b << k
    return val


async def script(dut, payload: bytes, capture_symbols: int, n_frames: int):
    """Send `payload`, sample the TX line from the last byte through a
    `capture_symbols`-symbol tail, decode exactly `n_frames` response bytes
    (the bridge replies while the command is still being transmitted)."""
    samples = []
    last = len(payload) - 1
    for idx, byte in enumerate(payload):
        for bit in frame_bits(byte):
            dut.io_rxIn.value = bit
            for _ in range(CLK_PER_BIT):
                await RisingEdge(dut.clk)
                if idx == last:
                    samples.append(int(dut.io_txOut.value))
        dut.io_rxIn.value = 1
        await RisingEdge(dut.clk)
        if idx == last:
            samples.append(int(dut.io_txOut.value))
    for _ in range(capture_symbols * CLK_PER_BIT):
        await RisingEdge(dut.clk)
        samples.append(int(dut.io_txOut.value))

    frames = []
    i = 1
    while len(frames) < n_frames and i < len(samples):
        if samples[i - 1] == 1 and samples[i] == 0:
            if i + 10 * CLK_PER_BIT <= len(samples):
                val = decode_frame(samples, i)
                if val is not None:
                    frames.append(val)
                    i += 10 * CLK_PER_BIT
                    continue
        i += 1
    assert len(frames) == n_frames, (
        f"Expected {n_frames} response frames, decoded {len(frames)}")
    return frames


async def bram_read(dut, addr, n_beats=1):
    """Issue an AXI read burst on io_bram and return the list of 64-bit words."""
    dut.io_bram_ar_valid.value = 1
    dut.io_bram_ar_payload_addr.value = addr
    dut.io_bram_ar_payload_id.value = 0
    dut.io_bram_ar_payload_len.value = (n_beats - 1) & 0xFF
    dut.io_bram_r_ready.value = 1
    words = []
    while True:
        await RisingEdge(dut.clk)
        if int(dut.io_bram_ar_ready.value) == 1 and int(dut.io_bram_ar_valid.value) == 1:
            dut.io_bram_ar_valid.value = 0
        if int(dut.io_bram_r_valid.value) == 1:
            words.append(int(dut.io_bram_r_payload_data.value))
            if int(dut.io_bram_r_payload_last.value) == 1:
                break
    for _ in range(10):
        await RisingEdge(dut.clk)
        if int(dut.io_bram_r_valid.value) == 0:
            break
    dut.io_bram_r_ready.value = 0
    return words


WORD0 = bytes([i % 256 for i in range(64)])  # 8 words, LSB-first per word


@cocotb.test()
async def cocotb_uart_bridge(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    dut.io_rxIn.value = 1
    dut.io_bram_ar_valid.value = 0
    dut.io_bram_r_ready.value = 0
    await RisingEdge(dut.clk)

    # --- 'V' version ---
    (ver,) = await script(dut, b"V", capture_symbols=12, n_frames=1)
    assert ver == 0x01, f"V reply {ver} != 1"

    # --- 'S' idle status: bit0 = 1 (protocol probe) ---
    (status,) = await script(dut, b"S", capture_symbols=12, n_frames=1)
    assert status == 0x01, f"idle status {hex(status)} != 0x01"

    # --- 'C' CSR writes (silent commands) ---
    await script(dut,
                 b"C" + (0x00000008).to_bytes(4, "little") + (0x00010000).to_bytes(4, "little"),
                 capture_symbols=2, n_frames=0)
    for _ in range(50):
        await RisingEdge(dut.clk)
    assert int(dut.io_imgBaseR.value) == 0x00010000, (
        f"CSR imgBase {hex(int(dut.io_imgBaseR.value))} != 0x10000")

    await script(dut,
                 b"C" + (0x0000000C).to_bytes(4, "little") + (0x00020000).to_bytes(4, "little"),
                 capture_symbols=2, n_frames=0)
    for _ in range(50):
        await RisingEdge(dut.clk)
    assert int(dut.io_weightBaseR.value) == 0x00020000, (
        f"CSR weightBase {hex(int(dut.io_weightBaseR.value))} != 0x20000")

    await script(dut,
                 b"C" + (0x0000001C).to_bytes(4, "little") + (0x00000001).to_bytes(4, "little"),
                 capture_symbols=2, n_frames=0)
    for _ in range(50):
        await RisingEdge(dut.clk)
    assert int(dut.io_runRegR.value) == 0x01, f"CSR run reg {hex(int(dut.io_runRegR.value))}"

    # --- 'W' memory write: image region 0x10000 (8 words) ---
    await script(dut,
                 b"W" + (0x00010000).to_bytes(4, "little") + (64).to_bytes(4, "little") + WORD0,
                 capture_symbols=2, n_frames=0)
    for _ in range(30):
        await RisingEdge(dut.clk)

    words = await bram_read(dut, 0x10000, n_beats=2)
    w0 = int.from_bytes(WORD0[0:8], "little")
    w1 = int.from_bytes(WORD0[8:16], "little")
    assert words[0] == w0, f"BRAM[0x10000] {hex(words[0])} != {hex(w0)}"
    assert words[1] == w1, f"BRAM[0x10008] {hex(words[1])} != {hex(w1)}"

    # --- 'W' weights region 0x20000 (mapped to +0x800) ---
    w_word = bytes(range(0x40, 0x48))
    await script(dut,
                 b"W" + (0x00020000).to_bytes(4, "little") + (8).to_bytes(4, "little") + w_word,
                 capture_symbols=2, n_frames=0)
    for _ in range(30):
        await RisingEdge(dut.clk)
    words = await bram_read(dut, 0x20000, n_beats=1)
    assert words[0] == int.from_bytes(w_word, "little"), (
        f"BRAM[0x20000] {hex(words[0])} != {hex(int.from_bytes(w_word, 'little'))}")

    # 'W' with len=0 must be a no-op (bridge returns to IDLE)
    await script(dut,
                 b"W" + (0x00020008).to_bytes(4, "little") + (0).to_bytes(4, "little"),
                 capture_symbols=2, n_frames=0)
    for _ in range(30):
        await RisingEdge(dut.clk)
    (ver,) = await script(dut, b"V", capture_symbols=12, n_frames=1)
    assert ver == 0x01, "bridge stuck after W len=0"

    # --- 'R' 10 logits from the stub stream (trigger it first via CSR 0x00) ---
    await script(dut,
                 b"C" + (0x00000000).to_bytes(4, "little") + (0x00000001).to_bytes(4, "little"),
                 capture_symbols=2, n_frames=0)
    for _ in range(20):
        await RisingEdge(dut.clk)
    assert int(dut.io_outValidR.value) == 1, "stub output stream never went valid"

    logits = await script(dut, b"R", capture_symbols=140, n_frames=10)
    assert logits == list(range(10)), f"logits {logits} != 0..9"
    for _ in range(5):
        await RisingEdge(dut.clk)
    assert int(dut.io_outValidR.value) == 0, "stub stream still active (all 10 consumed)"

    # 'S' after the session: status back to idle
    (status,) = await script(dut, b"S", capture_symbols=12, n_frames=1)
    assert status == 0x01, f"status after R {hex(status)} != 0x01"


def test_uart_bridge_scenario():
    v_file = run_mill("spinalML.io.UartBridgeTest", "uart_bridge_toplevel", "UartBridgeTestComp")
    run(
        simulator="verilator",
        verilog_sources=[v_file],
        toplevel="UartBridgeTestComp",
        module="test_uart_bridge",
        testcase="cocotb_uart_bridge",
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
    )
