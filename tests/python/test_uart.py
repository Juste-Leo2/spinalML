# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""
UART IP golden tests (UartRx + UartTx): timing parity with the reference
uart_rx.v / uart_tx.v (27 MHz @ 115200 baud, CLK_PER_BIT = 234).

- RX golden: the drive timeline is known cycle-exactly, so the decode is
  checked at the sampling offsets derived from the RTL: start bit confirmed
  at CLK_PER_BIT/2 after the edge, data bits at (k+1)*CLK_PER_BIT, stop
  validated at 9*CLK_PER_BIT (2-cycle resync included).
- TX golden: the transmitter re-emits the byte; the 10 symbols are sampled
  at mid-symbol and compared against the reference frame {0, [b0..b7], 1}.
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
from cocotb_test.simulator import run

from utils.tb_utils import run_mill, seed_random

CLK_FREQ = 27000000
BAUD_RATE = 115200
CLK_PER_BIT = CLK_FREQ // BAUD_RATE  # 234
HALF = CLK_PER_BIT // 2


def frame_bits(byte: int):
    return [0] + [(byte >> i) & 1 for i in range(8)] + [1]


@cocotb.test()
async def cocotb_uart_loopback(dut):
    seed_random(20260907)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    cycle = 0

    async def tick():
        nonlocal cycle
        await RisingEdge(dut.clk)
        cycle += 1

    dut.reset.value = 1
    await tick()
    dut.reset.value = 0
    await tick()
    for _ in range(8):
        await tick()

    bytes_to_test = [0x00, 0xFF, 0x55, 0xAA, 0xA5, 0x43, 0x57, 0x52, 0x53, 0x50, 0x7F,
                     random.randint(1, 255)]

    dut.io_rxIn.value = 1

    for expected in bytes_to_test:
        t0 = cycle
        # --- RX drive: start, 8 bits LSB-first, stop + trailing idle ---
        first0_edge = None  # the edge that first latches the start bit (0)
        valid_seen = None
        data_at_valid = None
        tx_samples = []  # tx line sampled every cycle during the whole phase

        seq = frame_bits(expected) + [1, 1]
        driven = []
        for bit in seq:
            driven += [bit] * CLK_PER_BIT

        for i, bit in enumerate(driven):
            if bit == 0:
                first0_edge = t0 + i + 1  # edge index sampling this value
                break
        for i, bit in enumerate(driven):
            dut.io_rxIn.value = bit
            await tick()
            tx_samples.append(int(dut.io_txOut.value))
            if int(dut.io_validOut.value) == 1 and valid_seen is None:
                valid_seen = cycle
                data_at_valid = int(dut.io_dataOut.value)

        assert first0_edge is not None
        assert valid_seen is not None, f"RX never decoded {hex(expected)}"

        # Capture the full TX frame (may outlast the RX drive window)
        for _ in range(12 * CLK_PER_BIT):
            dut.io_rxIn.value = 1
            await tick()
            tx_samples.append(int(dut.io_txOut.value))

        # --- RX timing golden (windowed: 1 cycle per DATA symbol, total
        # ~9 bits after the start — the window stays << half a bit) ---
        expected_valid = first0_edge + 2 + HALF + 9 * CLK_PER_BIT
        assert abs(valid_seen - expected_valid) <= 12, (
            f"RX valid cycle {valid_seen} != expected ~{expected_valid} "
            f"(byte {hex(expected)})")
        assert data_at_valid == expected, (
            f"RX data {hex(data_at_valid)} != {hex(expected)}")

        # --- TX golden: decode from the OBSERVED falling edge ---
        rel_valid = valid_seen - t0
        s0 = None
        for i in range(len(tx_samples)):
            if i > rel_valid and tx_samples[i] == 0:
                s0 = i
                break
        assert s0 is not None, (
            f"TX never started its frame after valid (byte {hex(expected)})")
        # frame must show a low start for ~CLK_PER_BIT and total 10 symbols
        assert s0 + 10 * CLK_PER_BIT <= len(tx_samples), "TX frame truncated"
        decoded_bits = [tx_samples[s0 + k * CLK_PER_BIT + HALF] for k in range(10)]
        assert decoded_bits == frame_bits(expected), (
            f"TX frame {decoded_bits} != reference for {hex(expected)}")

        # leave the line idle before the next test
        dut.io_rxIn.value = 1
        while int(dut.io_txOut.value) != 1:
            await tick()


def test_uart_rx_tx_golden():
    v_file = run_mill("spinalML.io.UartRxTest", "uart_rx_toplevel", "UartRxTxLoopbackComp")
    run(
        simulator="verilator",
        verilog_sources=[v_file],
        toplevel="UartRxTxLoopbackComp",
        module="test_uart",
        testcase="cocotb_uart_loopback",
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
    )
