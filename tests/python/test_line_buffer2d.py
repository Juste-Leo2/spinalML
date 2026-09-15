# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

# Python/Cocotb co-simulation of LineBuffer2D (Refactoring 1.1), the BRAM
# delay-line that now backs im2col/seq2col/poolings.
#
# LineBuffer2D is a Flow (no ready): every accepted push beat yields exactly
# one pop beat one cycle later, with the payload of the beat accepted `depth`
# pushes earlier. The first (depth - 1) pops read not-yet-written memory, so
# they are skipped unless `withMemInit = true` (bitstream/FPGA zero-init).
# Four generated tops cover I8/I16, depth 4, the depth = 1 register path and
# the withMemInit variant.

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge
from cocotb_test.simulator import run

from utils.tb_utils import run_mill, seed_random

FILTER = "Generate Verilog for Python co-simulation"


def to_signed(value, bits):
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


async def reset_dut(dut):
    dut.io_push_valid.value = 0
    dut.io_push_payload.value = 0
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)


async def drive_and_collect(dut, values, depth, bits, gaps=False, seed=1234, skip_first=None):
    """Push `values` with optional random valid gaps, return captured pops.

    `skip_first` defaults to depth - 1: those priming pops read memory before
    it has been written for the first time (uninitialized on ASIC).
    """
    if skip_first is None:
        skip_first = depth - 1
    mask = (1 << bits) - 1
    rnd = random.Random(seed)

    n = len(values)
    idx = 0
    pop_count = 0
    captured = []

    while pop_count < n:
        valid = idx < n
        if valid and gaps and rnd.random() < 0.5:
            valid = False

        dut.io_push_valid.value = 1 if valid else 0
        if valid:
            dut.io_push_payload.value = values[idx] & mask

        await ReadOnly()
        pop_valid = int(dut.io_pop_valid.value)
        if pop_valid and pop_count >= skip_first:
            captured.append(to_signed(int(dut.io_pop_payload.value), bits))
        await RisingEdge(dut.clk)

        if valid:
            idx += 1
        if pop_valid:
            pop_count += 1

    dut.io_push_valid.value = 0
    return captured


@cocotb.test()
async def cocotb_line_buffer2d_continuous(dut):
    seed_random()
    depth = 4
    values = [((i * 3 + 7) & 0xFF) - 128 for i in range(30)]

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    captured = await drive_and_collect(dut, values, depth, bits=8)

    expected = values[: len(values) - (depth - 1)]
    assert captured == expected, f"Continuous stream corrupted: {captured} != {expected}"
    print(f"LineBuffer2D continuous depth={depth}: {len(captured)} beats delayed exactly")


@cocotb.test()
async def cocotb_line_buffer2d_valid_gaps(dut):
    seed_random()
    depth = 4
    values = [((i * 5 + 11) & 0xFF) - 128 for i in range(25)]

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    captured = await drive_and_collect(dut, values, depth, bits=8, gaps=True, seed=9876)

    expected = values[: len(values) - (depth - 1)]
    assert captured == expected, f"Stream corrupted under valid gaps: {captured} != {expected}"
    print(f"LineBuffer2D depth={depth}: {len(captured)} beats preserved under random valid gaps")


@cocotb.test()
async def cocotb_line_buffer2d_i16(dut):
    seed_random()
    depth = 4
    values = [((i * 2557) % 60000) - 30000 for i in range(24)]

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    captured = await drive_and_collect(dut, values, depth, bits=16)

    expected = values[: len(values) - (depth - 1)]
    assert captured == expected, f"I16 stream corrupted: {captured} != {expected}"
    print(f"LineBuffer2D I16 depth={depth}: {len(captured)} signed 16-bit beats delayed exactly")


@cocotb.test()
async def cocotb_line_buffer2d_depth1(dut):
    # depth = 1 takes the dedicated single-register path: pure 1-cycle delay.
    seed_random()
    values = [11, -22, 33, -44, 55]

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    captured = await drive_and_collect(dut, values, depth=1, bits=8, skip_first=0)

    assert captured == values, f"depth=1 delay broken: {captured} != {values}"
    print("LineBuffer2D depth=1 register path validated")


@cocotb.test()
async def cocotb_line_buffer2d_mem_init(dut):
    # withMemInit = true zero-initializes the memory: the depth - 1 priming
    # pops must read zeros instead of uninitialized cells.
    seed_random()
    depth = 4
    values = [((i * 7 + 3) & 0xFF) - 128 for i in range(15)]

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    captured = await drive_and_collect(dut, values, depth, bits=8, skip_first=0)

    expected = [0] * (depth - 1) + values[: len(values) - (depth - 1)]
    assert captured == expected, f"withMemInit priming mismatch: {captured} != {expected}"
    print(f"LineBuffer2D withMemInit depth={depth}: {depth - 1} zero priming pops confirmed")


def _run_sim(testcase, toplevel, sim_build, request=None):
    v_file = run_mill("spinalML.memory.LineBuffer2DTest", FILTER, toplevel)

    from utils.test_layers_utils import safe_run_sim as run

    run(
        simulator="verilator",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_line_buffer2d",
        testcase=testcase,
        sim_build=sim_build,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
    )


def test_line_buffer2d_continuous(request):
    _run_sim("cocotb_line_buffer2d_continuous", "LineBuffer2DTestComp", "sim_build/py_line_buffer2d_cont", request)


def test_line_buffer2d_valid_gaps(request):
    _run_sim("cocotb_line_buffer2d_valid_gaps", "LineBuffer2DTestComp", "sim_build/py_line_buffer2d_gaps", request)


def test_line_buffer2d_i16(request):
    _run_sim("cocotb_line_buffer2d_i16", "LineBuffer2DI16TestComp", "sim_build/py_line_buffer2d_i16", request)


def test_line_buffer2d_depth1(request):
    _run_sim("cocotb_line_buffer2d_depth1", "LineBuffer2DDepth1TestComp", "sim_build/py_line_buffer2d_depth1", request)


def test_line_buffer2d_mem_init(request):
    _run_sim("cocotb_line_buffer2d_mem_init", "LineBuffer2DInitTestComp", "sim_build/py_line_buffer2d_init", request)
