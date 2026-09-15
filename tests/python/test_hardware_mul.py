# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

# Python/Cocotb co-simulation of HardwareMul / DspConfig (Refactoring 1.2):
# the same 8x8->16 signed multiplier is generated for four target policies and
# must be bit-exact against the exact product in every case:
#   - Generic + useDsp        (legacy default)
#   - Generic + no-DSP        (--no-dsp / soft LUT logic)
#   - Target.ASIC (Sky130)    (pure behavioral, no vendor blackbox)
#   - latency = 0             (combinational fallback path)
# Enable gating (hold the previous product) is checked on the pipelined path.

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge, Timer
from cocotb_test.simulator import run

from utils.tb_utils import run_mill

FILTER = "Generate Verilog for Python co-simulation"

VECTORS = [
    (12, 10, 120),
    (7, -5, -35),
    (-6, -8, 48),
    (-128, 1, -128),
    (-128, -1, 128),
    (127, 127, 16129),
    (0, 100, 0),
    (100, 0, 0),
    (-1, -1, 1),
    (-127, 127, -16129),
]


def to_signed(value, bits):
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


async def reset_mult(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.io_a.value = 0
    dut.io_b.value = 0
    dut.io_enable.value = 0
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)


async def run_pipelined(dut, label):
    await reset_mult(dut)
    dut.io_enable.value = 1
    await RisingEdge(dut.clk)

    held = None
    for a, b, expected in VECTORS:
        await RisingEdge(dut.clk)
        dut.io_a.value = a & 0xFF
        dut.io_b.value = b & 0xFF
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        await ReadOnly()
        got = to_signed(int(dut.io_result.value), 16)
        assert got == expected, f"[{label}] {a} * {b}: expected {expected}, got {got}"
        held = got

    # Enable gating: the previous product must be held while enable = 0
    await RisingEdge(dut.clk)
    dut.io_enable.value = 0
    dut.io_a.value = 10
    dut.io_b.value = 10
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await ReadOnly()
    got = to_signed(int(dut.io_result.value), 16)
    assert got == held, f"[{label}] enable gating failed: expected held {held}, got {got}"

    print(f"[{label}] {len(VECTORS)} products bit-exact + enable gating validated")


async def run_combinational(dut):
    # Latency-0 top is purely combinational: clk/reset are pruned from the netlist.
    dut.io_a.value = 0
    dut.io_b.value = 0
    dut.io_enable.value = 1
    await Timer(1, units="ns")

    for a, b, expected in VECTORS:
        dut.io_a.value = a & 0xFF
        dut.io_b.value = b & 0xFF
        await Timer(1, units="ns")
        got = to_signed(int(dut.io_result.value), 16)
        assert got == expected, f"[comb latency=0] {a} * {b}: expected {expected}, got {got}"

    print(f"[comb latency=0] {len(VECTORS)} products bit-exact without pipeline stage")


@cocotb.test()
async def cocotb_hardware_mul_generic(dut):
    await run_pipelined(dut, "Generic+DSP")


@cocotb.test()
async def cocotb_hardware_mul_nodsp(dut):
    await run_pipelined(dut, "Generic no-DSP")


@cocotb.test()
async def cocotb_hardware_mul_asic(dut):
    await run_pipelined(dut, "ASIC Sky130")


@cocotb.test()
async def cocotb_hardware_mul_comb(dut):
    await run_combinational(dut)


def _run_sim(testcase, toplevel, sim_build, request=None):
    v_file = run_mill("spinalML.dsp.DspMulTest", FILTER, toplevel)

    from utils.test_layers_utils import safe_run_sim as run

    run(
        simulator="verilator",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_hardware_mul",
        testcase=testcase,
        sim_build=sim_build,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
    )


def test_hardware_mul_generic(request):
    _run_sim("cocotb_hardware_mul_generic", "DspMulGenericTestComp", "sim_build/py_hw_mul_generic", request)


def test_hardware_mul_nodsp(request):
    _run_sim("cocotb_hardware_mul_nodsp", "DspMulNoDspTestComp", "sim_build/py_hw_mul_nodsp", request)


def test_hardware_mul_asic(request):
    _run_sim("cocotb_hardware_mul_asic", "DspMulAsicTestComp", "sim_build/py_hw_mul_asic", request)


def test_hardware_mul_comb(request):
    _run_sim("cocotb_hardware_mul_comb", "DspMulCombTestComp", "sim_build/py_hw_mul_comb", request)
