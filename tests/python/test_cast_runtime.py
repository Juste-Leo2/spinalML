# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

# Python/Cocotb co-simulation of the runtime-programmable dequantizing Cast
# (Refactoring: Cast(..., runtimeScale = true) + Accelerator CSR 0x30).
#
# The generated `CastRuntimeScaleTestComp` is purely combinational:
#   io_a_stream (I8, lanes = 4) -> Float.fromSInt -> Float.mul(scale)
#   io_c_stream (BF16, lanes = 4)
#   io_scale    (Bits(32), only the low 16 bits are decoded as BF16)
# It is the only dynamic verification of the runtime scale path: the Scala
# suite only generates its Verilog.

import random

import cocotb
from cocotb.triggers import Timer
from cocotb_test.simulator import run

from golden_models.dtypes import BF16
from golden_models.ops import cast_hw, floatml_mul
from utils.tb_utils import run_mill, seed_random

LANES = 4
IN_BITS = 8


def expected_bits(x, scale_bits):
    """Hardware chain: Float.fromSInt(x) then Float.mul(cast, scale)."""
    cast = cast_hw(x, IN_BITS, BF16)
    value = BF16.to_float(cast)
    scale = BF16.to_float(scale_bits)
    return BF16.from_float(floatml_mul(value, scale, BF16))


async def apply_row(dut, row, scale_bits):
    """Drive one combinational transaction and read back the BF16 outputs."""
    dut.io_scale.value = scale_bits
    for l in range(LANES):
        getattr(dut, f"io_a_stream_payload_{l}").value = row[l] & 0xFF

    dut.io_a_stream_valid.value = 1
    dut.io_c_stream_ready.value = 1
    await Timer(1, units="ns")

    assert int(dut.io_a_stream_ready.value) == 1, "io_a_stream_ready must follow io_c_stream_ready"
    assert int(dut.io_c_stream_valid.value) == 1, "io_c_stream_valid must follow io_a_stream_valid"

    out = []
    for l in range(LANES):
        sign = int(getattr(dut, f"io_c_stream_payload_{l}_sign").value)
        exp = int(getattr(dut, f"io_c_stream_payload_{l}_exponent").value)
        mant = int(getattr(dut, f"io_c_stream_payload_{l}_mantissa").value)
        out.append((sign << (BF16.exp_bits + BF16.mant_bits)) | (exp << BF16.mant_bits) | mant)

    dut.io_a_stream_valid.value = 0
    await Timer(1, units="ns")
    return out


def check_row(row, scale_bits, out):
    expected = [expected_bits(x, scale_bits) for x in row]
    for i, (got, exp) in enumerate(zip(out, expected)):
        if got != exp:
            raise AssertionError(
                f"Runtime scale mismatch at lane {i}: input {row[i]}, scale bits 0x{scale_bits:04X}: "
                f"got bits {got} ({BF16.to_float(got)}) instead of {exp} ({BF16.to_float(exp)})"
            )


@cocotb.test()
async def cocotb_cast_runtime_static_scales(dut):
    seed_random()
    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 1
    dut.io_scale.value = 0
    for l in range(LANES):
        getattr(dut, f"io_a_stream_payload_{l}").value = 0
    await Timer(1, units="ns")

    scales = [1.0, 0.5, 2.0, -1.0, -0.5, 0.25, 3.0]
    rows = [
        [0, 1, -1, 127],
        [-128, 100, -100, 42],
        [-42, 7, -7, 64],
        [random.randint(-128, 127) for _ in range(LANES)],
        [random.randint(-128, 127) for _ in range(LANES)],
    ]

    for scale in scales:
        scale_bits = BF16.from_float(scale)
        for row in rows:
            out = await apply_row(dut, row, scale_bits)
            check_row(row, scale_bits, out)

    print(f"Runtime scale Cast validated on {len(scales)} static scales x {len(rows)} rows")


@cocotb.test()
async def cocotb_cast_runtime_dynamic_scale(dut):
    # The scale port is combinational: it must be re-evaluated on every row.
    seed_random()
    dut.io_c_stream_ready.value = 1
    await Timer(1, units="ns")

    cases = [
        ([2, -2, 4, -4], 0.5),
        ([2, -2, 4, -4], 2.0),
        ([2, -2, 4, -4], -1.0),
        ([127, -128, 1, -1], 0.0),
        ([127, -128, 1, -1], 1.0),
        ([5, 10, -15, 20], -0.25),
        ([5, 10, -15, 20], 4.0),
    ]

    for row, scale in cases:
        scale_bits = BF16.from_float(scale)
        out = await apply_row(dut, row, scale_bits)
        check_row(row, scale_bits, out)

    print(f"Runtime scale Cast validated with {len(cases)} consecutive dynamic scale changes")


@cocotb.test()
async def cocotb_cast_runtime_upper_bits_ignored(dut):
    # Only the low expBits+mantBits+1 = 16 bits of io_scale feed the BF16 scale.
    seed_random()
    dut.io_c_stream_ready.value = 1
    await Timer(1, units="ns")

    row = [3, -3, 100, -100]
    for scale in [0.5, 1.5, -2.0]:
        scale_bits = BF16.from_float(scale)
        clean = await apply_row(dut, row, scale_bits)
        dirty = await apply_row(dut, row, 0xDEAD0000 | scale_bits)
        assert clean == dirty, (
            f"Upper 16 bits of io_scale leaked into the result for scale bits 0x{scale_bits:04X}: "
            f"{clean} != {dirty}"
        )

    print("Runtime scale Cast validated: upper 16 bits of io_scale are ignored")


def _run_sim(testcase, sim_build, request=None):
    v_file = run_mill("spinalML.ops.CastTest", "runtime", "CastRuntimeScaleTestComp")
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"

    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="CastRuntimeScaleTestComp",
        module="test_cast_runtime",
        testcase=testcase,
        simulator="verilator",
        sim_build=sim_build,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag},
    )


def test_cast_runtime_static(request):
    _run_sim("cocotb_cast_runtime_static_scales", "sim_build/py_cast_runtime_static", request)


def test_cast_runtime_dynamic(request):
    _run_sim("cocotb_cast_runtime_dynamic_scale", "sim_build/py_cast_runtime_dynamic", request)


def test_cast_runtime_upper_bits(request):
    _run_sim("cocotb_cast_runtime_upper_bits_ignored", "sim_build/py_cast_runtime_upper", request)
