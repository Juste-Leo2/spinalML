# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.triggers import Timer
from cocotb_test.simulator import run
import pytest
import random

from golden_models.dtypes import BF16, I8, I16, I32
from golden_models.ops import cast_hw
from utils.tb_utils import run_mill, copy_roms, seed_random

seed_random()


async def run_cast_test(dut, op_name, dtype_name, in_dtype, lanes=4):
    # CastOp is purely combinational: no clk/reset in the Verilog top.
    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 1

    in_bits = in_dtype.bit_width
    min_v = -(1 << (in_bits - 1))
    max_v = (1 << (in_bits - 1)) - 1

    edges = [0, 1, -1, max_v, min_v, 100, -100]
    rows = [edges[i * lanes:(i + 1) * lanes] for i in range((len(edges) + lanes - 1) // lanes)]
    while len(rows[-1]) < lanes:
        rows[-1].append(0)
    for _ in range(2):
        rows.append([random.randint(min_v, max_v) for _ in range(lanes)])
    flat_in = [v for row in rows for v in row]

    flat_out = []
    for row in rows:
        for l in range(lanes):
            getattr(dut, f"io_a_stream_payload_{l}").value = in_dtype.from_float(row[l])
        dut.io_a_stream_valid.value = 1
        await Timer(1, units="ns")
        for l in range(lanes):
            sign = int(getattr(dut, f"io_c_stream_payload_{l}_sign").value)
            exp = int(getattr(dut, f"io_c_stream_payload_{l}_exponent").value)
            mant = int(getattr(dut, f"io_c_stream_payload_{l}_mantissa").value)
            flat_out.append((sign << (BF16.exp_bits + BF16.mant_bits)) | (exp << BF16.mant_bits) | mant)
        dut.io_a_stream_valid.value = 0
        await Timer(1, units="ns")

    expected = [cast_hw(x, in_bits, BF16) for x in flat_in]
    for i, (out_bits, exp_bits) in enumerate(zip(flat_out, expected)):
        assert out_bits == exp_bits, f"HW Mismatch for {op_name} at element {i}: got bits {out_bits} instead of {exp_bits}"

    dut._log.info(f"[{op_name}][{dtype_name}->BF16] Passed Cast Check")


@cocotb.test()
async def cocotb_cast_i8(dut):
    await run_cast_test(dut, "Cast", "I8", I8)

@cocotb.test()
async def cocotb_cast_i16(dut):
    await run_cast_test(dut, "Cast", "I16", I16)

@cocotb.test()
async def cocotb_cast_i32(dut):
    await run_cast_test(dut, "Cast", "I32", I32)


def run_cast_sim(dtype_filter, testcase_name, request=None):
    toplevel = "CastTestComp"
    v_file = run_mill("spinalML.ops.CastTest", dtype_filter, toplevel)
    build_dir = f"sim_build/cast_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_cast",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_cast_i8(request): run_cast_sim("I8", "cocotb_cast_i8", request)
def test_cast_i16(request): run_cast_sim("I16", "cocotb_cast_i16", request)
def test_cast_i32(request): run_cast_sim("I32", "cocotb_cast_i32", request)