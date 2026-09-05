# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb_test.simulator import run
import pytest

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import floatml_sub
from utils.tb_utils import run_mill, copy_roms
from utils.cocotb_helpers import run_binary_test

@cocotb.test()
async def cocotb_sub_i8(dut):
    def expected_fn(a, b): return (I8.from_float(a) - I8.from_float(b)) & ((1 << I8.bit_width) - 1)
    await run_binary_test(dut, "Sub", "I8", I8, [(5.0, 2.0), (-120.0, 5.0)], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=lambda a,b: a-b, edge_cases=[(-120.0, 5.0)])

@cocotb.test()
async def cocotb_sub_fp8(dut):
    def expected_fn(a, b): return FP8_E4M3.from_float(floatml_sub(FP8_E4M3.to_float(FP8_E4M3.from_float(a)), FP8_E4M3.to_float(FP8_E4M3.from_float(b)), FP8_E4M3))
    await run_binary_test(dut, "Sub", "FP8", FP8_E4M3, [(2.5, 1.5), (-1.5, 2.5)], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=lambda a,b: a-b)

@cocotb.test()
async def cocotb_sub_i16(dut):
    def expected_fn(a, b): return (I16.from_float(a) - I16.from_float(b)) & ((1 << I16.bit_width) - 1)
    await run_binary_test(dut, "Sub", "I16", I16, [(20.0, 10.0), (-32000.0, 700.0)], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=lambda a,b: a-b, edge_cases=[(-32000.0, 700.0)])

@cocotb.test()
async def cocotb_sub_bf16(dut):
    def expected_fn(a, b): return BF16.from_float(floatml_sub(BF16.to_float(BF16.from_float(a)), BF16.to_float(BF16.from_float(b)), BF16))
    await run_binary_test(dut, "Sub", "BF16", BF16, [(20.25, 10.5), (100.0, -100.0)], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=lambda a,b: a-b, edge_cases=[(100.0, -100.0)])

def run_sub_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.SubTest", dtype_filter, "SubTestComp")
    build_dir = f"sim_build/sub_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="SubTestComp",
        module="test_sub",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_sub_i8(request): run_sub_sim("I8", "cocotb_sub_i8", request)
def test_sub_fp8(request): run_sub_sim("FP8", "cocotb_sub_fp8", request)
def test_sub_i16(request): run_sub_sim("I16", "cocotb_sub_i16", request)
def test_sub_bf16(request): run_sub_sim("BF16", "cocotb_sub_bf16", request)
