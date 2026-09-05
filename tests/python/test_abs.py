# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb_test.simulator import run
import pytest
import os
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import abs_hw
from utils.tb_utils import run_mill, copy_roms
from utils.tb_utils import cleanup_verilog
from utils.cocotb_helpers import run_unary_test

def true_abs(val):
    return abs(val)

@cocotb.test()
async def cocotb_abs_i8(dut):
    def expected_fn(val):
        return I8.from_float(abs_hw(val, I8))
    await run_unary_test(dut, "Abs", "I8", I8, [-100.0, 50.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_abs, edge_cases=[-128.0])

@cocotb.test()
async def cocotb_abs_fp8(dut):
    def expected_fn(val):
        return FP8_E4M3.from_float(abs_hw(val, FP8_E4M3))
    await run_unary_test(dut, "Abs", "FP8", FP8_E4M3, [-2.5, 4.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_abs, edge_cases=[-448.0])

@cocotb.test()
async def cocotb_abs_i16(dut):
    def expected_fn(val):
        return I16.from_float(abs_hw(val, I16))
    await run_unary_test(dut, "Abs", "I16", I16, [-30000.0, 15000.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_abs, edge_cases=[-32768.0])

@cocotb.test()
async def cocotb_abs_bf16(dut):
    def expected_fn(val):
        return BF16.from_float(abs_hw(val, BF16))
    await run_unary_test(dut, "Abs", "BF16", BF16, [-256.0, 128.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_abs)

def run_abs_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.AbsTest", dtype_filter, "AbsTestComp")
    build_dir = f"sim_build/abs_{dtype_filter.lower()}"
    copy_roms(build_dir)
    
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="AbsTestComp",
        module="test_abs",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_abs_i8(request): run_abs_sim("I8", "cocotb_abs_i8", request)
def test_abs_fp8(request): run_abs_sim("FP8", "cocotb_abs_fp8", request)
def test_abs_i16(request): run_abs_sim("I16", "cocotb_abs_i16", request)
def test_abs_bf16(request): run_abs_sim("BF16", "cocotb_abs_bf16", request)
