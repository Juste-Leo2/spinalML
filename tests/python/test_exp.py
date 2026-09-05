# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb_test.simulator import run
import pytest
import math

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import exp, pwl_exp_int, pwl_exp_float
from utils.tb_utils import run_mill, copy_roms
from utils.tb_utils import cleanup_verilog
from utils.cocotb_helpers import run_unary_test

# =========================================================================
# Cocotb Test Logic
# =========================================================================

@cocotb.test()
async def cocotb_exp_i8(dut):
    def expected_fn(val):
        return I8.from_float(exp(I8.to_float(I8.from_float(val)), I8))
    await run_unary_test(dut, "Exp", "I8", I8, [1.0, -1.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=math.exp, edge_cases=[-1.0])

@cocotb.test()
async def cocotb_exp_fp8(dut):
    def expected_fn(val):
        return FP8_E4M3.from_float(exp(FP8_E4M3.to_float(FP8_E4M3.from_float(val)), FP8_E4M3))
    await run_unary_test(dut, "Exp", "FP8", FP8_E4M3, [1.0, -1.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=math.exp, edge_cases=[-1.0])

@cocotb.test()
async def cocotb_exp_i16(dut):
    def expected_fn(val):
        return pwl_exp_int(val, 16, index_bits=8)
    await run_unary_test(dut, "Exp", "I16", I16, [2.5, -4.2], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=math.exp, edge_cases=[-4.2])

@cocotb.test()
async def cocotb_exp_bf16(dut):
    def expected_fn(val):
        return pwl_exp_float(val, BF16, index_bits=8)
    await run_unary_test(dut, "Exp", "BF16", BF16, [2.5, 12.5], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=math.exp, edge_cases=[12.5])

# =========================================================================
# Pytest Launchers
# =========================================================================

def run_exp_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.ExpTest", dtype_filter, "ExpTestComp")
    build_dir = f"sim_build/exp_{dtype_filter.lower()}"
    copy_roms(build_dir)
    
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="ExpTestComp",
        module="test_exp",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_exp_i8(request): run_exp_sim("I8", "cocotb_exp_i8", request)
def test_exp_fp8(request): run_exp_sim("FP8", "cocotb_exp_fp8", request)
def test_exp_i16(request): run_exp_sim("I16", "cocotb_exp_i16", request)
def test_exp_bf16(request): run_exp_sim("BF16", "cocotb_exp_bf16", request)
