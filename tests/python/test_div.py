# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb_test.simulator import run
import pytest
import math

from golden_models.dtypes import FP8_E4M3, BF16
from golden_models.ops import floatml_div, pwl_reciprocal_float, floatml_mul, reciprocal
from utils.tb_utils import run_mill, copy_roms
from utils.cocotb_helpers import run_binary_test

# OPS-12: DivOp rejects integer types at elaboration (no Qm.n scale yet);
# the I8/I16 goldens (lut/pwl_reciprocal_int) and testcases were removed here
# in the same commit. Only the FloatML paths below are covered.

@cocotb.test()
async def cocotb_div_fp8(dut):
    def expected_fn(a, b): return FP8_E4M3.from_float(floatml_div(FP8_E4M3.to_float(FP8_E4M3.from_float(a)), FP8_E4M3.to_float(FP8_E4M3.from_float(b)), FP8_E4M3))
    def true_math(a, b): return a / b if b != 0 else float('inf')
    await run_binary_test(dut, "Div", "FP8", FP8_E4M3, [(2.5, 1.5), (-1.5, 2.5)], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_math)

@cocotb.test()
async def cocotb_div_bf16(dut):
    def expected_fn(a, b):
        # BF16 uses PWL for reciprocal
        inv_b_bits = pwl_reciprocal_float(b, BF16)
        inv_b_val = BF16.to_float(inv_b_bits)
        res_bits = BF16.from_float(floatml_mul(BF16.to_float(BF16.from_float(a)), inv_b_val, BF16))
        return res_bits
    
    def true_math(a, b): return a / b if b != 0 else float('inf')
    await run_binary_test(dut, "Div", "BF16", BF16, [(20.25, 2.0), (100.0, -100.0)], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_math, edge_cases=[(100.0, -100.0)])

def run_div_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.DivTest", dtype_filter, "DivTestComp")
    build_dir = f"sim_build/div_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="DivTestComp",
        module="test_div",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_div_fp8(request): run_div_sim("FP8", "cocotb_div_fp8", request)
def test_div_bf16(request): run_div_sim("BF16", "cocotb_div_bf16", request)
