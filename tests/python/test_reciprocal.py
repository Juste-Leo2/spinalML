# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb_test.simulator import run
import pytest
import os
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import pwl_reciprocal_int, pwl_reciprocal_float
from utils.tb_utils import run_mill, copy_roms
from utils.tb_utils import cleanup_verilog
from utils.cocotb_helpers import run_unary_test

def true_reciprocal(val):
    return 1.0 / (val + (1e-9 if val >= 0 else -1e-9))

@cocotb.test()
async def cocotb_reciprocal_i8(dut):
    def expected_fn(val):
        return pwl_reciprocal_int(val, 8, index_bits=8)
    await run_unary_test(dut, "Reciprocal", "I8", I8, [2.0, 10.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_reciprocal, edge_cases=[127.0])

@cocotb.test()
async def cocotb_reciprocal_fp8(dut):
    def expected_fn(val):
        return pwl_reciprocal_float(val, FP8_E4M3, index_bits=8)
    await run_unary_test(dut, "Reciprocal", "FP8", FP8_E4M3, [2.0, 8.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_reciprocal, edge_cases=[448.0])

@cocotb.test()
async def cocotb_reciprocal_i16(dut):
    def expected_fn(val):
        return pwl_reciprocal_int(val, 16, index_bits=8)
    await run_unary_test(dut, "Reciprocal", "I16", I16, [100.0, 1000.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_reciprocal, edge_cases=[32767.0])

@cocotb.test()
async def cocotb_reciprocal_bf16(dut):
    def expected_fn(val):
        return pwl_reciprocal_float(val, BF16, index_bits=8)
    await run_unary_test(dut, "Reciprocal", "BF16", BF16, [2.0, 50.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_reciprocal)

def run_reciprocal_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.ReciprocalTest", dtype_filter, "ReciprocalTestComp")
    build_dir = f"sim_build/reciprocal_{dtype_filter.lower()}"
    copy_roms(build_dir)
    
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="ReciprocalTestComp",
        module="test_reciprocal",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_reciprocal_i8(request): run_reciprocal_sim("I8", "cocotb_reciprocal_i8", request)
def test_reciprocal_fp8(request): run_reciprocal_sim("FP8", "cocotb_reciprocal_fp8", request)
def test_reciprocal_i16(request): run_reciprocal_sim("I16", "cocotb_reciprocal_i16", request)
def test_reciprocal_bf16(request): run_reciprocal_sim("BF16", "cocotb_reciprocal_bf16", request)
