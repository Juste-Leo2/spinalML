# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb_test.simulator import run
import pytest
import os
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import pwl_sqrt_int, pwl_sqrt_float
from utils.tb_utils import run_mill, copy_roms
from utils.tb_utils import cleanup_verilog
from utils.cocotb_helpers import run_unary_test

def true_sqrt(val):
    return np.sqrt(abs(val))

@cocotb.test()
async def cocotb_sqrt_i8(dut):
    def expected_fn(val):
        return pwl_sqrt_int(val, 8, index_bits=8)
    await run_unary_test(dut, "Sqrt", "I8", I8, [16.0, 64.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_sqrt, edge_cases=[127.0])

@cocotb.test()
async def cocotb_sqrt_fp8(dut):
    def expected_fn(val):
        return pwl_sqrt_float(val, FP8_E4M3, index_bits=8)
    await run_unary_test(dut, "Sqrt", "FP8", FP8_E4M3, [4.0, 16.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_sqrt, edge_cases=[448.0])

@cocotb.test()
async def cocotb_sqrt_i16(dut):
    def expected_fn(val):
        return pwl_sqrt_int(val, 16, index_bits=8)
    await run_unary_test(dut, "Sqrt", "I16", I16, [144.0, 1024.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_sqrt, edge_cases=[32767.0])

@cocotb.test()
async def cocotb_sqrt_bf16(dut):
    def expected_fn(val):
        return BF16.from_float(true_sqrt(val))
    await run_unary_test(dut, "Sqrt", "BF16", BF16, [2.0, 100.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_sqrt)

def run_sqrt_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.SqrtTest", dtype_filter, "SqrtTestComp")
    build_dir = f"sim_build/sqrt_{dtype_filter.lower()}"
    copy_roms(build_dir)
    
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="SqrtTestComp",
        module="test_sqrt",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_sqrt_i8(request): run_sqrt_sim("I8", "cocotb_sqrt_i8", request)
def test_sqrt_fp8(request): run_sqrt_sim("FP8", "cocotb_sqrt_fp8", request)
def test_sqrt_i16(request): run_sqrt_sim("I16", "cocotb_sqrt_i16", request)
def test_sqrt_bf16(request): run_sqrt_sim("BF16", "cocotb_sqrt_bf16", request)
