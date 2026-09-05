# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb_test.simulator import run
import pytest
import math

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import log_b, pwl_log_int, pwl_log_float
from utils.tb_utils import run_mill, copy_roms
from utils.tb_utils import cleanup_verilog
from utils.cocotb_helpers import run_unary_test

# =========================================================================
# Cocotb Test Logic
# =========================================================================

@cocotb.test()
async def cocotb_log_i8(dut):
    def expected_fn(val):
        return I8.from_float(log_b(I8.to_float(I8.from_float(val))))
    await run_unary_test(dut, "Log", "I8", I8, [2.0, 10.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=log_b, edge_cases=[0.0, -1.0])

@cocotb.test()
async def cocotb_log_fp8(dut):
    def expected_fn(val):
        return FP8_E4M3.from_float(log_b(FP8_E4M3.to_float(FP8_E4M3.from_float(val))))
    await run_unary_test(dut, "Log", "FP8", FP8_E4M3, [2.0, 10.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=log_b, edge_cases=[0.0, -1.0])

@cocotb.test()
async def cocotb_log_i16(dut):
    def expected_fn(val):
        return pwl_log_int(val, 16, index_bits=8)
    await run_unary_test(dut, "Log", "I16", I16, [100.0, 1000.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=log_b, edge_cases=[0.0, -1.0])

@cocotb.test()
async def cocotb_log_bf16(dut):
    def expected_fn(val):
        return pwl_log_float(val, BF16, index_bits=8)
    await run_unary_test(dut, "Log", "BF16", BF16, [2.5, 12.5], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=log_b, edge_cases=[0.0, -1.0])

@cocotb.test()
async def cocotb_log_bf16_base10(dut):
    def expected_fn(val):
        return pwl_log_float(val, BF16, base=10.0, index_bits=8)
    await run_unary_test(dut, "Log", "BF16.base10", BF16, [2.5, 12.5], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=lambda x: log_b(x, 10.0), edge_cases=[0.0, -1.0])

# =========================================================================
# Pytest Launchers
# =========================================================================

def run_log_sim(dtype_filter, toplevel, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.LogTest", dtype_filter, toplevel)
    build_dir = f"sim_build/{toplevel.lower()}"
    copy_roms(build_dir)
    
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_log",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_log_i8(request): run_log_sim("I8", "LogTestComp", "cocotb_log_i8", request)
def test_log_fp8(request): run_log_sim("FP8", "LogTestComp", "cocotb_log_fp8", request)
def test_log_i16(request): run_log_sim("I16", "LogTestComp", "cocotb_log_i16", request)
def test_log_bf16(request): run_log_sim("BF16 default", "LogTestComp", "cocotb_log_bf16", request)
def test_log_bf16_base10(request): run_log_sim("base10", "LogTestComp10", "cocotb_log_bf16_base10", request)