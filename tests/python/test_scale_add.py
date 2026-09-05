# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb_test.simulator import run
import pytest
import os
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import scale_add_hw
from utils.tb_utils import run_mill, copy_roms
from utils.tb_utils import cleanup_verilog
from utils.cocotb_helpers import run_ternary_test

def true_scale_add(x, a, b):
    return (x * a) + b

@cocotb.test()
async def cocotb_scale_add_i8(dut):
    def expected_fn(x, a, b):
        return I8.from_float(scale_add_hw(x, a, b, I8))
    await run_ternary_test(dut, "ScaleAdd", "I8", I8, [(2.0, 3.0, 4.0), (10.0, -2.0, 5.0)], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_scale_add, edge_cases=[(15.0, 8.0, 10.0)])

@cocotb.test()
async def cocotb_scale_add_fp8(dut):
    def expected_fn(x, a, b):
        return FP8_E4M3.from_float(scale_add_hw(x, a, b, FP8_E4M3))
    await run_ternary_test(dut, "ScaleAdd", "FP8", FP8_E4M3, [(1.5, 2.0, 0.5), (4.0, -1.0, 2.0)], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_scale_add, edge_cases=[(16.0, 16.0, 16.0)])

@cocotb.test()
async def cocotb_scale_add_i16(dut):
    def expected_fn(x, a, b):
        return I16.from_float(scale_add_hw(x, a, b, I16))
    await run_ternary_test(dut, "ScaleAdd", "I16", I16, [(100.0, 20.0, 50.0), (500.0, -10.0, 100.0)], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_scale_add, edge_cases=[(200.0, 150.0, 500.0)])

@cocotb.test()
async def cocotb_scale_add_bf16(dut):
    def expected_fn(x, a, b):
        return BF16.from_float(scale_add_hw(x, a, b, BF16))
    await run_ternary_test(dut, "ScaleAdd", "BF16", BF16, [(2.0, 3.0, 4.0), (5.0, 2.5, -1.0)], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_scale_add)

def run_scale_add_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.ScaleAddTest", dtype_filter, "ScaleAddTestComp")
    build_dir = f"sim_build/scale_add_{dtype_filter.lower()}"
    copy_roms(build_dir)
    
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="ScaleAddTestComp",
        module="test_scale_add",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_scale_add_i8(request): run_scale_add_sim("I8", "cocotb_scale_add_i8", request)
def test_scale_add_fp8(request): run_scale_add_sim("FP8", "cocotb_scale_add_fp8", request)
def test_scale_add_i16(request): run_scale_add_sim("I16", "cocotb_scale_add_i16", request)
def test_scale_add_bf16(request): run_scale_add_sim("BF16", "cocotb_scale_add_bf16", request)
