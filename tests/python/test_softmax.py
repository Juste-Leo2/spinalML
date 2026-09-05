# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb_test.simulator import run
import pytest
import os
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import exp, pwl_exp_int, pwl_exp_float
from golden_models.ops import pwl_reciprocal_int, pwl_reciprocal_float
from golden_models.ops import floatml_add, floatml_mul
from utils.tb_utils import run_mill, copy_roms
from utils.tb_utils import cleanup_verilog
from utils.cocotb_helpers import run_softmax_test

def true_softmax(val_arr):
    x = np.array(val_arr, dtype=np.float64)
    max_val = np.max(x)
    exp_vals = np.exp(x - max_val)
    sum_val = np.sum(exp_vals)
    return exp_vals / sum_val if sum_val != 0 else np.zeros_like(exp_vals)

def softmax_hw(val_arr, dtype):
    is_float = hasattr(dtype, 'exp_bits')
    
    # HW uses bitwise operations, so simulate exact bit representation flow
    bits_arr = [dtype.from_float(v) for v in val_arr]
    hw_vals = [dtype.to_float(b) for b in bits_arr]
    
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0) + 1)
    
    # 1. Max
    max_val = max(hw_vals)
    
    # 2. Sub & 3. Exp
    exp_vals = []
    for v in hw_vals:
        shifted = v - max_val
        if is_float:
            # FloatML add handles sub
            shifted = floatml_add(v, -max_val, dtype)
            
        # Exp HW approximation
        if is_float:
            if bit_width <= 8:
                # LUT for 8-bit Float
                e_val = exp(shifted, dtype)
                e_val = dtype.to_float(dtype.from_float(e_val))
            else:
                e_bits = pwl_exp_float(shifted, dtype, index_bits=8)
                e_val = dtype.to_float(e_bits)
        else:
            if bit_width <= 8:
                e_bits = pwl_exp_int(shifted, bit_width, index_bits=8)
                e_val = dtype.to_float(e_bits)
            else:
                e_bits = pwl_exp_int(shifted, bit_width, index_bits=8)
                e_val = dtype.to_float(e_bits)
        exp_vals.append(e_val)
        
    # 4. Adder Tree
    sum_val = exp_vals[0]
    for i in range(1, len(exp_vals)):
        if is_float:
            sum_val = floatml_add(sum_val, exp_vals[i], dtype)
        else:
            sum_val = dtype.to_float(dtype.from_float(sum_val + exp_vals[i]))
            
    # 5. Reciprocal HW approximation
    if is_float:
        if bit_width <= 8:
            r_bits = pwl_reciprocal_float(sum_val, dtype, index_bits=8)
            r_val = dtype.to_float(r_bits)
        else:
            r_bits = pwl_reciprocal_float(sum_val, dtype, index_bits=8)
            r_val = dtype.to_float(r_bits)
    else:
        if bit_width <= 8:
            r_bits = pwl_reciprocal_int(sum_val, bit_width, index_bits=8)
            r_val = dtype.to_float(r_bits)
        else:
            r_bits = pwl_reciprocal_int(sum_val, bit_width, index_bits=8)
            r_val = dtype.to_float(r_bits)
            
    # 6. Final Mul
    final_vals = []
    for e in exp_vals:
        if is_float:
            final_vals.append(floatml_mul(e, r_val, dtype))
        else:
            final_vals.append(dtype.to_float(dtype.from_float(e * r_val)))
            
    return [dtype.from_float(v) for v in final_vals]

@cocotb.test()
async def cocotb_softmax_i8(dut):
    def expected_fn(val_arr):
        return softmax_hw(val_arr, I8)
    await run_softmax_test(dut, "Softmax", "I8", I8, [
        (0.0, 1.0, -1.0, 2.0),
        (10.0, 10.0, 10.0, 10.0),
        (-50.0, 0.0, 50.0, 20.0),
        (0.0, 0.0, 0.0, 0.0), # Edge case: uniform distribution
        (100.0, -100.0, -100.0, -100.0), # Edge case: one extreme dominant
        (-120.0, -120.0, -120.0, -120.0) # Edge case: all extremely negative
    ], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_softmax)

@cocotb.test()
async def cocotb_softmax_fp8(dut):
    def expected_fn(val_arr):
        return softmax_hw(val_arr, FP8_E4M3)
    await run_softmax_test(dut, "Softmax", "FP8", FP8_E4M3, [
        (0.0, 1.0, -1.0, 2.0),
        (4.0, 4.0, 4.0, 4.0),
        (-2.5, 0.0, 2.5, 1.0),
        (0.0, 0.0, 0.0, 0.0), # Edge case
        (15.0, -15.0, -15.0, -15.0), # Edge case
        (-10.0, -10.0, -10.0, -10.0) # Edge case
    ], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_softmax)

@cocotb.test()
async def cocotb_softmax_i16(dut):
    def expected_fn(val_arr):
        return softmax_hw(val_arr, I16)
    await run_softmax_test(dut, "Softmax", "I16", I16, [
        (0.0, 100.0, -100.0, 200.0),
        (1000.0, 1000.0, 1000.0, 1000.0),
        (-500.0, 0.0, 500.0, 200.0),
        (0.0, 0.0, 0.0, 0.0), # Edge case
        (5000.0, -5000.0, -5000.0, -5000.0), # Edge case
        (-10000.0, -10000.0, -10000.0, -10000.0) # Edge case
    ], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_softmax)

@cocotb.test()
async def cocotb_softmax_bf16(dut):
    def expected_fn(val_arr):
        return softmax_hw(val_arr, BF16)
    await run_softmax_test(dut, "Softmax", "BF16", BF16, [
        (0.0, 1.0, -1.0, 2.0),
        (10.0, 10.0, 10.0, 10.0),
        (-5.0, 0.0, 5.0, 2.0),
        (0.0, 0.0, 0.0, 0.0), # Edge case
        (50.0, -50.0, -50.0, -50.0), # Edge case
        (-30.0, -30.0, -30.0, -30.0) # Edge case
    ], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_softmax)


def run_softmax_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.activations.SoftmaxTest", dtype_filter, "SoftmaxTestComp")
    build_dir = f"sim_build/softmax_{dtype_filter.lower()}"
    copy_roms(build_dir)
    
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="SoftmaxTestComp",
        module="test_softmax",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_softmax_i8(request): run_softmax_sim("I8", "cocotb_softmax_i8", request)
def test_softmax_fp8(request): run_softmax_sim("FP8", "cocotb_softmax_fp8", request)
def test_softmax_i16(request): run_softmax_sim("I16", "cocotb_softmax_i16", request)
def test_softmax_bf16(request): run_softmax_sim("BF16", "cocotb_softmax_bf16", request)
