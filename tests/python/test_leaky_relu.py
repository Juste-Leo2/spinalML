# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest
import os
import random
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import leaky_relu_hw
from utils.tb_utils import run_mill, copy_roms, seed_random, SEED
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, DEFAULT_NUM_TRIALS

seed_random()

async def run_leaky_relu_test(dut, op_name, dtype_name, dtype, X, shift, is_floatml, collect=None):
    # Combinatorial circuit, no clk/reset needed
    dut.io_x_stream_valid.value = 0
    dut.io_y_stream_ready.value = 1
    
    Y_out = []
    Y_out_bits = []
    
    lanes = 2
    chunks = (len(X) + lanes - 1) // lanes
    flattened_x = [x[0] for x in X]
    
    for chunk in range(chunks):
        if is_floatml:
            for l in range(lanes):
                idx = chunk * lanes + l
                val = flattened_x[idx] if idx < len(flattened_x) else 0.0
                val_bits = dtype.from_float(val)
                sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                mant = val_bits & ((1 << dtype.mant_bits) - 1)
                getattr(dut, f"io_x_stream_payload_{l}_sign").value = sign
                getattr(dut, f"io_x_stream_payload_{l}_exponent").value = exp_val
                getattr(dut, f"io_x_stream_payload_{l}_mantissa").value = mant
        else:
            for l in range(lanes):
                idx = chunk * lanes + l
                val = flattened_x[idx] if idx < len(flattened_x) else 0.0
                val_bits = dtype.from_float(val)
                getattr(dut, f"io_x_stream_payload_{l}").value = val_bits
                
        dut.io_x_stream_valid.value = 1
        from cocotb.triggers import Timer
        await Timer(10, units="ns")
        
        for l in range(lanes):
            idx = chunk * lanes + l
            if idx < len(flattened_x):
                if is_floatml:
                    out_sign = int(getattr(dut, f"io_y_stream_payload_{l}_sign").value)
                    out_exp = int(getattr(dut, f"io_y_stream_payload_{l}_exponent").value)
                    out_mant = int(getattr(dut, f"io_y_stream_payload_{l}_mantissa").value)
                    bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
                else:
                    bits = int(getattr(dut, f"io_y_stream_payload_{l}").value)
                    
                Y_out_bits.append([bits])
                Y_out.append([dtype.to_float(bits)])
                
    dut.io_x_stream_valid.value = 0

    # True Math
    X_np = np.array([x[0] for x in X])
    # The HW uses `shift` for negative slope.
    # We will compute the exact factor: if floatml, shift is substracted from exponent, so * (2^-shift)
    # If integer, arithmetic right shift.
    alpha = 2.0 ** (-shift)
    
    Y_true = [[float(x) if x >= 0 else float(x * alpha)] for x in X_np]
    # For integers, negative numbers are arithmetic right shifted by shift
    if not is_floatml:
        Y_true = []
        for x in X_np:
            if x >= 0:
                Y_true.append([float(x)])
            else:
                x_int = int(x)
                if hasattr(dtype, 'bit_width'):
                    # To accurately reflect the Python true math error for integers:
                    # we must compute the expected right shift value for negative numbers.
                    sval = int(dtype.from_float(x))
                    if sval & (1 << (dtype.bit_width - 1)):
                        sval = sval - (1 << dtype.bit_width)
                    shifted = sval >> shift
                    out_bits = shifted & ((1 << dtype.bit_width) - 1)
                    Y_true.append([dtype.to_float(out_bits)])
                else:
                    Y_true.append([float(x_int >> shift)])

    
    if collect is not None:
        collect["out"].append(Y_out)
        collect["true"].append(Y_true)
    else:
        log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, Y_out, Y_true)
        dut._log.info(log_msg)
    
    # Exact HW Math
    Y_expected = leaky_relu_hw(X, shift, dtype)
    
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    for m in range(len(X)):
        for n in range(1):
            exp_val = Y_expected[m][n]
            exp_bits = dtype.from_float(exp_val)
            out_bits = Y_out_bits[m][0]
            out_val = Y_out[m][0]
            if bit_width > 8 and is_floatml:
                assert abs(out_bits - exp_bits) <= 1, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {dtype.to_float(exp_bits)}"
            else:
                assert out_bits == exp_bits, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {dtype.to_float(exp_bits)}"


# ----------------- COCOTB TESTS -----------------
@cocotb.test()
async def cocotb_leaky_relu_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 1), 10.0, True)
        await run_leaky_relu_test(dut, "LeakyReLU", "I8", I8, X, 1, False, collect=collect)
    details = f"N=4, shift=1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LeakyReLU", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_leaky_relu_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 1), 5.0, False)
        await run_leaky_relu_test(dut, "LeakyReLU", "FP8", FP8_E4M3, X, 1, True, collect=collect)
    details = f"N=4, shift=1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LeakyReLU", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_leaky_relu_i16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 1), 100.0, True)
        await run_leaky_relu_test(dut, "LeakyReLU", "I16", I16, X, 1, False, collect=collect)
    details = f"N=4, shift=1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LeakyReLU", "I16", I16, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_leaky_relu_bf16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 1), 10.0, False)
        await run_leaky_relu_test(dut, "LeakyReLU", "BF16", BF16, X, 1, True, collect=collect)
    details = f"N=4, shift=1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LeakyReLU", "BF16", BF16, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)


# ----------------- PYTEST RUNNERS -----------------
def run_leaky_relu_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.activations.LeakyReLUTest", dtype_filter, "LeakyReLUTestComp")
    build_dir = f"sim_build/leakyrelu_leakyrelutestcomp_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="LeakyReLUTestComp",
        module="test_leaky_relu",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_pytest_leaky_relu_i8(request): run_leaky_relu_sim("I8", "cocotb_leaky_relu_i8", request)
def test_pytest_leaky_relu_fp8(request): run_leaky_relu_sim("FP8", "cocotb_leaky_relu_fp8", request)
def test_pytest_leaky_relu_i16(request): run_leaky_relu_sim("I16", "cocotb_leaky_relu_i16", request)
def test_pytest_leaky_relu_bf16(request): run_leaky_relu_sim("BF16", "cocotb_leaky_relu_bf16", request)
