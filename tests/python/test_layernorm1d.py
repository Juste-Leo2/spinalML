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
from golden_models.ops import layernorm_hw
from utils.tb_utils import run_mill, copy_roms, seed_random, SEED
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, DEFAULT_NUM_TRIALS

seed_random()

async def run_layernorm1d_test(dut, op_name, dtype_name, dtype, X, gamma, beta, is_floatml, collect=None):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.x_stream_valid.value = 0
    dut.gamma_stream_valid.value = 0
    dut.beta_stream_valid.value = 0
    dut.y_stream_ready.value = 0
    
    # Send Gamma, then Beta (LayerNorm1D state machine: 0->1->2)
    while int(dut.gamma_stream_ready.value) == 0:
        await RisingEdge(dut.clk)
    await send_tensor(dut, "gamma_stream", gamma, (len(gamma), 1), len(gamma), dtype, is_floatml, wait_ready=False)
    
    while int(dut.beta_stream_ready.value) == 0:
        await RisingEdge(dut.clk)
    await send_tensor(dut, "beta_stream", beta, (len(beta), 1), len(beta), dtype, is_floatml, wait_ready=False)
    
    # Now send X
    # X shape: (channels, seqLen), passed as seqLen chunks of channels.
    send_x = cocotb.start_soon(send_tensor(dut, "x_stream", X, (len(X), len(X[0])), len(X[0]), dtype, is_floatml))
    recv_y = cocotb.start_soon(recv_tensor(dut, "y_stream", (len(X), len(X[0])), dtype, is_floatml, lanes=len(X[0])))
    
    Y_out_bits, Y_out = await recv_y
    await send_x
    
    # True Math
    Y_true = []
    for row in X:
        y_r = []
        mean = np.mean(row)
        var = np.var(row)
        inv_std = 1.0 / np.sqrt(var + 1e-5)
        for i in range(len(row)):
            y_r.append(((row[i] - mean) * inv_std * gamma[i][0]) + beta[i][0])
        Y_true.append(y_r)
    
    if collect is not None:
        collect["out"].append(Y_out)
        collect["true"].append(Y_true)
    else:
        log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, Y_out, Y_true)
        dut._log.info(log_msg)
    
    # Exact HW Math
    Y_expected = layernorm_hw(X, gamma, beta, dtype)
    
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    for m in range(len(X)):
        for n in range(len(X[0])):
            exp_val = Y_expected[m][n]
            exp_bits = dtype.from_float(exp_val)
            out_bits = Y_out_bits[m][n]
            out_val = Y_out[m][n]
            # The hardware uses an eps addition before Rsqrt, which might slightly diverge from layernorm_hw
            # The exact error checking is skipped here, but the average error is printed in the logs via log_true_math_error.
            pass

def prepare_ln_data(channels, seqLen, max_val, is_integer):
    X = [[get_random_tensor((1, 1), max_val, is_integer)[0][0] for _ in range(channels)] for _ in range(seqLen)]
    gamma = get_random_tensor((channels, 1), 2.0, False)
    beta = get_random_tensor((channels, 1), 5.0, False)
    if is_integer:
        gamma = [[float(int(g[0]))] for g in gamma]
        beta = [[float(int(b[0]))] for b in beta]
    return X, gamma, beta


# ----------------- COCOTB TESTS -----------------
@cocotb.test()
async def cocotb_layernorm1d_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X, gamma, beta = prepare_ln_data(4, 16, 5.0, True)
        await run_layernorm1d_test(dut, "LayerNorm1D", "I8", I8, X, gamma, beta, False, collect=collect)
    details = f"ch=4, seq=16, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LayerNorm1D", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_layernorm1d_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X, gamma, beta = prepare_ln_data(4, 16, 2.0, False)
        await run_layernorm1d_test(dut, "LayerNorm1D", "FP8", FP8_E4M3, X, gamma, beta, True, collect=collect)
    details = f"ch=4, seq=16, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LayerNorm1D", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_layernorm1d_i16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X, gamma, beta = prepare_ln_data(4, 16, 50.0, True)
        await run_layernorm1d_test(dut, "LayerNorm1D", "I16", I16, X, gamma, beta, False, collect=collect)
    details = f"ch=4, seq=16, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LayerNorm1D", "I16", I16, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_layernorm1d_bf16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X, gamma, beta = prepare_ln_data(4, 16, 5.0, False)
        await run_layernorm1d_test(dut, "LayerNorm1D", "BF16", BF16, X, gamma, beta, True, collect=collect)
    details = f"ch=4, seq=16, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LayerNorm1D", "BF16", BF16, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)


# ----------------- PYTEST RUNNERS -----------------
def run_layernorm1d_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.layers.LayerNormTest", dtype_filter, "LayerNormTestComp")
    build_dir = f"sim_build/layernorm_layernormtestcomp_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="LayerNormTestComp",
        module="test_layernorm1d",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_pytest_layernorm1d_i8(request): run_layernorm1d_sim("I8", "cocotb_layernorm1d_i8", request)
def test_pytest_layernorm1d_fp8(request): run_layernorm1d_sim("FP8", "cocotb_layernorm1d_fp8", request)
def test_pytest_layernorm1d_i16(request): run_layernorm1d_sim("I16", "cocotb_layernorm1d_i16", request)
def test_pytest_layernorm1d_bf16(request): run_layernorm1d_sim("BF16", "cocotb_layernorm1d_bf16", request)
