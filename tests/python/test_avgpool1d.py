# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import numpy as np
import os

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, DEFAULT_NUM_TRIALS
from utils.tb_utils import run_mill, copy_roms, seed_random, SEED

seed_random()

# AvgPool1D golden model
def avgpool1d_hw(X, poolSize, stride, dtype):
    L_in = len(X)
    channels = len(X[0])
    L_out = (L_in - poolSize) // stride + 1
    Y = []
    
    import math
    shift = int(math.log2(poolSize))
    
    for i in range(L_out):
        start = i * stride
        window = X[start:start+poolSize]
        
        y_row = []
        for ch in range(channels):
            if getattr(dtype, 'is_floatml', False):
                val = sum([w[ch] for w in window]) / poolSize
                bits = dtype.from_float(val)
            else:
                # HW integer behavior
                acc = sum([int(w[ch]) for w in window])
                # Shift
                val = acc >> shift
                bits = dtype.from_float(val)
                
            y_row.append(dtype.to_float(bits))
        Y.append(y_row)
    return Y

async def run_avgpool1d_test(dut, op_name, dtype_name, dtype, X, poolSize, stride, is_floatml, collect=None):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 0
    
    L_in = len(X)
    channels = len(X[0])
    L_out = (L_in - poolSize) // stride + 1
    
    send_x = cocotb.start_soon(send_tensor(dut, "io_a_stream", X, (L_in, channels), channels, dtype, is_floatml))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_c_stream", (L_out, channels), dtype, is_floatml, lanes=channels))
    
    Y_out_bits, Y_out = await recv_y
    await send_x
    
    # True Math
    X_np = np.array(X)
    Y_true = []
    for i in range(L_out):
        start = i * stride
        window = X_np[start:start+poolSize, :]
        Y_true.append(np.mean(window, axis=0).tolist())
        
    if collect is not None:
        collect["out"].append(Y_out)
        collect["true"].append(Y_true)
    else:
        log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, Y_out, Y_true)
        dut._log.info(log_msg)
    
    # Exact HW Math
    Y_expected = avgpool1d_hw(X, poolSize, stride, dtype)
    
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    for m in range(L_out):
        for n in range(channels):
            exp_val = Y_expected[m][n]
            exp_bits = dtype.from_float(exp_val)
            out_bits = Y_out_bits[m][n]
            out_val = Y_out[m][n]
            if is_floatml:
                assert abs(out_val - exp_val) <= max(0.05, abs(exp_val) * 0.25), f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {exp_val}"
            else:
                assert out_bits == exp_bits, f"HW Mismatch at Y[{m}][{n}]: got {out_val} (bits {out_bits}) instead of {dtype.to_float(exp_bits)} (bits {exp_bits})"

@cocotb.test()
async def cocotb_avgpool1d_i8(dut):
    collect = {"out": [], "true": []}
    setattr(I8, 'is_floatml', False)
    setattr(I8, 'signed', True)
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 2), 50.0, True)
        await run_avgpool1d_test(dut, "AvgPool1D", "I8", I8, X, 2, 2, False, collect=collect)
    details = f"L=4x2, pool=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("AvgPool1D", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_avgpool1d_fp8(dut):
    collect = {"out": [], "true": []}
    setattr(FP8_E4M3, 'is_floatml', True)
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 2), 5.0, False)
        await run_avgpool1d_test(dut, "AvgPool1D", "FP8", FP8_E4M3, X, 2, 2, True, collect=collect)
    details = f"L=4x2, pool=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("AvgPool1D", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_avgpool1d_i16(dut):
    collect = {"out": [], "true": []}
    setattr(I16, 'is_floatml', False)
    setattr(I16, 'signed', True)
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 2), 500.0, True)
        await run_avgpool1d_test(dut, "AvgPool1D", "I16", I16, X, 2, 2, False, collect=collect)
    details = f"L=4x2, pool=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("AvgPool1D", "I16", I16, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_avgpool1d_bf16(dut):
    collect = {"out": [], "true": []}
    setattr(BF16, 'is_floatml', True)
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 2), 5.0, False)
        await run_avgpool1d_test(dut, "AvgPool1D", "BF16", BF16, X, 2, 2, True, collect=collect)
    details = f"L=4x2, pool=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("AvgPool1D", "BF16", BF16, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def run_pool_sim(layer_name, dtype_filter, testcase_name, toplevel, request=None):
    v_file = run_mill(f"spinalML.poolings.{layer_name}Test", dtype_filter, toplevel)
    build_dir = f"sim_build/{layer_name.lower()}_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module=f"test_{layer_name.lower()}",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_pytest_avgpool1d_i8(request): run_pool_sim("AvgPool1D", "I8", "cocotb_avgpool1d_i8", "AvgPool1DTestComp", request)
def test_pytest_avgpool1d_fp8(request): run_pool_sim("AvgPool1D", "FP8", "cocotb_avgpool1d_fp8", "AvgPool1DTestComp", request)
def test_pytest_avgpool1d_i16(request): run_pool_sim("AvgPool1D", "I16", "cocotb_avgpool1d_i16", "AvgPool1DTestComp", request)
def test_pytest_avgpool1d_bf16(request): run_pool_sim("AvgPool1D", "BF16", "cocotb_avgpool1d_bf16", "AvgPool1DTestComp", request)
