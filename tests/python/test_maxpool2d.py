# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import numpy as np
import os

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import maxpool2d_hw
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, DEFAULT_NUM_TRIALS
from utils.tb_utils import run_mill, copy_roms, seed_random, SEED

seed_random()

async def run_maxpool2d_test(dut, op_name, dtype_name, dtype, X, poolSize, stride, is_floatml, collect=None):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)

    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 0

    H = len(X)
    W_in = len(X[0])
    is_3d = isinstance(X[0][0], list)
    C = len(X[0][0]) if is_3d else 1
    H_out = (H - poolSize) // stride + 1
    W_out = (W_in - poolSize) // stride + 1
    X_shape = (H, W_in, C) if is_3d else (H, W_in)
    Y_shape = (H_out, W_out, C) if is_3d else (H_out, W_out)

    send_x = cocotb.start_soon(send_tensor(dut, "io_a_stream", X, X_shape, 1, dtype, is_floatml))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_c_stream", Y_shape, dtype, is_floatml, lanes=C))

    Y_out_bits, Y_out = await recv_y
    await send_x

    # True Math (for logging true error)
    X_np = np.array(X)
    Y_true = []
    for i in range(H_out):
        row = []
        for j in range(W_out):
            window = X_np[i * stride:i * stride + poolSize, j * stride:j * stride + poolSize]
            if is_3d:
                row.append(np.max(window.reshape(poolSize * poolSize, -1), axis=0).tolist())
            else:
                row.append(float(np.max(window)))
        Y_true.append(row)

    if collect is not None:
        collect["out"].append(np.array(Y_out).reshape(Y_shape).tolist())
        collect["true"].append(Y_true)
    else:
        log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, np.array(Y_out).reshape(Y_shape).tolist(), Y_true)
        dut._log.info(log_msg)

    # Exact HW Math
    Y_expected = maxpool2d_hw(X, poolSize, stride, dtype)

    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    exp_flat = np.array(Y_expected).flatten().tolist()
    bits_flat = np.array(Y_out_bits).flatten().tolist()
    vals_flat = np.array(Y_out).flatten().tolist()
    for idx in range(len(exp_flat)):
        exp_bits = dtype.from_float(exp_flat[idx])
        out_bits = bits_flat[idx]
        out_val = vals_flat[idx]
        if bit_width > 8 and is_floatml:
            assert abs(out_bits - exp_bits) <= 1, f"HW Mismatch at [{idx}]: got {out_val} instead of {dtype.to_float(exp_bits)}"
        else:
            assert out_bits == exp_bits, f"HW Mismatch at [{idx}]: got {out_val} instead of {dtype.to_float(exp_bits)}"

@cocotb.test()
async def cocotb_maxpool2d_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 4), 10.0, True)
        await run_maxpool2d_test(dut, "MaxPool2D", "I8", I8, X, 2, 2, False, collect=collect)
    details = f"X=4x4, K=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MaxPool2D", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_maxpool2d_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 4), 5.0, False)
        await run_maxpool2d_test(dut, "MaxPool2D", "FP8", FP8_E4M3, X, 2, 2, True, collect=collect)
    details = f"X=4x4, K=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MaxPool2D", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_maxpool2d_i16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 4), 100.0, True)
        await run_maxpool2d_test(dut, "MaxPool2D", "I16", I16, X, 2, 2, False, collect=collect)
    details = f"X=4x4, K=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MaxPool2D", "I16", I16, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_maxpool2d_bf16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 4), 5.0, False)
        await run_maxpool2d_test(dut, "MaxPool2D", "BF16", BF16, X, 2, 2, True, collect=collect)
    details = f"X=4x4, K=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MaxPool2D", "BF16", BF16, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

# Multi-channel tests
@cocotb.test()
async def cocotb_maxpool2dmulti_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 4, 2), 10.0, True)
        await run_maxpool2d_test(dut, "MaxPool2DMulti", "I8", I8, X, 2, 2, False, collect=collect)
    details = f"X=4x4x2, K=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MaxPool2DMulti", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_maxpool2dmulti_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((4, 4, 2), 5.0, False)
        await run_maxpool2d_test(dut, "MaxPool2DMulti", "FP8", FP8_E4M3, X, 2, 2, True, collect=collect)
    details = f"X=4x4x2, K=2, stride=2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MaxPool2DMulti", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
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

def test_pytest_maxpool2d_i8(request): run_pool_sim("MaxPool2D", "I8", "cocotb_maxpool2d_i8", "MaxPool2DTestComp", request)
def test_pytest_maxpool2d_fp8(request): run_pool_sim("MaxPool2D", "FP8", "cocotb_maxpool2d_fp8", "MaxPool2DTestComp", request)
def test_pytest_maxpool2d_i16(request): run_pool_sim("MaxPool2D", "I16", "cocotb_maxpool2d_i16", "MaxPool2DTestComp", request)
def test_pytest_maxpool2d_bf16(request): run_pool_sim("MaxPool2D", "BF16", "cocotb_maxpool2d_bf16", "MaxPool2DTestComp", request)

def test_pytest_maxpool2dmulti_i8(request): run_pool_sim("MaxPool2D", "I8", "cocotb_maxpool2dmulti_i8", "MaxPool2DTestCompMulti", request)
def test_pytest_maxpool2dmulti_fp8(request): run_pool_sim("MaxPool2D", "FP8", "cocotb_maxpool2dmulti_fp8", "MaxPool2DTestCompMulti", request)
