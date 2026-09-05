# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np
import os

from golden_models.dtypes import I8, FP8_E4M3, I32, I4, BF16, FP4_E2M1
from golden_models.ops import linear_hw, linear_hw_wxay
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, run_layer_sim, DEFAULT_NUM_TRIALS
from utils.tb_utils import seed_random, SEED

seed_random()

# ==========================================
# LINEAR LAYER
# ==========================================
async def run_linear_test(dut, op_name, dtype_name, dtype, A, W, b, is_floatml, A_shape=(1, 2), A_lanes=2, W_shape=(2, 1), W_lanes=2, b_shape=(1, 1), b_lanes=1, Y_shape=(1, 1), Y_lanes=1, collect=None):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_a_stream_valid.value = 0
    dut.io_w_stream_valid.value = 0
    dut.io_b_stream_valid.value = 0
    dut.io_y_stream_ready.value = 0
    
    # Send W (transposed because Linear/Matmul expects column-major weight streaming)
    W_T = np.array(W).T.tolist()
    await send_tensor(dut, "io_w_stream", W_T, (W_shape[1], W_shape[0]), W_lanes, dtype, is_floatml)
    
    acc_dtype = dtype if is_floatml else I32
    b_shape = (1, len(b[0]))
    await send_tensor(dut, "io_b_stream", b, b_shape, b_lanes, acc_dtype, is_floatml)
    
    send_a = cocotb.start_soon(send_tensor(dut, "io_a_stream", A, A_shape, A_lanes, dtype, is_floatml))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_y_stream", Y_shape, acc_dtype, is_floatml, Y_lanes))
    
    Y_out_bits, Y_out = await recv_y
    await send_a
    
    # True Math
    A_np = np.array(A)
    W_np = np.array(W)
    b_np = np.array(b)
    Y_true = np.matmul(A_np, W_np) + b_np
    
    if collect is not None:
        collect["out"].append(Y_out)
        collect["true"].append(Y_true.tolist())
    else:
        log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, Y_out, Y_true.tolist())
        dut._log.info(log_msg)
    
    # Exact HW Math
    Y_expected = linear_hw(A, W, b, dtype)
    
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    for m in range(Y_shape[0]):
        for n in range(Y_shape[1]):
            exp_val = Y_expected[m][n]
            exp_bits = acc_dtype.from_float(exp_val)
            out_bits = Y_out_bits[m][n]
            out_val = Y_out[m][n]
            
            if bit_width > 8 and is_floatml:
                assert abs(out_bits - exp_bits) <= 1, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {acc_dtype.to_float(exp_bits)}"
            else:
                assert out_bits == exp_bits, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {acc_dtype.to_float(exp_bits)}"

@cocotb.test()
async def cocotb_linear_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        A = get_random_tensor((1, 2), 10.0, True)
        W = get_random_tensor((2, 1), 10.0, True)
        b = get_random_tensor((1, 1), 10.0, True)
        await run_linear_test(dut, "Linear", "I8", I8, A, W, b, False, collect=collect)
    details = f"A=1x2, W=2x1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Linear", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_linear_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        A = get_random_tensor((1, 2), 5.0, False)
        W = get_random_tensor((2, 1), 5.0, False)
        b = get_random_tensor((1, 1), 5.0, False)
        await run_linear_test(dut, "Linear", "FP8", FP8_E4M3, A, W, b, True, collect=collect)
    details = f"A=1x2, W=2x1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Linear", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_linearmulti_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        A = get_random_tensor((2, 3), 10.0, True)
        W = get_random_tensor((3, 4), 10.0, True)
        b = get_random_tensor((1, 4), 10.0, True)
        await run_linear_test(dut, "LinearMulti", "I8", I8, A, W, b, False, (2,3), 3, (3,4), 3, (1,4), 1, (2,4), 1, collect=collect)
    details = f"A=2x3, W=3x4, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LinearMulti", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_linearmulti_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        A = get_random_tensor((2, 3), 5.0, False)
        W = get_random_tensor((3, 4), 5.0, False)
        b = get_random_tensor((1, 4), 5.0, False)
        await run_linear_test(dut, "LinearMulti", "FP8", FP8_E4M3, A, W, b, True, (2,3), 3, (3,4), 3, (1,4), 1, (2,4), 1, collect=collect)
    details = f"A=2x3, W=3x4, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LinearMulti", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_linear_i8(request): run_layer_sim("Linear", "I8", "cocotb_linear_i8", "LinearTestComp", request)
def test_pytest_linear_fp8(request): run_layer_sim("Linear", "FP8", "cocotb_linear_fp8", "LinearTestComp", request)
def test_pytest_linearmulti_i8(request): run_layer_sim("Linear", "I8", "cocotb_linearmulti_i8", "LinearTestCompMulti", request)
def test_pytest_linearmulti_fp8(request): run_layer_sim("Linear", "FP8", "cocotb_linearmulti_fp8", "LinearTestCompMulti", request)

# ==========================================
# LINEAR LAYER — WEIGHT-ONLY QUANTIZATION (wXaY)
# SInt weights (I4/I8) + compile-time scale(s), float activations
# ==========================================
QUANT_COMBOS = {
    "w8a16": (I8, 8, BF16),
    "w4a16": (I4, 4, BF16),
    "w8a8": (I8, 8, FP8_E4M3),
    "w4a8": (I4, 4, FP8_E4M3),
    "w8a4": (I8, 8, FP4_E2M1),
    "w4a4": (I4, 4, FP4_E2M1),
}

async def run_linear_quant_test(dut, op_name, combo_name, w_dtype, w_bits, act_dtype, A, W, b,
                                A_shape=(1, 2), A_lanes=2, W_shape=(2, 1), W_lanes=2,
                                b_shape=(1, 1), b_lanes=1, Y_shape=(1, 1), Y_lanes=1,
                                scales=(1.0,), collect=None):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)

    dut.io_a_stream_valid.value = 0
    dut.io_w_stream_valid.value = 0
    dut.io_b_stream_valid.value = 0
    dut.io_y_stream_ready.value = 0

    # Send W (transposed because Linear/Matmul expects column-major weight streaming) as raw SInt
    W_T = np.array(W).T.tolist()
    await send_tensor(dut, "io_w_stream", W_T, (W_shape[1], W_shape[0]), W_lanes, w_dtype, False)

    await send_tensor(dut, "io_b_stream", b, b_shape, b_lanes, act_dtype, True)

    send_a = cocotb.start_soon(send_tensor(dut, "io_a_stream", A, A_shape, A_lanes, act_dtype, True))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_y_stream", Y_shape, act_dtype, True, Y_lanes))

    Y_out_bits, Y_out = await recv_y
    await send_a

    # True Math (ideal dequantized weights: per-channel or per-tensor broadcast scale)
    A_np = np.array(A)
    W_np = np.array(W) * np.array(scales)
    b_np = np.array(b)
    Y_true = np.matmul(A_np, W_np) + b_np

    if collect is not None:
        collect["out"].append(Y_out)
        collect["true"].append(Y_true.tolist())
    else:
        log_msg = log_true_math_error(op_name, combo_name, act_dtype, True, Y_out, Y_true.tolist())
        dut._log.info(log_msg)

    # Exact HW Math
    Y_expected = linear_hw_wxay(A, W, b, act_dtype, scales, weight_bits=w_bits)

    bit_width = getattr(act_dtype, 'exp_bits', 0) + getattr(act_dtype, 'mant_bits', 0)
    for m in range(Y_shape[0]):
        for n in range(Y_shape[1]):
            exp_val = Y_expected[m][n]
            exp_bits = act_dtype.from_float(exp_val)
            out_bits = Y_out_bits[m][n]
            out_val = Y_out[m][n]

            if bit_width > 8:
                assert abs(out_bits - exp_bits) <= 1, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {act_dtype.to_float(exp_bits)}"
            else:
                assert out_bits == exp_bits, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {act_dtype.to_float(exp_bits)}"

async def _run_linear_quant_combo(dut, combo_name, scales=(1.0,)):
    w_dt, w_bits, a_dt = QUANT_COMBOS[combo_name]
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        A = get_random_tensor((2, 3), 5.0, False)
        W = get_random_tensor((3, 4), 5.0, True)
        b = get_random_tensor((1, 4), 5.0, False)
        await run_linear_quant_test(dut, "LinearQuant", combo_name, w_dt, w_bits, a_dt, A, W, b,
                                    (2, 3), 3, (3, 4), 3, (1, 4), 1, (2, 4), 1,
                                    scales=scales, collect=collect)
    details = f"A=2x3, W=3x4, scales={scales}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LinearQuant", combo_name, a_dt, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_linear_quant_w8a16(dut): await _run_linear_quant_combo(dut, "w8a16")

@cocotb.test()
async def cocotb_linear_quant_w4a16(dut): await _run_linear_quant_combo(dut, "w4a16")

@cocotb.test()
async def cocotb_linear_quant_w8a8(dut): await _run_linear_quant_combo(dut, "w8a8")

@cocotb.test()
async def cocotb_linear_quant_w4a8(dut): await _run_linear_quant_combo(dut, "w4a8")

@cocotb.test()
async def cocotb_linear_quant_w8a4(dut): await _run_linear_quant_combo(dut, "w8a4")

@cocotb.test()
async def cocotb_linear_quant_w4a4(dut): await _run_linear_quant_combo(dut, "w4a4")

@cocotb.test()
async def cocotb_linear_quant_perchannel(dut):
    # Per-channel scales: one weight scale per output feature (N=4 columns)
    await _run_linear_quant_combo(dut, "w8a16", scales=(0.5, -0.25, 1.5, 2.0))


def test_pytest_linear_quant_w8a16(request): run_layer_sim("Linear", "w8a16", "cocotb_linear_quant_w8a16", "LinearQuantTestCompMulti", request)
def test_pytest_linear_quant_w4a16(request): run_layer_sim("Linear", "w4a16", "cocotb_linear_quant_w4a16", "LinearQuantTestCompMulti", request)
def test_pytest_linear_quant_w8a8(request): run_layer_sim("Linear", "w8a8", "cocotb_linear_quant_w8a8", "LinearQuantTestCompMulti", request)
def test_pytest_linear_quant_w4a8(request): run_layer_sim("Linear", "w4a8", "cocotb_linear_quant_w4a8", "LinearQuantTestCompMulti", request)
def test_pytest_linear_quant_w8a4(request): run_layer_sim("Linear", "w8a4", "cocotb_linear_quant_w8a4", "LinearQuantTestCompMulti", request)
def test_pytest_linear_quant_w4a4(request): run_layer_sim("Linear", "w4a4", "cocotb_linear_quant_w4a4", "LinearQuantTestCompMulti", request)
def test_pytest_linear_quant_perchannel(request): run_layer_sim("Linear", "PerChannel", "cocotb_linear_quant_perchannel", "LinearQuantTestCompMulti", request)