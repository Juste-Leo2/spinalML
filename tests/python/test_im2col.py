# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest
import os
import random
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from utils.tb_utils import run_mill, copy_roms

async def run_im2col_test(dut, op_name, dtype_name, dtype, X_test, H, W, K, is_floatml):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 0
    
    H_out = H - K + 1
    W_out = W - K + 1
    totalWindows = H_out * W_out
    lanes = K * K
    
    # 1. Send Matrix A (size H x W, lanes=1)
    async def send_a():
        for h in range(H):
            for w in range(W):
                val = X_test[h][w]
                val_bits = dtype.from_float(val)
                if is_floatml:
                    sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                    exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                    mant = val_bits & ((1 << dtype.mant_bits) - 1)
                    dut.io_a_stream_payload_0_sign.value = sign
                    dut.io_a_stream_payload_0_exponent.value = exp_val
                    dut.io_a_stream_payload_0_mantissa.value = mant
                else:
                    dut.io_a_stream_payload_0.value = val_bits
                    
                dut.io_a_stream_valid.value = 1
                await RisingEdge(dut.clk)
                while dut.io_a_stream_ready.value == 0:
                    await RisingEdge(dut.clk)
        dut.io_a_stream_valid.value = 0

    # 2. Receive Matrix C (size totalWindows x lanes)
    async def recv_c():
        Y_out = [[0.0] * lanes for _ in range(totalWindows)]
        Y_out_bits = [[0] * lanes for _ in range(totalWindows)]
        dut.io_c_stream_ready.value = 1
        for w_idx in range(totalWindows):
            while dut.io_c_stream_valid.value == 0:
                await RisingEdge(dut.clk)
            
            for l in range(lanes):
                if is_floatml:
                    out_sign = int(getattr(dut, f"io_c_stream_payload_{l}_sign").value)
                    out_exp = int(getattr(dut, f"io_c_stream_payload_{l}_exponent").value)
                    out_mant = int(getattr(dut, f"io_c_stream_payload_{l}_mantissa").value)
                    out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
                else:
                    out_bits = int(getattr(dut, f"io_c_stream_payload_{l}").value)
                    
                Y_out_bits[w_idx][l] = out_bits
                Y_out[w_idx][l] = dtype.to_float(out_bits)
            await RisingEdge(dut.clk)
        return Y_out_bits, Y_out

    send_task = cocotb.start_soon(send_a())
    recv_task = cocotb.start_soon(recv_c())
    
    Y_out_bits, Y_out = await recv_task
    await send_task
    
    # Golden model
    X_np = np.array(X_test)
    Y_expected = sliding_window_view(X_np, (K, K)).reshape(totalWindows, lanes)
    
    # Check bit exactness
    for w_idx in range(totalWindows):
        for l in range(lanes):
            exp_val = float(Y_expected[w_idx][l])
            exp_bits = dtype.from_float(exp_val)
            out_bits = Y_out_bits[w_idx][l]
            out_val = Y_out[w_idx][l]
            
            assert out_bits == exp_bits, f"HW Mismatch for {op_name} at Window {w_idx}, lane {l}: got {out_val} (bits {out_bits}) instead of {dtype.to_float(exp_bits)} (bits {exp_bits})"
            
    dut._log.info(f"[{op_name}][{dtype_name}] Test {H}x{W} K={K} | Passed Bit-Exact Check")


def get_random_image(H, W, range_val=5.0, integer=True):
    X = [[0.0]*W for _ in range(H)]
    for h in range(H):
        for w in range(W):
            val = random.uniform(-range_val, range_val)
            X[h][w] = round(val) if integer else val
    return X


@cocotb.test()
async def cocotb_im2col_3x3_k2_i8(dut):
    X = get_random_image(3, 3, range_val=100.0, integer=True)
    await run_im2col_test(dut, "Im2Col", "I8", I8, X, 3, 3, 2, is_floatml=False)

@cocotb.test()
async def cocotb_im2col_3x3_k2_fp8(dut):
    X = get_random_image(3, 3, range_val=5.0, integer=False)
    await run_im2col_test(dut, "Im2Col", "FP8", FP8_E4M3, X, 3, 3, 2, is_floatml=True)

@cocotb.test()
async def cocotb_im2col_3x3_k2_i16(dut):
    X = get_random_image(3, 3, range_val=1000.0, integer=True)
    await run_im2col_test(dut, "Im2Col", "I16", I16, X, 3, 3, 2, is_floatml=False)

@cocotb.test()
async def cocotb_im2col_3x3_k2_bf16(dut):
    X = get_random_image(3, 3, range_val=10.0, integer=False)
    await run_im2col_test(dut, "Im2Col", "BF16", BF16, X, 3, 3, 2, is_floatml=True)

@cocotb.test()
async def cocotb_im2col_4x4_k3_i8(dut):
    X = get_random_image(4, 4, range_val=50.0, integer=True)
    await run_im2col_test(dut, "Im2Col", "I8", I8, X, 4, 4, 3, is_floatml=False)


def run_im2col_sim(dtype_filter, testcase_name, toplevel, request=None):
    v_file = run_mill("spinalML.ops.Im2ColTest", dtype_filter, toplevel)
    build_dir = f"sim_build/im2col_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_im2col",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_im2col_3x3_k2_i8(request): run_im2col_sim("I8", "cocotb_im2col_3x3_k2_i8", "Im2ColTestComp_3x3_K2", request)
def test_im2col_3x3_k2_fp8(request): run_im2col_sim("FP8", "cocotb_im2col_3x3_k2_fp8", "Im2ColTestComp_3x3_K2", request)
def test_im2col_3x3_k2_i16(request): run_im2col_sim("I16", "cocotb_im2col_3x3_k2_i16", "Im2ColTestComp_3x3_K2", request)
def test_im2col_3x3_k2_bf16(request): run_im2col_sim("BF16", "cocotb_im2col_3x3_k2_bf16", "Im2ColTestComp_3x3_K2", request)

def test_im2col_4x4_k3_i8(request): run_im2col_sim("I8", "cocotb_im2col_4x4_k3_i8", "Im2ColTestComp_4x4_K3", request)
