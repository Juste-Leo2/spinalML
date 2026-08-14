import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest
import os
import random
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from utils.tb_utils import run_mill, copy_roms

async def run_transpose_test(dut, op_name, dtype_name, dtype, X_test, M, N, is_floatml):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 0
    
    # 1. Send Matrix A (size M x N) and await C (size N x M)
    async def send_a():
        for m in range(M):
            for n in range(N):
                val = X_test[m][n]
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

    async def recv_c():
        Y_out = [[0.0] * M for _ in range(N)]
        Y_out_bits = [[0] * M for _ in range(N)]
        dut.io_c_stream_ready.value = 1
        for n in range(N):
            for m in range(M):
                while dut.io_c_stream_valid.value == 0:
                    await RisingEdge(dut.clk)
                
                if is_floatml:
                    out_sign = int(dut.io_c_stream_payload_0_sign.value)
                    out_exp = int(dut.io_c_stream_payload_0_exponent.value)
                    out_mant = int(dut.io_c_stream_payload_0_mantissa.value)
                    out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
                else:
                    out_bits = int(dut.io_c_stream_payload_0.value)
                    
                Y_out_bits[n][m] = out_bits
                Y_out[n][m] = dtype.to_float(out_bits)
                await RisingEdge(dut.clk)
        return Y_out_bits, Y_out

    send_task = cocotb.start_soon(send_a())
    recv_task = cocotb.start_soon(recv_c())
    
    Y_out_bits, Y_out = await recv_task
    await send_task
    
    # Golden model
    X_np = np.array(X_test)
    Y_expected = np.transpose(X_np)
    
    # Check bit exactness
    for n in range(N):
        for m in range(M):
            exp_val = float(Y_expected[n][m])
            exp_bits = dtype.from_float(exp_val)
            out_bits = Y_out_bits[n][m]
            out_val = Y_out[n][m]
            
            assert out_bits == exp_bits, f"HW Mismatch for {op_name} at Y[{n}][{m}]: got {out_val} (bits {out_bits}) instead of {dtype.to_float(exp_bits)} (bits {exp_bits})"
            
    dut._log.info(f"[{op_name}][{dtype_name}] Test {M}x{N} | Passed Bit-Exact Check")


def get_random_matrix(M, N, range_val=5.0, integer=True):
    X = [[0.0]*N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            val = random.uniform(-range_val, range_val)
            X[m][n] = round(val) if integer else val
    return X


@cocotb.test()
async def cocotb_transpose_2x3_i8(dut):
    X = get_random_matrix(2, 3, range_val=100.0, integer=True)
    await run_transpose_test(dut, "Transpose", "I8", I8, X, 2, 3, is_floatml=False)

@cocotb.test()
async def cocotb_transpose_2x3_fp8(dut):
    X = get_random_matrix(2, 3, range_val=5.0, integer=False)
    await run_transpose_test(dut, "Transpose", "FP8", FP8_E4M3, X, 2, 3, is_floatml=True)

@cocotb.test()
async def cocotb_transpose_2x3_i16(dut):
    X = get_random_matrix(2, 3, range_val=1000.0, integer=True)
    await run_transpose_test(dut, "Transpose", "I16", I16, X, 2, 3, is_floatml=False)

@cocotb.test()
async def cocotb_transpose_2x3_bf16(dut):
    X = get_random_matrix(2, 3, range_val=10.0, integer=False)
    await run_transpose_test(dut, "Transpose", "BF16", BF16, X, 2, 3, is_floatml=True)

@cocotb.test()
async def cocotb_transpose_4x4_i8(dut):
    X = get_random_matrix(4, 4, range_val=50.0, integer=True)
    await run_transpose_test(dut, "Transpose", "I8", I8, X, 4, 4, is_floatml=False)


def run_transpose_sim(dtype_filter, testcase_name, toplevel, request=None):
    v_file = run_mill("spinalML.ops.TransposeTest", dtype_filter, toplevel)
    build_dir = f"sim_build/transpose_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_transpose",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_transpose_2x3_i8(request): run_transpose_sim("I8", "cocotb_transpose_2x3_i8", "TransposeTestComp_2x3", request)
def test_transpose_2x3_fp8(request): run_transpose_sim("FP8", "cocotb_transpose_2x3_fp8", "TransposeTestComp_2x3", request)
def test_transpose_2x3_i16(request): run_transpose_sim("I16", "cocotb_transpose_2x3_i16", "TransposeTestComp_2x3", request)
def test_transpose_2x3_bf16(request): run_transpose_sim("BF16", "cocotb_transpose_2x3_bf16", "TransposeTestComp_2x3", request)

def test_transpose_4x4_i8(request): run_transpose_sim("I8", "cocotb_transpose_4x4_i8", "TransposeTestComp_4x4", request)
