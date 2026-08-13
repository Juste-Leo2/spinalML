import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest
import os
import random

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import matmul_hw
from utils.tb_utils import run_mill, copy_roms

async def run_matmul_test(dut, op_name, dtype_name, dtype, A_test, B_test, M, K, N, lanes, is_floatml):
    """
    Generic Cocotb test method for MatmulOp.
    A_test: list of lists (M x K)
    B_test: list of lists (K x N)
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    chunks_K = (K + lanes - 1) // lanes
    
    dut.io_a_stream_valid.value = 0
    dut.io_b_stream_valid.value = 0
    dut.io_c_stream_ready.value = 0
    
    # 1. Send Matrix B
    for n in range(N):
        for chunk in range(chunks_K):
            if is_floatml:
                for l in range(lanes):
                    k_idx = chunk * lanes + l
                    b_val = B_test[k_idx][n] if k_idx < K else 0.0
                    val_bits = dtype.from_float(b_val)
                    sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                    exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                    mant = val_bits & ((1 << dtype.mant_bits) - 1)
                    getattr(dut, f"io_b_stream_payload_{l}_sign").value = sign
                    getattr(dut, f"io_b_stream_payload_{l}_exponent").value = exp_val
                    getattr(dut, f"io_b_stream_payload_{l}_mantissa").value = mant
            else:
                for l in range(lanes):
                    k_idx = chunk * lanes + l
                    b_val = B_test[k_idx][n] if k_idx < K else 0.0
                    val_bits = dtype.from_float(b_val)
                    getattr(dut, f"io_b_stream_payload_{l}").value = val_bits
                    
            dut.io_b_stream_valid.value = 1
            await RisingEdge(dut.clk)
            while dut.io_b_stream_ready.value == 0:
                await RisingEdge(dut.clk)
    dut.io_b_stream_valid.value = 0
    
    # 2. Send Matrix A and wait for Matrix C concurrently
    async def send_a():
        for m in range(M):
            for chunk in range(chunks_K):
                if is_floatml:
                    for l in range(lanes):
                        k_idx = chunk * lanes + l
                        a_val = A_test[m][k_idx] if k_idx < K else 0.0
                        val_bits = dtype.from_float(a_val)
                        sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                        exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                        mant = val_bits & ((1 << dtype.mant_bits) - 1)
                        getattr(dut, f"io_a_stream_payload_{l}_sign").value = sign
                        getattr(dut, f"io_a_stream_payload_{l}_exponent").value = exp_val
                        getattr(dut, f"io_a_stream_payload_{l}_mantissa").value = mant
                else:
                    for l in range(lanes):
                        k_idx = chunk * lanes + l
                        a_val = A_test[m][k_idx] if k_idx < K else 0.0
                        val_bits = dtype.from_float(a_val)
                        getattr(dut, f"io_a_stream_payload_{l}").value = val_bits
                        
                dut.io_a_stream_valid.value = 1
                await RisingEdge(dut.clk)
                while dut.io_a_stream_ready.value == 0:
                    await RisingEdge(dut.clk)
        dut.io_a_stream_valid.value = 0

    async def recv_c():
        C_out = [[0.0] * N for _ in range(M)]
        C_out_bits = [[0] * N for _ in range(M)]
        dut.io_c_stream_ready.value = 1
        for m in range(M):
            for n in range(N):
                while dut.io_c_stream_valid.value == 0:
                    await RisingEdge(dut.clk)
                
                if is_floatml:
                    out_sign = int(dut.io_c_stream_payload_0_sign.value)
                    out_exp = int(dut.io_c_stream_payload_0_exponent.value)
                    out_mant = int(dut.io_c_stream_payload_0_mantissa.value)
                    out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
                else:
                    out_bits = int(dut.io_c_stream_payload_0.value)
                    
                C_out_bits[m][n] = out_bits
                C_out[m][n] = dtype.to_float(out_bits)
                await RisingEdge(dut.clk)
        return C_out_bits, C_out

    send_task = cocotb.start_soon(send_a())
    recv_task = cocotb.start_soon(recv_c())
    
    C_out_bits, C_out = await recv_task
    await send_task
    
    # True Math Error Logging
    import numpy as np
    A_np = np.array(A_test)
    B_np = np.array(B_test)
    C_true = np.matmul(A_np, B_np)
    
    errors = []
    for m in range(M):
        for n in range(N):
            out_val = C_out[m][n]
            true_expected = float(C_true[m][n])
            if is_floatml:
                if true_expected != 0:
                    err = abs((out_val - true_expected) / true_expected) * 100
                else:
                    err = abs(out_val) * 100
                errors.append(err)
            else:
                fs_val = (1 << (dtype.bit_width - 1)) - 1
                err = abs(out_val - true_expected) / fs_val * 100
                errors.append(err)
                
    avg_err = sum(errors) / len(errors) if errors else 0.0
    error_str = f"{avg_err:.2f}%" if is_floatml else f"{avg_err:.2f}% FS"
    
    log_msg = f"[{op_name}][{dtype_name}] Test {M}x{K} @ {K}x{N} | Avg Error: {error_str}"
    dut._log.info(log_msg)
    
    if os.environ.get("DEBUG_MATH") == "1":
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "true_math_errors.log")
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
    
    # 3. Verify exactly against matmul_hw
    C_expected = matmul_hw(A_test, B_test, dtype)
    
    # Check bit exactness
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    for m in range(M):
        for n in range(N):
            exp_val = C_expected[m][n]
            exp_bits = dtype.from_float(exp_val)
            out_bits = C_out_bits[m][n]
            out_val = C_out[m][n]
            
            if bit_width > 8 and is_floatml:
                assert abs(out_bits - exp_bits) <= 1, f"HW Mismatch for {op_name} at C[{m}][{n}]: got {out_val} (bits {out_bits}) instead of {dtype.to_float(exp_bits)} (bits {exp_bits})"
            else:
                assert out_bits == exp_bits, f"HW Mismatch for {op_name} at C[{m}][{n}]: got {out_val} (bits {out_bits}) instead of {dtype.to_float(exp_bits)} (bits {exp_bits})"


def get_random_matrices(M, K, N, range_val=5.0, integer=True):
    A = [[0.0]*K for _ in range(M)]
    B = [[0.0]*N for _ in range(K)]
    for m in range(M):
        for k in range(K):
            val = random.uniform(-range_val, range_val)
            A[m][k] = round(val) if integer else val
    for k in range(K):
        for n in range(N):
            val = random.uniform(-range_val, range_val)
            B[k][n] = round(val) if integer else val
    return A, B

# ----------------- VECTOR -----------------
@cocotb.test()
async def cocotb_matmul_vector_i8(dut):
    A, B = get_random_matrices(1, 2, 1, range_val=10.0, integer=True)
    await run_matmul_test(dut, "MatmulVector", "I8", I8, A, B, 1, 2, 1, 2, is_floatml=False)

@cocotb.test()
async def cocotb_matmul_vector_fp8(dut):
    A, B = get_random_matrices(1, 2, 1, range_val=5.0, integer=False)
    await run_matmul_test(dut, "MatmulVector", "FP8", FP8_E4M3, A, B, 1, 2, 1, 2, is_floatml=True)

@cocotb.test()
async def cocotb_matmul_vector_i16(dut):
    A, B = get_random_matrices(1, 2, 1, range_val=100.0, integer=True)
    await run_matmul_test(dut, "MatmulVector", "I16", I16, A, B, 1, 2, 1, 2, is_floatml=False)

@cocotb.test()
async def cocotb_matmul_vector_bf16(dut):
    A, B = get_random_matrices(1, 2, 1, range_val=20.0, integer=False)
    await run_matmul_test(dut, "MatmulVector", "BF16", BF16, A, B, 1, 2, 1, 2, is_floatml=True)

# ----------------- GEMM PARALLEL -----------------
@cocotb.test()
async def cocotb_matmul_gemm_parallel_i8(dut):
    A, B = get_random_matrices(2, 4, 2, range_val=5.0, integer=True)
    await run_matmul_test(dut, "MatmulGEMMPar", "I8", I8, A, B, 2, 4, 2, 2, is_floatml=False)

@cocotb.test()
async def cocotb_matmul_gemm_parallel_fp8(dut):
    A, B = get_random_matrices(2, 4, 2, range_val=3.0, integer=False)
    await run_matmul_test(dut, "MatmulGEMMPar", "FP8", FP8_E4M3, A, B, 2, 4, 2, 2, is_floatml=True)

# ----------------- GEMM SEQUENTIAL -----------------
@cocotb.test()
async def cocotb_matmul_gemm_sequential_i16(dut):
    A, B = get_random_matrices(2, 4, 2, range_val=100.0, integer=True)
    await run_matmul_test(dut, "MatmulGEMMSeq", "I16", I16, A, B, 2, 4, 2, 2, is_floatml=False)

@cocotb.test()
async def cocotb_matmul_gemm_sequential_bf16(dut):
    A, B = get_random_matrices(2, 4, 2, range_val=10.0, integer=False)
    await run_matmul_test(dut, "MatmulGEMMSeq", "BF16", BF16, A, B, 2, 4, 2, 2, is_floatml=True)

# ----------------- DYNAMIC PADDING -----------------
@cocotb.test()
async def cocotb_matmul_dyn_pad_i8(dut):
    A, B = get_random_matrices(1, 3, 1, range_val=10.0, integer=True)
    await run_matmul_test(dut, "MatmulDynPad", "I8", I8, A, B, 1, 3, 1, 2, is_floatml=False)

@cocotb.test()
async def cocotb_matmul_dyn_pad_fp8(dut):
    A, B = get_random_matrices(1, 3, 1, range_val=5.0, integer=False)
    await run_matmul_test(dut, "MatmulDynPad", "FP8", FP8_E4M3, A, B, 1, 3, 1, 2, is_floatml=True)

# ----------------- RUNNERS -----------------
def run_matmul_sim(dtype_filter, testcase_name, toplevel, request=None):
    v_file = run_mill("spinalML.ops.MatmulTest", dtype_filter, toplevel)
    build_dir = f"sim_build/matmul_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_matmul",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_matmul_vector_i8(request): run_matmul_sim("I8", "cocotb_matmul_vector_i8", "MatmulTest_Vector", request)
def test_matmul_vector_fp8(request): run_matmul_sim("FP8", "cocotb_matmul_vector_fp8", "MatmulTest_Vector", request)
def test_matmul_vector_i16(request): run_matmul_sim("I16", "cocotb_matmul_vector_i16", "MatmulTest_Vector", request)
def test_matmul_vector_bf16(request): run_matmul_sim("BF16", "cocotb_matmul_vector_bf16", "MatmulTest_Vector", request)

def test_matmul_gemm_parallel_i8(request): run_matmul_sim("I8", "cocotb_matmul_gemm_parallel_i8", "MatmulTest_GEMM_Parallel", request)
def test_matmul_gemm_parallel_fp8(request): run_matmul_sim("FP8", "cocotb_matmul_gemm_parallel_fp8", "MatmulTest_GEMM_Parallel", request)

def test_matmul_gemm_sequential_i16(request): run_matmul_sim("I16", "cocotb_matmul_gemm_sequential_i16", "MatmulTest_GEMM_Sequential", request)
def test_matmul_gemm_sequential_bf16(request): run_matmul_sim("BF16", "cocotb_matmul_gemm_sequential_bf16", "MatmulTest_GEMM_Sequential", request)

def test_matmul_dyn_pad_i8(request): run_matmul_sim("I8", "cocotb_matmul_dyn_pad_i8", "MatmulTest_DynamicPadding", request)
def test_matmul_dyn_pad_fp8(request): run_matmul_sim("FP8", "cocotb_matmul_dyn_pad_fp8", "MatmulTest_DynamicPadding", request)
