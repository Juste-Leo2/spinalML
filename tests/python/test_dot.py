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

async def run_dot_test(dut, op_name, dtype_name, dtype, A_test, B_test, K, lanes, is_floatml):
    """
    Generic Cocotb test method for DotOp (wraps matmul M=1, N=1).
    A_test: list of K values (1D vector)
    B_test: list of K values (1D vector)
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

    # 1. Send vector B (buffered first, like matmul)
    for chunk in range(chunks_K):
        if is_floatml:
            for l in range(lanes):
                k_idx = chunk * lanes + l
                b_val = B_test[k_idx] if k_idx < K else 0.0
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
                b_val = B_test[k_idx] if k_idx < K else 0.0
                val_bits = dtype.from_float(b_val)
                getattr(dut, f"io_b_stream_payload_{l}").value = val_bits

        dut.io_b_stream_valid.value = 1
        await RisingEdge(dut.clk)
        while dut.io_b_stream_ready.value == 0:
            await RisingEdge(dut.clk)
    dut.io_b_stream_valid.value = 0

    # 2. Send vector A and wait for the scalar output C concurrently
    async def send_a():
        for chunk in range(chunks_K):
            if is_floatml:
                for l in range(lanes):
                    k_idx = chunk * lanes + l
                    a_val = A_test[k_idx] if k_idx < K else 0.0
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
                    a_val = A_test[k_idx] if k_idx < K else 0.0
                    val_bits = dtype.from_float(a_val)
                    getattr(dut, f"io_a_stream_payload_{l}").value = val_bits

            dut.io_a_stream_valid.value = 1
            await RisingEdge(dut.clk)
            while dut.io_a_stream_ready.value == 0:
                await RisingEdge(dut.clk)
        dut.io_a_stream_valid.value = 0

    async def recv_c():
        dut.io_c_stream_ready.value = 1
        while dut.io_c_stream_valid.value == 0:
            await RisingEdge(dut.clk)

        if is_floatml:
            out_sign = int(dut.io_c_stream_payload_0_sign.value)
            out_exp = int(dut.io_c_stream_payload_0_exponent.value)
            out_mant = int(dut.io_c_stream_payload_0_mantissa.value)
            out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
        else:
            out_bits = int(dut.io_c_stream_payload_0.value)
        await RisingEdge(dut.clk)
        return out_bits

    send_task = cocotb.start_soon(send_a())
    recv_task = cocotb.start_soon(recv_c())

    C_out_bits = await recv_task
    await send_task

    # True Math Error Logging
    import numpy as np
    A_np = np.array(A_test).reshape(1, K)
    B_np = np.array(B_test).reshape(K, 1)
    C_true = float(np.matmul(A_np, B_np)[0][0])

    out_val = dtype.to_float(C_out_bits)
    if is_floatml:
        if C_true != 0:
            err = abs((out_val - C_true) / C_true) * 100
        else:
            err = abs(out_val) * 100
    else:
        fs_val = (1 << (dtype.bit_width - 1)) - 1
        err = abs(out_val - C_true) / fs_val * 100

    log_msg = f"[{op_name}][{dtype_name}] Test K={K} lanes={lanes} | Avg Error: {err:.2f}%"
    dut._log.info(log_msg)

    if os.environ.get("DEBUG_MATH") == "1":
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "true_math_errors.log")
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")

    # Verify bit-exactly against matmul_hw golden (M=1, N=1)
    C_expected = matmul_hw([A_test], [[b] for b in B_test], dtype)
    exp_val = C_expected[0][0]
    exp_bits = dtype.from_float(exp_val)

    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    if bit_width > 8 and is_floatml:
        assert abs(C_out_bits - exp_bits) <= 1, f"HW Mismatch for {op_name} at C[0][0]: got {dtype.to_float(C_out_bits)} (bits {C_out_bits}) instead of {dtype.to_float(exp_bits)} (bits {exp_bits})"
    else:
        assert C_out_bits == exp_bits, f"HW Mismatch for {op_name} at C[0][0]: got {dtype.to_float(C_out_bits)} (bits {C_out_bits}) instead of {dtype.to_float(exp_bits)} (bits {exp_bits})"


def get_random_vectors(K, range_val=5.0, integer=True):
    A = []
    B = []
    for k in range(K):
        val = random.uniform(-range_val, range_val)
        A.append(round(val) if integer else val)
    for k in range(K):
        val = random.uniform(-range_val, range_val)
        B.append(round(val) if integer else val)
    return A, B


# ----------------- DOT (N=8, lanes=2) -----------------
@cocotb.test()
async def cocotb_dot_i8(dut):
    A, B = get_random_vectors(8, range_val=10.0, integer=True)
    await run_dot_test(dut, "Dot", "I8", I8, A, B, 8, 2, is_floatml=False)

@cocotb.test()
async def cocotb_dot_fp8(dut):
    A, B = get_random_vectors(8, range_val=5.0, integer=False)
    await run_dot_test(dut, "Dot", "FP8", FP8_E4M3, A, B, 8, 2, is_floatml=True)

@cocotb.test()
async def cocotb_dot_i16(dut):
    A, B = get_random_vectors(8, range_val=100.0, integer=True)
    await run_dot_test(dut, "Dot", "I16", I16, A, B, 8, 2, is_floatml=False)

@cocotb.test()
async def cocotb_dot_bf16(dut):
    A, B = get_random_vectors(8, range_val=20.0, integer=False)
    await run_dot_test(dut, "Dot", "BF16", BF16, A, B, 8, 2, is_floatml=True)

# ----------------- RUNNERS -----------------
def run_dot_sim(dtype_filter, testcase_name, request=None):
    toplevel = "DotTestComp"
    v_file = run_mill("spinalML.ops.DotTest", dtype_filter, toplevel)
    build_dir = f"sim_build/dot_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_dot",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_dot_i8(request): run_dot_sim("I8", "cocotb_dot_i8", request)
def test_dot_fp8(request): run_dot_sim("FP8", "cocotb_dot_fp8", request)
def test_dot_i16(request): run_dot_sim("I16", "cocotb_dot_i16", request)
def test_dot_bf16(request): run_dot_sim("BF16", "cocotb_dot_bf16", request)