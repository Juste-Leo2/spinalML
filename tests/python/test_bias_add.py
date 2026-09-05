# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest
import random

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import bias_add_hw
from utils.tb_utils import run_mill, copy_roms, seed_random

seed_random()


def to_signed(bits, width):
    if bits & (1 << (width - 1)):
        return bits - (1 << width)
    return bits


async def run_bias_add_test(dut, op_name, dtype_name, dtype, lanes, num_bias, num_transfers,
                            is_floatml, toplevel):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)

    dut.io_a_stream_valid.value = 0
    dut.io_b_stream_valid.value = 0
    dut.io_c_stream_ready.value = 0

    bias = [random.uniform(-5.0, 5.0) for _ in range(num_bias)]
    test_data = [[random.uniform(-5.0, 5.0) for _ in range(lanes)] for _ in range(num_transfers)]
    if not is_floatml:
        bias = [round(v) for v in bias]
        test_data = [[round(v) for v in row] for row in test_data]

    # 1. Send the bias vector (lanes=1, sequential)
    for b in bias:
        val_bits = dtype.from_float(b)
        if is_floatml:
            dut.io_b_stream_payload_0_sign.value = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
            dut.io_b_stream_payload_0_exponent.value = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
            dut.io_b_stream_payload_0_mantissa.value = val_bits & ((1 << dtype.mant_bits) - 1)
        else:
            dut.io_b_stream_payload_0.value = val_bits
        dut.io_b_stream_valid.value = 1
        await RisingEdge(dut.clk)
        while dut.io_b_stream_ready.value == 0:
            await RisingEdge(dut.clk)
    dut.io_b_stream_valid.value = 0

    # 2. Send A and receive C concurrently
    async def send_a():
        for row in test_data:
            for l in range(lanes):
                val = row[l]
                val_bits = dtype.from_float(val)
                if is_floatml:
                    getattr(dut, f"io_a_stream_payload_{l}_sign").value = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                    getattr(dut, f"io_a_stream_payload_{l}_exponent").value = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                    getattr(dut, f"io_a_stream_payload_{l}_mantissa").value = val_bits & ((1 << dtype.mant_bits) - 1)
                else:
                    getattr(dut, f"io_a_stream_payload_{l}").value = val_bits
            dut.io_a_stream_valid.value = 1
            await RisingEdge(dut.clk)
            while dut.io_a_stream_ready.value == 0:
                await RisingEdge(dut.clk)
        dut.io_a_stream_valid.value = 0

    async def recv_c():
        dut.io_c_stream_ready.value = 1
        Y_out = []
        for _ in range(num_transfers):
            while dut.io_c_stream_valid.value == 0:
                await RisingEdge(dut.clk)
            row = []
            for l in range(lanes):
                if is_floatml:
                    sign = int(getattr(dut, f"io_c_stream_payload_{l}_sign").value)
                    exp = int(getattr(dut, f"io_c_stream_payload_{l}_exponent").value)
                    mant = int(getattr(dut, f"io_c_stream_payload_{l}_mantissa").value)
                    row.append((sign << (dtype.exp_bits + dtype.mant_bits)) | (exp << dtype.mant_bits) | mant)
                else:
                    row.append(int(getattr(dut, f"io_c_stream_payload_{l}").value))
            Y_out.append(row)
            await RisingEdge(dut.clk)
        return Y_out

    send_task = cocotb.start_soon(send_a())
    recv_task = cocotb.start_soon(recv_c())

    Y_out = await recv_task
    await send_task

    flat_in = [v for row in test_data for v in row]
    flat_out = [b for row in Y_out for b in row]
    expected = bias_add_hw(flat_in, bias, dtype)

    for i, (out_val, exp_val) in enumerate(zip(flat_out, expected)):
        if is_floatml:
            exp_bits = dtype.from_float(exp_val)
            assert out_val == exp_bits, f"HW Mismatch for {op_name} at element {i}: got bits {out_val} instead of {exp_bits}"
        else:
            out_signed = to_signed(out_val, dtype.bit_width)
            assert out_signed == exp_val, f"HW Mismatch for {op_name} at element {i}: got {out_signed} instead of {exp_val}"

    dut._log.info(f"[{op_name}][{dtype_name}] Passed BiasAdd Check")


# --------- BiasAdd scalar (lanes=1, N=1) ---------
@cocotb.test()
async def cocotb_bias_add_i8_scalar(dut):
    await run_bias_add_test(dut, "BiasAddScalar", "I8", I8, 1, 1, 4, False, "BiasAddTestComp")

# --------- BiasAdd broadcast (lanes=2, N=2) ---------
@cocotb.test()
async def cocotb_bias_add_i8_bcast(dut):
    await run_bias_add_test(dut, "BiasAddBcast", "I8", I8, 2, 2, 2, False, "BiasAddTestComp2")

@cocotb.test()
async def cocotb_bias_add_fp8_bcast(dut):
    await run_bias_add_test(dut, "BiasAddBcast", "FP8", FP8_E4M3, 2, 2, 2, True, "BiasAddTestComp2")

@cocotb.test()
async def cocotb_bias_add_i16_bcast(dut):
    await run_bias_add_test(dut, "BiasAddBcast", "I16", I16, 2, 2, 2, False, "BiasAddTestComp2")

@cocotb.test()
async def cocotb_bias_add_bf16_bcast(dut):
    await run_bias_add_test(dut, "BiasAddBcast", "BF16", BF16, 2, 2, 2, True, "BiasAddTestComp2")


def run_bias_add_sim(dtype_filter, testcase_name, toplevel, request=None):
    v_file = run_mill("spinalML.ops.BiasAddTest", dtype_filter, toplevel)
    build_dir = f"sim_build/biasadd_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_bias_add",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_bias_add_i8_scalar(request): run_bias_add_sim("I8", "cocotb_bias_add_i8_scalar", "BiasAddTestComp", request)
def test_bias_add_i8_bcast(request): run_bias_add_sim("I8", "cocotb_bias_add_i8_bcast", "BiasAddTestComp2", request)
def test_bias_add_fp8_bcast(request): run_bias_add_sim("FP8", "cocotb_bias_add_fp8_bcast", "BiasAddTestComp2", request)
def test_bias_add_i16_bcast(request): run_bias_add_sim("I16", "cocotb_bias_add_i16_bcast", "BiasAddTestComp2", request)
def test_bias_add_bf16_bcast(request): run_bias_add_sim("BF16", "cocotb_bias_add_bf16_bcast", "BiasAddTestComp2", request)