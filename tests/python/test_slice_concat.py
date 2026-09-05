# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest
import os
import random

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from utils.tb_utils import run_mill, copy_roms

async def run_slice_test(dut, op_name, dtype_name, dtype, num_transfers_in, lanes, start, end, is_floatml):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 0
    
    test_data = [[random.uniform(-5.0, 5.0) for _ in range(lanes)] for _ in range(num_transfers_in)]
    if not is_floatml:
        test_data = [[round(v) for v in row] for row in test_data]
        
    num_transfers_out = end - start
    
    async def send_a():
        for i in range(num_transfers_in):
            for l in range(lanes):
                val = test_data[i][l]
                val_bits = dtype.from_float(val)
                if is_floatml:
                    sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                    exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                    mant = val_bits & ((1 << dtype.mant_bits) - 1)
                    getattr(dut, f"io_a_stream_payload_{l}_sign").value = sign
                    getattr(dut, f"io_a_stream_payload_{l}_exponent").value = exp_val
                    getattr(dut, f"io_a_stream_payload_{l}_mantissa").value = mant
                else:
                    getattr(dut, f"io_a_stream_payload_{l}").value = val_bits
                    
            dut.io_a_stream_valid.value = 1
            await RisingEdge(dut.clk)
            while dut.io_a_stream_ready.value == 0:
                await RisingEdge(dut.clk)
        dut.io_a_stream_valid.value = 0

    async def recv_c():
        Y_out_bits = [[0] * lanes for _ in range(num_transfers_out)]
        dut.io_c_stream_ready.value = 1
        for i in range(num_transfers_out):
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
                Y_out_bits[i][l] = out_bits
            await RisingEdge(dut.clk)
        return Y_out_bits

    send_task = cocotb.start_soon(send_a())
    recv_task = cocotb.start_soon(recv_c())
    
    Y_out_bits = await recv_task
    await send_task
    
    # Check exactness
    for i in range(num_transfers_out):
        for l in range(lanes):
            exp_val = test_data[start + i][l]
            exp_bits = dtype.from_float(exp_val)
            out_bits = Y_out_bits[i][l]
            assert out_bits == exp_bits, f"HW Mismatch for {op_name} at transfer {i}, lane {l}: got bits {out_bits} instead of {exp_bits}"
            
    dut._log.info(f"[{op_name}][{dtype_name}] Passed Metadata Check")


async def run_concat_test(dut, op_name, dtype_name, dtype, num_A, num_B, lanes, is_floatml):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_a_stream_valid.value = 0
    dut.io_b_stream_valid.value = 0
    dut.io_c_stream_ready.value = 0
    
    test_A = [[random.uniform(-5.0, 5.0) for _ in range(lanes)] for _ in range(num_A)]
    test_B = [[random.uniform(-5.0, 5.0) for _ in range(lanes)] for _ in range(num_B)]
    if not is_floatml:
        test_A = [[round(v) for v in row] for row in test_A]
        test_B = [[round(v) for v in row] for row in test_B]
        
    num_transfers_out = num_A + num_B
    
    async def send_tensor(stream_name, data_matrix):
        for i in range(len(data_matrix)):
            for l in range(lanes):
                val = data_matrix[i][l]
                val_bits = dtype.from_float(val)
                if is_floatml:
                    sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                    exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                    mant = val_bits & ((1 << dtype.mant_bits) - 1)
                    getattr(dut, f"io_{stream_name}_stream_payload_{l}_sign").value = sign
                    getattr(dut, f"io_{stream_name}_stream_payload_{l}_exponent").value = exp_val
                    getattr(dut, f"io_{stream_name}_stream_payload_{l}_mantissa").value = mant
                else:
                    getattr(dut, f"io_{stream_name}_stream_payload_{l}").value = val_bits
                    
            getattr(dut, f"io_{stream_name}_stream_valid").value = 1
            await RisingEdge(dut.clk)
            while getattr(dut, f"io_{stream_name}_stream_ready").value == 0:
                await RisingEdge(dut.clk)
        getattr(dut, f"io_{stream_name}_stream_valid").value = 0

    async def recv_c():
        Y_out_bits = [[0] * lanes for _ in range(num_transfers_out)]
        dut.io_c_stream_ready.value = 1
        for i in range(num_transfers_out):
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
                Y_out_bits[i][l] = out_bits
            await RisingEdge(dut.clk)
        return Y_out_bits

    send_a_task = cocotb.start_soon(send_tensor("a", test_A))
    send_b_task = cocotb.start_soon(send_tensor("b", test_B))
    recv_task = cocotb.start_soon(recv_c())
    
    Y_out_bits = await recv_task
    await send_a_task
    await send_b_task
    
    # Check exactness
    expected_data = test_A + test_B
    for i in range(num_transfers_out):
        for l in range(lanes):
            exp_val = expected_data[i][l]
            exp_bits = dtype.from_float(exp_val)
            out_bits = Y_out_bits[i][l]
            assert out_bits == exp_bits, f"HW Mismatch for {op_name} at transfer {i}, lane {l}: got bits {out_bits} instead of {exp_bits}"
            
    dut._log.info(f"[{op_name}][{dtype_name}] Passed Metadata Check")

# --------- SLICE ---------
@cocotb.test()
async def cocotb_slice_i8(dut): await run_slice_test(dut, "Slice", "I8", I8, 4, 2, 1, 3, False)
@cocotb.test()
async def cocotb_slice_fp8(dut): await run_slice_test(dut, "Slice", "FP8", FP8_E4M3, 4, 2, 1, 3, True)
@cocotb.test()
async def cocotb_slice_i16(dut): await run_slice_test(dut, "Slice", "I16", I16, 4, 2, 1, 3, False)
@cocotb.test()
async def cocotb_slice_bf16(dut): await run_slice_test(dut, "Slice", "BF16", BF16, 4, 2, 1, 3, True)

# --------- CONCATENATE ---------
@cocotb.test()
async def cocotb_concat_i8(dut): await run_concat_test(dut, "Concatenate", "I8", I8, 2, 4, 2, False)
@cocotb.test()
async def cocotb_concat_fp8(dut): await run_concat_test(dut, "Concatenate", "FP8", FP8_E4M3, 2, 4, 2, True)
@cocotb.test()
async def cocotb_concat_i16(dut): await run_concat_test(dut, "Concatenate", "I16", I16, 2, 4, 2, False)
@cocotb.test()
async def cocotb_concat_bf16(dut): await run_concat_test(dut, "Concatenate", "BF16", BF16, 2, 4, 2, True)


def run_sim(test_module, dtype_filter, testcase_name, request=None):
    toplevel = f"{test_module}TestComp"
    v_file = run_mill(f"spinalML.ops.{test_module}Test", dtype_filter, toplevel)
    build_dir = f"sim_build/{test_module.lower()}_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_slice_concat",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_slice_i8(request): run_sim("Slice", "I8", "cocotb_slice_i8", request)
def test_slice_fp8(request): run_sim("Slice", "FP8", "cocotb_slice_fp8", request)
def test_slice_i16(request): run_sim("Slice", "I16", "cocotb_slice_i16", request)
def test_slice_bf16(request): run_sim("Slice", "BF16", "cocotb_slice_bf16", request)

def test_concat_i8(request): run_sim("Concatenate", "I8", "cocotb_concat_i8", request)
def test_concat_fp8(request): run_sim("Concatenate", "FP8", "cocotb_concat_fp8", request)
def test_concat_i16(request): run_sim("Concatenate", "I16", "cocotb_concat_i16", request)
def test_concat_bf16(request): run_sim("Concatenate", "BF16", "cocotb_concat_bf16", request)
