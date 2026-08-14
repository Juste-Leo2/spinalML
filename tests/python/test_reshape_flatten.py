import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest
import os
import random

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from utils.tb_utils import run_mill, copy_roms

async def run_reshape_test(dut, op_name, dtype_name, dtype, num_transfers, lanes, is_floatml):
    from cocotb.triggers import Timer
    
    dut.io_a_stream_valid.value = 0
    dut.io_reshaped_stream_ready.value = 0
    
    test_data = [[random.uniform(-5.0, 5.0) for _ in range(lanes)] for _ in range(num_transfers)]
    if not is_floatml:
        test_data = [[round(v) for v in row] for row in test_data]
    
    # 1. Send data
    async def send_a():
        for i in range(num_transfers):
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
            await Timer(10, units="ns")
        dut.io_a_stream_valid.value = 0

    # 2. Receive data
    async def recv_c():
        Y_out_bits = [[0] * lanes for _ in range(num_transfers)]
        dut.io_reshaped_stream_ready.value = 1
        for i in range(num_transfers):
            await Timer(10, units="ns")
            
            for l in range(lanes):
                if is_floatml:
                    out_sign = int(getattr(dut, f"io_reshaped_stream_payload_{l}_sign").value)
                    out_exp = int(getattr(dut, f"io_reshaped_stream_payload_{l}_exponent").value)
                    out_mant = int(getattr(dut, f"io_reshaped_stream_payload_{l}_mantissa").value)
                    out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
                else:
                    out_bits = int(getattr(dut, f"io_reshaped_stream_payload_{l}").value)
                Y_out_bits[i][l] = out_bits
        return Y_out_bits

    send_task = cocotb.start_soon(send_a())
    recv_task = cocotb.start_soon(recv_c())
    
    Y_out_bits = await recv_task
    await send_task
    
    # Check exactness
    for i in range(num_transfers):
        for l in range(lanes):
            exp_val = test_data[i][l]
            exp_bits = dtype.from_float(exp_val)
            out_bits = Y_out_bits[i][l]
            assert out_bits == exp_bits, f"HW Mismatch for {op_name} at transfer {i}, lane {l}: got bits {out_bits} instead of {exp_bits}"
            
    dut._log.info(f"[{op_name}][{dtype_name}] Passed Metadata Check")


@cocotb.test()
async def cocotb_reshape_i8(dut):
    await run_reshape_test(dut, "ReshapeFlatten", "I8", I8, 4, 2, is_floatml=False)

@cocotb.test()
async def cocotb_reshape_fp8(dut):
    await run_reshape_test(dut, "ReshapeFlatten", "FP8", FP8_E4M3, 4, 2, is_floatml=True)

@cocotb.test()
async def cocotb_reshape_i16(dut):
    await run_reshape_test(dut, "ReshapeFlatten", "I16", I16, 4, 2, is_floatml=False)

@cocotb.test()
async def cocotb_reshape_bf16(dut):
    await run_reshape_test(dut, "ReshapeFlatten", "BF16", BF16, 4, 2, is_floatml=True)


def run_reshape_sim(dtype_filter, testcase_name, request=None):
    toplevel = "ReshapeTestComp"
    v_file = run_mill("spinalML.ops.ReshapeTest", dtype_filter, toplevel)
    build_dir = f"sim_build/reshape_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_reshape_flatten",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_reshape_i8(request): run_reshape_sim("I8", "cocotb_reshape_i8", request)
def test_reshape_fp8(request): run_reshape_sim("FP8", "cocotb_reshape_fp8", request)
def test_reshape_i16(request): run_reshape_sim("I16", "cocotb_reshape_i16", request)
def test_reshape_bf16(request): run_reshape_sim("BF16", "cocotb_reshape_bf16", request)
