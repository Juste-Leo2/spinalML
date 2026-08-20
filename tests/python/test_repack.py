import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest
import os
import random

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from utils.tb_utils import run_mill, copy_roms, seed_random

seed_random()

async def run_repack_test(dut, op_name, dtype_name, dtype, num_transfers_in, in_lanes, out_lanes, is_floatml):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)

    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 0

    test_data = [[random.uniform(-5.0, 5.0) for _ in range(in_lanes)] for _ in range(num_transfers_in)]
    if not is_floatml:
        test_data = [[round(v) for v in row] for row in test_data]

    num_transfers_out = num_transfers_in * in_lanes // out_lanes

    async def send_a():
        for i in range(num_transfers_in):
            for l in range(in_lanes):
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
        Y_out_bits = [[0] * out_lanes for _ in range(num_transfers_out)]
        dut.io_c_stream_ready.value = 1
        for i in range(num_transfers_out):
            while dut.io_c_stream_valid.value == 0:
                await RisingEdge(dut.clk)

            for l in range(out_lanes):
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

    # Repack is a pure gearbox: data passes through unchanged, only regrouped
    # from in_lanes-wide words into out_lanes-wide words, in the same order.
    flat_in = [val for row in test_data for val in row]
    flat_out = [bits for row in Y_out_bits for bits in row]
    for i, (exp_val, out_bits) in enumerate(zip(flat_in, flat_out)):
        exp_bits = dtype.from_float(exp_val)
        assert out_bits == exp_bits, f"HW Mismatch for {op_name} at element {i}: got bits {out_bits} instead of {exp_bits}"

    dut._log.info(f"[{op_name}][{dtype_name}] Passed Metadata Check")


# --------- REPACK ---------
@cocotb.test()
async def cocotb_repack_i8(dut): await run_repack_test(dut, "Repack", "I8", I8, 4, 2, 4, False)
@cocotb.test()
async def cocotb_repack_fp8(dut): await run_repack_test(dut, "Repack", "FP8", FP8_E4M3, 4, 2, 4, True)
@cocotb.test()
async def cocotb_repack_i16(dut): await run_repack_test(dut, "Repack", "I16", I16, 4, 2, 4, False)
@cocotb.test()
async def cocotb_repack_bf16(dut): await run_repack_test(dut, "Repack", "BF16", BF16, 4, 2, 4, True)


def run_repack_sim(dtype_filter, testcase_name, request=None):
    toplevel = "RepackTestComp"
    v_file = run_mill("spinalML.ops.RepackTest", dtype_filter, toplevel)
    build_dir = f"sim_build/repack_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_repack",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_repack_i8(request): run_repack_sim("I8", "cocotb_repack_i8", request)
def test_repack_fp8(request): run_repack_sim("FP8", "cocotb_repack_fp8", request)
def test_repack_i16(request): run_repack_sim("I16", "cocotb_repack_i16", request)
def test_repack_bf16(request): run_repack_sim("BF16", "cocotb_repack_bf16", request)