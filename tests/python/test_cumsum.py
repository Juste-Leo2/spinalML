# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest
import os
import random

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from utils.tb_utils import run_mill, copy_roms, seed_random, SEED
from utils.math_metrics import compute_metrics, format_metrics_line, log_math_line

seed_random()

async def run_cumsum_test(dut, op_name, dtype_name, dtype, test_sequence, is_floatml):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_in_stream_valid.value = 0
    dut.io_out_stream_ready.value = 0
    
    async def send_in():
        for row in test_sequence:
            if is_floatml:
                for l in range(2):
                    val_bits = dtype.from_float(row[l])
                    sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                    exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                    mant = val_bits & ((1 << dtype.mant_bits) - 1)
                    getattr(dut, f"io_in_stream_payload_{l}_sign").value = sign
                    getattr(dut, f"io_in_stream_payload_{l}_exponent").value = exp_val
                    getattr(dut, f"io_in_stream_payload_{l}_mantissa").value = mant
            else:
                for l in range(2):
                    val_bits = dtype.from_float(row[l])
                    getattr(dut, f"io_in_stream_payload_{l}").value = val_bits
                    
            dut.io_in_stream_valid.value = 1
            await RisingEdge(dut.clk)
            while dut.io_in_stream_ready.value == 0:
                await RisingEdge(dut.clk)
                
        dut.io_in_stream_valid.value = 0

    async def recv_out():
        out_rows = []
        dut.io_out_stream_ready.value = 1
        for _ in range(len(test_sequence)):
            while dut.io_out_stream_valid.value == 0:
                await RisingEdge(dut.clk)
            
            row_out = []
            for l in range(2):
                if is_floatml:
                    out_sign = int(getattr(dut, f"io_out_stream_payload_{l}_sign").value)
                    out_exp = int(getattr(dut, f"io_out_stream_payload_{l}_exponent").value)
                    out_mant = int(getattr(dut, f"io_out_stream_payload_{l}_mantissa").value)
                    out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
                else:
                    out_bits = int(getattr(dut, f"io_out_stream_payload_{l}").value)
                row_out.append(dtype.to_float(out_bits))
            out_rows.append(row_out)
            await RisingEdge(dut.clk)
        return out_rows

    task_in = cocotb.start_soon(send_in())
    task_out = cocotb.start_soon(recv_out())
    
    out_rows = await task_out
    await task_in
    
    # Check Math
    from golden_models.ops import floatml_add
    
    current_sum_val = [0.0, 0.0]
    
    results = []
    
    for i, row in enumerate(test_sequence):
        for l in range(2):
            val = row[l]
            if is_floatml:
                # Accumulate the value just like the hardware would using FloatML rules
                current_sum_val[l] = dtype.to_float(dtype.from_float(floatml_add(current_sum_val[l], val, dtype)))
            else:
                # Handle Integer wrap-around for I8 and I16
                raw_sum = current_sum_val[l] + val
                mask = (1 << dtype.bit_width) - 1
                wrapped_sum = int(raw_sum) & mask
                # Sign extension
                if wrapped_sum & (1 << (dtype.bit_width - 1)):
                    current_sum_val[l] = wrapped_sum - (1 << dtype.bit_width)
                else:
                    current_sum_val[l] = wrapped_sum
                
            out_val = out_rows[i][l]
            true_expected = current_sum_val[l]
            results.append((out_val, true_expected))
            
            if not is_floatml:
                assert int(out_val) == int(true_expected), f"Mismatch at row {i} lane {l}: got {out_val} exp {true_expected}"

    details = f"L={len(test_sequence)}x2, trials=1, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = format_metrics_line(op_name, dtype_name, compute_metrics([r[0] for r in results], [r[1] for r in results], is_floatml, dtype), is_floatml, details=details)
    dut._log.info(log_msg)
    log_math_line(log_msg)

@cocotb.test()
async def cocotb_cumsum_i8(dut):
    seq = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    await run_cumsum_test(dut, "CumSum", "I8", I8, seq, is_floatml=False)

@cocotb.test()
async def cocotb_cumsum_fp8(dut):
    seq = [[1.0, 2.0], [1.5, 0.5], [0.5, 1.0]]
    await run_cumsum_test(dut, "CumSum", "FP8", FP8_E4M3, seq, is_floatml=True)

def run_cumsum_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.CumsumTest", dtype_filter, "CumsumTestComp")
    build_dir = f"sim_build/cumsum_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="CumsumTestComp",
        module="test_cumsum",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_cumsum_i8(request): run_cumsum_sim("I8", "cocotb_cumsum_i8", request)
def test_cumsum_fp8(request): run_cumsum_sim("FP8", "cocotb_cumsum_fp8", request)
