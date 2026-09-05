# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import cocotb
from cocotb.triggers import Timer
from cocotb_test.simulator import run
import pytest
import random

from golden_models.dtypes import I32
from golden_models.ops import requantize_hw
from utils.tb_utils import run_mill, copy_roms, seed_random

seed_random()


def to_signed(bits, width):
    if bits & (1 << (width - 1)):
        return bits - (1 << width)
    return bits


async def run_requantize_test(dut, op_name, num_transfers, lanes=4):
    # RequantizeOp is purely combinational: no clk/reset in the Verilog top.
    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 1

    edges = [0, 1000, -1000, 508, -508, (1 << 31) - 1, -(1 << 31), 127, -128]
    rows = []
    for _ in range(num_transfers):
        rows.append([random.randint(-(1 << 31), (1 << 31) - 1) for _ in range(lanes)])
    edge_rows = [edges[i * lanes:(i + 1) * lanes] for i in range((len(edges) + lanes - 1) // lanes)]
    while len(edge_rows[-1]) < lanes:
        edge_rows[-1].append(0)
    rows.extend(edge_rows)
    flat_in = [v for row in rows for v in row]

    flat_out = []
    for row in rows:
        for l in range(lanes):
            getattr(dut, f"io_a_stream_payload_{l}").value = I32.from_float(row[l])
        dut.io_a_stream_valid.value = 1
        await Timer(1, units="ns")
        flat_out.extend(
            to_signed(int(getattr(dut, f"io_c_stream_payload_{l}").value), 8) for l in range(lanes)
        )
        dut.io_a_stream_valid.value = 0
        await Timer(1, units="ns")

    expected = [requantize_hw(x, 32, 8, 2) for x in flat_in]
    for i, (out_val, exp_val) in enumerate(zip(flat_out, expected)):
        assert out_val == exp_val, f"HW Mismatch for {op_name} at element {i}: got {out_val} instead of {exp_val}"

    dut._log.info(f"[{op_name}][I32->I8] Passed Requantize Check")


@cocotb.test()
async def cocotb_requantize(dut):
    await run_requantize_test(dut, "Requantize", 4)


def run_requantize_sim(request=None):
    toplevel = "RequantizeTestComp"
    v_file = run_mill("spinalML.ops.RequantizeTest", "I32_I8", toplevel)
    build_dir = f"sim_build/requantize_{toplevel.lower()}_i32_i8"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_requantize",
        testcase="cocotb_requantize",
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_requantize(request): run_requantize_sim(request)