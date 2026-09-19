# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""Cocotb HW tests for CastOp float-source paths (OPS-10).

F2F: BF16 -> FP8 narrowing (RNE/trunc via SPINALML_ROUNDING env).
F2I: BF16 -> I8 round-then-saturate. CastOp is combinational: no clock.
Goldens: cast_float_to_float_hw / cast_float_to_sint_hw (env-following).
"""

import cocotb
from cocotb.triggers import Timer
from cocotb_test.simulator import run
import pytest

from golden_models.dtypes import BF16, FP8_E4M3, I8
from golden_models.ops import cast_float_to_float_hw, cast_float_to_sint_hw
from utils.tb_utils import run_mill, copy_roms, rounding_mode


def bf16_bits(sign, exp, mant):
    return (sign << (BF16.exp_bits + BF16.mant_bits)) | (exp << BF16.mant_bits) | mant


async def sample_f2f(dut, vectors):
    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 1
    got = []
    for (s, e, m) in vectors:
        dut.io_a_stream_payload_0_sign.value = s
        dut.io_a_stream_payload_0_exponent.value = e
        dut.io_a_stream_payload_0_mantissa.value = m
        dut.io_a_stream_valid.value = 1
        await Timer(1, units="ns")
        os = int(dut.io_c_stream_payload_0_sign.value)
        oe = int(dut.io_c_stream_payload_0_exponent.value)
        om = int(dut.io_c_stream_payload_0_mantissa.value)
        got.append((os << (FP8_E4M3.exp_bits + FP8_E4M3.mant_bits)) | (oe << FP8_E4M3.mant_bits) | om)
        dut.io_a_stream_valid.value = 0
        await Timer(1, units="ns")
    return got


async def sample_f2i(dut, vectors):
    dut.io_a_stream_valid.value = 0
    dut.io_c_stream_ready.value = 1
    got = []
    for (s, e, m) in vectors:
        dut.io_a_stream_payload_0_sign.value = s
        dut.io_a_stream_payload_0_exponent.value = e
        dut.io_a_stream_payload_0_mantissa.value = m
        dut.io_a_stream_valid.value = 1
        await Timer(1, units="ns")
        raw = int(dut.io_c_stream_payload_0.value)
        got.append(raw - (1 << I8.bit_width) if raw >= (1 << (I8.bit_width - 1)) else raw)
        dut.io_a_stream_valid.value = 0
        await Timer(1, units="ns")
    return got


def to_signed(raw, bits):
    return raw - (1 << bits) if raw >= (1 << (bits - 1)) else raw


@cocotb.test()
async def cocotb_cast_f2f(dut):
    # 140.5 -> RNE 136 / trunc 128; 1152.0 -> sat 448; -0.0 -> +0; -140.5 keeps sign.
    vectors = [(0, 134, 13), (0, 137, 16), (0, 0, 0), (1, 134, 13)]
    got = await sample_f2f(dut, vectors)
    mode = rounding_mode()
    for (v, out) in zip(vectors, got):
        exp = cast_float_to_float_hw(bf16_bits(*v), BF16, FP8_E4M3, rounding=mode)
        assert out == exp, f"F2F {v} [{mode}]: got {out:#x} instead of {exp:#x}"


@cocotb.test()
async def cocotb_cast_f2i(dut):
    # 43.5 tie, 1152.0 sat, -43.5, -128.0 exact min, +0.0, 4.0 exact,
    # below-0.5 magnitudes (0.375, 0.3008, -0.375 -> 0) and the 0.25/0.5/0.75
    # boundaries (RNE: 0, 0, 1).
    vectors = [(0, 132, 46), (0, 137, 16), (1, 132, 46), (1, 134, 0),
               (0, 0, 0), (0, 129, 0), (0, 125, 64), (0, 125, 26),
               (1, 125, 64), (0, 125, 0), (0, 126, 0), (0, 126, 64)]
    got = await sample_f2i(dut, vectors)
    mode = rounding_mode()
    for (v, out) in zip(vectors, got):
        exp_bits = cast_float_to_sint_hw(bf16_bits(*v), BF16, I8.bit_width, rounding=mode)
        exp = to_signed(exp_bits, I8.bit_width)
        assert out == exp, f"F2I {v} [{mode}]: got {out} instead of {exp}"


def run_cast_float_sim(toplevel, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.CastTest", toplevel.replace("TestComp", ""), toplevel)
    build_dir = f"sim_build/cast_{toplevel.lower()}"
    copy_roms(build_dir)
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_cast_float",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
    )


def test_cast_f2f(request): run_cast_float_sim("CastF2FTestComp", "cocotb_cast_f2f", request)
def test_cast_f2i(request): run_cast_float_sim("CastF2ITestComp", "cocotb_cast_f2i", request)
