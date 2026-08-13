import cocotb
from cocotb_test.simulator import run
import pytest

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import floatml_mul
from utils.tb_utils import run_mill, copy_roms
from utils.cocotb_helpers import run_binary_test

@cocotb.test()
async def cocotb_mul_i8(dut):
    def expected_fn(a, b): return (I8.from_float(a) * I8.from_float(b)) & ((1 << I8.bit_width) - 1)
    await run_binary_test(dut, "Mul", "I8", I8, [(5.0, 2.0), (10.0, -10.0)], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=lambda a,b: a*b, edge_cases=[(10.0, -10.0)])

@cocotb.test()
async def cocotb_mul_fp8(dut):
    def expected_fn(a, b): return FP8_E4M3.from_float(floatml_mul(FP8_E4M3.to_float(FP8_E4M3.from_float(a)), FP8_E4M3.to_float(FP8_E4M3.from_float(b)), FP8_E4M3))
    await run_binary_test(dut, "Mul", "FP8", FP8_E4M3, [(2.5, 1.5), (-1.5, 2.5)], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=lambda a,b: a*b)

@cocotb.test()
async def cocotb_mul_i16(dut):
    def expected_fn(a, b): return (I16.from_float(a) * I16.from_float(b)) & ((1 << I16.bit_width) - 1)
    await run_binary_test(dut, "Mul", "I16", I16, [(20.0, 10.0), (-100.0, 300.0)], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=lambda a,b: a*b, edge_cases=[(-100.0, 300.0)])

@cocotb.test()
async def cocotb_mul_bf16(dut):
    def expected_fn(a, b): return BF16.from_float(floatml_mul(BF16.to_float(BF16.from_float(a)), BF16.to_float(BF16.from_float(b)), BF16))
    await run_binary_test(dut, "Mul", "BF16", BF16, [(20.25, 10.5), (100.0, -100.0)], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=lambda a,b: a*b, edge_cases=[(100.0, -100.0)])

def run_mul_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.MulTest", dtype_filter, "MulTestComp")
    build_dir = f"sim_build/mul_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="MulTestComp",
        module="test_mul",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_mul_i8(request): run_mul_sim("I8", "cocotb_mul_i8", request)
def test_mul_fp8(request): run_mul_sim("FP8", "cocotb_mul_fp8", request)
def test_mul_i16(request): run_mul_sim("I16", "cocotb_mul_i16", request)
def test_mul_bf16(request): run_mul_sim("BF16", "cocotb_mul_bf16", request)
