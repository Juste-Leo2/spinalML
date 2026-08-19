import cocotb
from cocotb_test.simulator import run
import pytest
import math

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import sigmoid_hw
from utils.tb_utils import run_mill, copy_roms
from utils.tb_utils import cleanup_verilog
from utils.cocotb_helpers import run_unary_test

# =========================================================================
# Cocotb Test Logic
# =========================================================================

def true_sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

@cocotb.test()
async def cocotb_sigmoid_i8(dut):
    def expected_fn(val):
        return sigmoid_hw(val, I8)
    await run_unary_test(dut, "Sigmoid", "I8", I8, [2.0, -2.0, 5.0, -5.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_sigmoid, edge_cases=[-4.0])

@cocotb.test()
async def cocotb_sigmoid_fp8(dut):
    def expected_fn(val):
        return sigmoid_hw(val, FP8_E4M3)
    await run_unary_test(dut, "Sigmoid", "FP8", FP8_E4M3, [2.0, -2.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_sigmoid, edge_cases=[0.0])

@cocotb.test()
async def cocotb_sigmoid_i16(dut):
    def expected_fn(val):
        return sigmoid_hw(val, I16)
    await run_unary_test(dut, "Sigmoid", "I16", I16, [100.0, -100.0, 5.0, -5.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_sigmoid, edge_cases=[-100.0])

@cocotb.test()
async def cocotb_sigmoid_bf16(dut):
    def expected_fn(val):
        return sigmoid_hw(val, BF16)
    await run_unary_test(dut, "Sigmoid", "BF16", BF16, [2.5, -2.5, 5.0, -5.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_sigmoid, edge_cases=[0.0])

# =========================================================================
# Pytest Launchers
# =========================================================================

def run_sigmoid_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.activations.SigmoidTest", dtype_filter, "SigmoidTestComp")
    build_dir = f"sim_build/sigmoid_{dtype_filter.lower()}"
    copy_roms(build_dir)
    
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="SigmoidTestComp",
        module="test_sigmoid",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_sigmoid_i8(request): run_sigmoid_sim("I8", "cocotb_sigmoid_i8", request)
def test_sigmoid_fp8(request): run_sigmoid_sim("FP8", "cocotb_sigmoid_fp8", request)
def test_sigmoid_i16(request): run_sigmoid_sim("I16", "cocotb_sigmoid_i16", request)
def test_sigmoid_bf16(request): run_sigmoid_sim("BF16", "cocotb_sigmoid_bf16", request)