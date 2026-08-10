import cocotb
from cocotb_test.simulator import run
import pytest
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16
from golden_models.ops import rsqrt, pwl_rsqrt_int, pwl_rsqrt_float
from utils.tb_utils import run_mill, copy_roms
from utils.tb_utils import cleanup_verilog
from utils.cocotb_helpers import run_unary_test

# =========================================================================
# Cocotb Test Logic
# =========================================================================

def true_rsqrt(val):
    if val <= 0: return 0.0
    return 1.0 / np.sqrt(val)

@cocotb.test()
async def cocotb_rsqrt_i8(dut):
    def expected_fn(val):
        return I8.from_float(rsqrt(I8.to_float(I8.from_float(val)), I8))
    await run_unary_test(dut, "Rsqrt", "I8", I8, [4.0, 7.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_rsqrt, edge_cases=[7.0])

@cocotb.test()
async def cocotb_rsqrt_fp8(dut):
    def expected_fn(val):
        return FP8_E4M3.from_float(rsqrt(FP8_E4M3.to_float(FP8_E4M3.from_float(val)), FP8_E4M3))
    await run_unary_test(dut, "Rsqrt", "FP8", FP8_E4M3, [4.0, 7.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_rsqrt, edge_cases=[7.0])

@cocotb.test()
async def cocotb_rsqrt_i16(dut):
    def expected_fn(val):
        return pwl_rsqrt_int(val, 16, index_bits=8)
    await run_unary_test(dut, "Rsqrt", "I16", I16, [4.0, 314.0], is_floatml=False, expected_bits_fn=expected_fn, true_math_fn=true_rsqrt, edge_cases=[314.0])

@cocotb.test()
async def cocotb_rsqrt_bf16(dut):
    def expected_fn(val):
        return pwl_rsqrt_float(val, BF16, index_bits=8)
    await run_unary_test(dut, "Rsqrt", "BF16", BF16, [2.0, 100.0], is_floatml=True, expected_bits_fn=expected_fn, true_math_fn=true_rsqrt, edge_cases=[100.0])

# =========================================================================
# Pytest Launchers
# =========================================================================

def run_rsqrt_sim(dtype_filter, testcase_name, request=None):
    v_file = run_mill("spinalML.ops.RsqrtTest", dtype_filter, "RsqrtTestComp")
    build_dir = f"sim_build/rsqrt_{dtype_filter.lower()}"
    copy_roms(build_dir)
    
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="RsqrtTestComp",
        module="test_rsqrt",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
        extra_env={"DEBUG_MATH": debug_flag}
    )

def test_rsqrt_i8(request): run_rsqrt_sim("I8", "cocotb_rsqrt_i8", request)
def test_rsqrt_fp8(request): run_rsqrt_sim("FP8", "cocotb_rsqrt_fp8", request)
def test_rsqrt_i16(request): run_rsqrt_sim("I16", "cocotb_rsqrt_i16", request)
def test_rsqrt_bf16(request): run_rsqrt_sim("BF16", "cocotb_rsqrt_bf16", request)
