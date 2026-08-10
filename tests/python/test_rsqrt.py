import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import pytest

from golden_models.dtypes import I4, FP4_E2M1, I16, BF16
from golden_models.ops import rsqrt
from utils.tb_utils import run_mill, copy_roms
# Important : importer la fixture de nettoyage pour qu'elle s'exécute avec pytest
from utils.tb_utils import cleanup_verilog

# =========================================================================
# Cocotb Test Logic
# =========================================================================

async def run_test_rsqrt(dut, dtype, test_values, is_floatml):
    """Méthode de test générique pour Rsqrt"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    for val in test_values:
        val_bits = dtype.from_float(val)
        
        # Injection
        if is_floatml:
            sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
            exp = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
            mant = val_bits & ((1 << dtype.mant_bits) - 1)
            
            dut.io_a_stream_payload_0_sign.value = sign
            dut.io_a_stream_payload_0_exponent.value = exp
            dut.io_a_stream_payload_0_mantissa.value = mant
            
            dut.io_a_stream_payload_1_sign.value = sign
            dut.io_a_stream_payload_1_exponent.value = exp
            dut.io_a_stream_payload_1_mantissa.value = mant
        else:
            dut.io_a_stream_payload_0.value = val_bits
            dut.io_a_stream_payload_1.value = val_bits
            
        dut.io_a_stream_valid.value = 1
        dut.io_c_stream_ready.value = 1
        
        # Latch input
        await RisingEdge(dut.clk)
        dut.io_a_stream_valid.value = 0
        
        # Wait for result valid
        cycles = 0
        while dut.io_c_stream_valid.value != 1 and cycles < 10:
            await RisingEdge(dut.clk)
            cycles += 1
            
        assert dut.io_c_stream_valid.value == 1, f"Timeout pour {val} !"
        
        if is_floatml:
            out_sign = int(dut.io_c_stream_payload_0_sign.value)
            out_exp = int(dut.io_c_stream_payload_0_exponent.value)
            out_mant = int(dut.io_c_stream_payload_0_mantissa.value)
            out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
        else:
            out_bits = int(dut.io_c_stream_payload_0.value)
        
        # Calcul du Golden Model
        bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
        if bit_width > 8 and not is_floatml:
            # Pour I16, le PWL produit une erreur massive près de 0, on simule l'interpolation exacte.
            from golden_models.ops import pwl_rsqrt_int
            expected_bits = pwl_rsqrt_int(val, bit_width, index_bits=8)
            expected = dtype.to_float(expected_bits)
        else:
            expected = rsqrt(dtype.to_float(val_bits), dtype)
            expected_bits = dtype.from_float(expected)
        
        # Comparaison !
        if bit_width > 8 and is_floatml:
            out_float = dtype.to_float(out_bits)
            # Tolérance de 20% pour le PWL Flottant
            margin = 0.20 * expected if expected != 0 else 0.20
            assert abs(out_float - expected) <= margin, f"PWL Float Failed for {val}: got {out_float} au lieu de {expected}"
        else:
            assert out_bits == expected_bits, f"LUT/PWL-Int Failed for {val}: got {dtype.to_float(out_bits)} au lieu de {expected}"


@cocotb.test()
async def cocotb_rsqrt_i4(dut):
    # Tests de 1 à 7 pour un I4 signés (max 7)
    await run_test_rsqrt(dut, I4, [1.0, 2.0, 3.0, 4.0, 7.0], is_floatml=False)

@cocotb.test()
async def cocotb_rsqrt_fp4(dut):
    await run_test_rsqrt(dut, FP4_E2M1, [1.0, 1.614, 2.0, 3.0], is_floatml=True)

@cocotb.test()
async def cocotb_rsqrt_i16(dut):
    # Tests pour I16 (max 32767)
    await run_test_rsqrt(dut, I16, [1.0, 4.0, 10.0, 100.0, 30000.0], is_floatml=False)

@cocotb.test()
async def cocotb_rsqrt_bf16(dut):
    await run_test_rsqrt(dut, BF16, [1.0, 1.614, 2.0, 100.0], is_floatml=True)

# =========================================================================
# Pytest Launchers
# =========================================================================

def run_rsqrt_sim(dtype_filter, testcase_name):
    v_file = run_mill("spinalML.ops.RsqrtTest", dtype_filter, "RsqrtTestComp")
    build_dir = f"sim_build/rsqrt_{dtype_filter.lower()}"
    copy_roms(build_dir)
    
    run(
        language="verilog",
        verilog_sources=[v_file],
        toplevel="RsqrtTestComp",
        module="test_rsqrt",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"]
    )

def test_rsqrt_i4():
    run_rsqrt_sim("I4", "cocotb_rsqrt_i4")

def test_rsqrt_fp4():
    run_rsqrt_sim("FP4", "cocotb_rsqrt_fp4")

def test_rsqrt_i16():
    run_rsqrt_sim("I16", "cocotb_rsqrt_i16")

def test_rsqrt_bf16():
    run_rsqrt_sim("BF16", "cocotb_rsqrt_bf16")
