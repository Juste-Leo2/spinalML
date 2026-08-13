import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

import os

async def run_unary_test(dut, op_name, dtype_name, dtype, test_values, is_floatml, expected_bits_fn, true_math_fn, edge_cases=None):
    """
    Generic Cocotb test method for Unary Operators (Rsqrt, Exp, etc).
    Performs bit-exact verification against expected_bits_fn (HW golden model).
    Logs the true mathematical error against true_math_fn for analysis.
    """
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
            exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
            mant = val_bits & ((1 << dtype.mant_bits) - 1)
            
            dut.io_a_stream_payload_0_sign.value = sign
            dut.io_a_stream_payload_0_exponent.value = exp_val
            dut.io_a_stream_payload_0_mantissa.value = mant
            
            dut.io_a_stream_payload_1_sign.value = sign
            dut.io_a_stream_payload_1_exponent.value = exp_val
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
            
        assert dut.io_c_stream_valid.value == 1, f"Timeout for {val}!"
        
        if is_floatml:
            out_sign = int(dut.io_c_stream_payload_0_sign.value)
            out_exp = int(dut.io_c_stream_payload_0_exponent.value)
            out_mant = int(dut.io_c_stream_payload_0_mantissa.value)
            out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
        else:
            out_bits = int(dut.io_c_stream_payload_0.value)
            
        out_val = dtype.to_float(out_bits)
        
        # True Math Error Logging
        true_expected = true_math_fn(val)
        if is_floatml:
            if true_expected != 0:
                error_val = abs((out_val - true_expected) / true_expected) * 100
            else:
                error_val = abs(out_val) * 100
            error_str = f"{error_val:.2f}%"
        else:
            # Full Scale (FS) Error for Integers to avoid "choux vs carottes" massive percentages
            # FS is the maximum positive value of the datatype
            fs_val = (1 << (dtype.bit_width - 1)) - 1
            error_val = abs(out_val - true_expected) / fs_val * 100
            error_str = f"{error_val:.2f}% FS"
            
        edge_str = " (Edge Case)" if edge_cases and val in edge_cases else ""
        log_msg = f"[{op_name}][{dtype_name}] Test x={val} | Error: {error_str}{edge_str} (HW: {out_val}, True Math: {true_expected})"
        dut._log.info(log_msg)
        
        # Write to file only if DEBUG_MATH environment variable is set
        if os.environ.get("DEBUG_MATH") == "1":
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "true_math_errors.log")
            with open(log_path, "a") as f:
                f.write(log_msg + "\n")
        
        # HW Exact Assertion
        expected_bits = expected_bits_fn(val)
        expected_hw_val = dtype.to_float(expected_bits)
        
        bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
        if bit_width > 8 and is_floatml:
            # 1 ULP tolerance for Float PWL chained operations rounding
            assert abs(out_bits - expected_bits) <= 1, f"HW Mismatch for {val}: got {out_val} (bits {out_bits}) instead of {expected_hw_val} (bits {expected_bits})"
        else:
            assert out_bits == expected_bits, f"HW Mismatch for {val}: got {out_val} (bits {out_bits}) instead of {expected_hw_val} (bits {expected_bits})"

async def run_binary_test(dut, op_name, dtype_name, dtype, test_values, is_floatml, expected_bits_fn, true_math_fn, edge_cases=None):
    """
    Generic Cocotb test method for Binary Operators (Add, Sub, Mul, Div).
    test_values is a list of tuples: [(a1, b1), (a2, b2), ...]
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    for val_a, val_b in test_values:
        val_a_bits = dtype.from_float(val_a)
        val_b_bits = dtype.from_float(val_b)
        
        # Injection
        if is_floatml:
            sign_a = (val_a_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
            exp_val_a = (val_a_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
            mant_a = val_a_bits & ((1 << dtype.mant_bits) - 1)
            
            sign_b = (val_b_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
            exp_val_b = (val_b_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
            mant_b = val_b_bits & ((1 << dtype.mant_bits) - 1)
            
            dut.io_a_stream_payload_0_sign.value = sign_a
            dut.io_a_stream_payload_0_exponent.value = exp_val_a
            dut.io_a_stream_payload_0_mantissa.value = mant_a
            dut.io_a_stream_payload_1_sign.value = sign_a
            dut.io_a_stream_payload_1_exponent.value = exp_val_a
            dut.io_a_stream_payload_1_mantissa.value = mant_a
            
            dut.io_b_stream_payload_0_sign.value = sign_b
            dut.io_b_stream_payload_0_exponent.value = exp_val_b
            dut.io_b_stream_payload_0_mantissa.value = mant_b
            dut.io_b_stream_payload_1_sign.value = sign_b
            dut.io_b_stream_payload_1_exponent.value = exp_val_b
            dut.io_b_stream_payload_1_mantissa.value = mant_b
        else:
            dut.io_a_stream_payload_0.value = val_a_bits
            dut.io_a_stream_payload_1.value = val_a_bits
            dut.io_b_stream_payload_0.value = val_b_bits
            dut.io_b_stream_payload_1.value = val_b_bits
            
        dut.io_a_stream_valid.value = 1
        dut.io_b_stream_valid.value = 1
        dut.io_c_stream_ready.value = 1
        
        # Latch input
        await RisingEdge(dut.clk)
        dut.io_a_stream_valid.value = 0
        dut.io_b_stream_valid.value = 0
        
        # Wait for result valid
        cycles = 0
        while dut.io_c_stream_valid.value != 1 and cycles < 30:
            await RisingEdge(dut.clk)
            cycles += 1
            
        assert dut.io_c_stream_valid.value == 1, f"Timeout for ({val_a}, {val_b})!"
        
        if is_floatml:
            out_sign = int(dut.io_c_stream_payload_0_sign.value)
            out_exp = int(dut.io_c_stream_payload_0_exponent.value)
            out_mant = int(dut.io_c_stream_payload_0_mantissa.value)
            out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
        else:
            out_bits = int(dut.io_c_stream_payload_0.value)
            
        out_val = dtype.to_float(out_bits)
        
        # True Math Error Logging
        true_expected = true_math_fn(val_a, val_b)
        if is_floatml:
            if true_expected != 0:
                error_val = abs((out_val - true_expected) / true_expected) * 100
            else:
                error_val = abs(out_val) * 100
            error_str = f"{error_val:.2f}%"
        else:
            fs_val = (1 << (dtype.bit_width - 1)) - 1
            error_val = abs(out_val - true_expected) / fs_val * 100
            error_str = f"{error_val:.2f}% FS"
            
        edge_str = " (Edge Case)" if edge_cases and (val_a, val_b) in edge_cases else ""
        log_msg = f"[{op_name}][{dtype_name}] Test A={val_a}, B={val_b} | Error: {error_str}{edge_str} (HW: {out_val}, True Math: {true_expected})"
        dut._log.info(log_msg)
        
        if os.environ.get("DEBUG_MATH") == "1":
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "true_math_errors.log")
            with open(log_path, "a") as f:
                f.write(log_msg + "\n")
        
        # HW Exact Assertion
        expected_bits = expected_bits_fn(val_a, val_b)
        expected_hw_val = dtype.to_float(expected_bits)
        
        bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
        if bit_width > 8 and is_floatml:
            # 1 ULP tolerance for Float PWL chained operations rounding
            assert abs(out_bits - expected_bits) <= 1, f"HW Mismatch for ({val_a}, {val_b}): got {out_val} (bits {out_bits}) instead of {expected_hw_val} (bits {expected_bits})"
        else:
            assert out_bits == expected_bits, f"HW Mismatch for ({val_a}, {val_b}): got {out_val} (bits {out_bits}) instead of {expected_hw_val} (bits {expected_bits})"

