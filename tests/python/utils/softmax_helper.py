async def run_softmax_test(dut, op_name, dtype_name, dtype, test_values, is_floatml, expected_bits_fn, true_math_fn, edge_cases=None):
    """
    Generic Cocotb test method for Softmax1D (lanes=4).
    test_values is a list of 4-element tuples/lists: [(x0, x1, x2, x3), ...]
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    for val_arr in test_values:
        val_bits_arr = [dtype.from_float(v) for v in val_arr]
        
        # Injection
        if is_floatml:
            for i, val_bits in enumerate(val_bits_arr):
                sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                mant = val_bits & ((1 << dtype.mant_bits) - 1)
                getattr(dut, f"io_x_stream_payload_{i}_sign").value = sign
                getattr(dut, f"io_x_stream_payload_{i}_exponent").value = exp_val
                getattr(dut, f"io_x_stream_payload_{i}_mantissa").value = mant
        else:
            for i, val_bits in enumerate(val_bits_arr):
                getattr(dut, f"io_x_stream_payload_{i}").value = val_bits
                
        dut.io_x_stream_valid.value = 1
        dut.io_y_stream_ready.value = 1
        
        # Latch input
        await RisingEdge(dut.clk)
        dut.io_x_stream_valid.value = 0
        
        # Wait for result valid
        cycles = 0
        while dut.io_y_stream_valid.value != 1 and cycles < 50:
            await RisingEdge(dut.clk)
            cycles += 1
            
        assert dut.io_y_stream_valid.value == 1, f"Timeout for {val_arr}!"
        
        out_bits_arr = []
        out_val_arr = []
        for i in range(4):
            if is_floatml:
                out_sign = int(getattr(dut, f"io_y_stream_payload_{i}_sign").value)
                out_exp = int(getattr(dut, f"io_y_stream_payload_{i}_exponent").value)
                out_mant = int(getattr(dut, f"io_y_stream_payload_{i}_mantissa").value)
                out_bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
            else:
                out_bits = int(getattr(dut, f"io_y_stream_payload_{i}").value)
            out_bits_arr.append(out_bits)
            out_val_arr.append(dtype.to_float(out_bits))
            
        # True Math Error Logging
        true_expected_arr = true_math_fn(val_arr)
        
        errors = []
        for i in range(4):
            out_val = out_val_arr[i]
            true_expected = true_expected_arr[i]
            if is_floatml:
                if true_expected != 0:
                    err = abs((out_val - true_expected) / true_expected) * 100
                else:
                    err = abs(out_val) * 100
                errors.append(err)
            else:
                fs_val = (1 << (dtype.bit_width - 1)) - 1
                err = abs(out_val - true_expected) / fs_val * 100
                errors.append(err)
        
        avg_err = sum(errors) / len(errors)
        error_str = f"{avg_err:.2f}%" if is_floatml else f"{avg_err:.2f}% FS"
        
        edge_str = " (Edge Case)" if edge_cases and val_arr in edge_cases else ""
        log_msg = f"[{op_name}][{dtype_name}] Test X={val_arr} | Avg Error: {error_str}{edge_str} (HW: {out_val_arr}, True: {true_expected_arr})"
        dut._log.info(log_msg)
        
        if os.environ.get("DEBUG_MATH") == "1":
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "true_math_errors.log")
            with open(log_path, "a") as f:
                f.write(log_msg + "\n")
        
        # HW Exact Assertion
        expected_bits_arr = expected_bits_fn(val_arr)
        
        bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
        for i in range(4):
            out_bits = out_bits_arr[i]
            expected_bits = expected_bits_arr[i]
            expected_hw_val = dtype.to_float(expected_bits)
            if bit_width > 8 and is_floatml:
                assert abs(out_bits - expected_bits) <= 1, f"HW Mismatch for {val_arr} at index {i}: got {out_val_arr[i]} (bits {out_bits}) instead of {expected_hw_val} (bits {expected_bits})"
            else:
                assert out_bits == expected_bits, f"HW Mismatch for {val_arr} at index {i}: got {out_val_arr[i]} (bits {out_bits}) instead of {expected_hw_val} (bits {expected_bits})"

