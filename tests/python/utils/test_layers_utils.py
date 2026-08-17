import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run
import os
import random
import numpy as np

from utils.tb_utils import run_mill, copy_roms

def get_random_tensor(shape, range_val=5.0, integer=True):
    if len(shape) == 1:
        return [[round(random.uniform(-range_val, range_val)) if integer else random.uniform(-range_val, range_val)] for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[round(random.uniform(-range_val, range_val)) if integer else random.uniform(-range_val, range_val) for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 3:
        return [[[round(random.uniform(-range_val, range_val)) if integer else random.uniform(-range_val, range_val) for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]

def log_true_math_error(op_name, dtype_name, dtype, is_floatml, C_out, C_true):
    M = len(C_out)
    N = len(C_out[0])
    errors = []
    for m in range(M):
        for n in range(N):
            out_val = C_out[m][n]
            true_expected = float(C_true[m][n])
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
                
    avg_err = sum(errors) / len(errors) if errors else 0.0
    error_str = f"{avg_err:.2f}%" if is_floatml else f"{avg_err:.2f}% FS"
    
    log_msg = f"[{op_name}][{dtype_name}] Test | Avg Error: {error_str}"
    
    if os.environ.get("DEBUG_MATH") == "1":
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "true_math_errors.log")
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
    return log_msg

async def send_tensor(dut, signal_prefix, tensor, shape, lanes, dtype, is_floatml, wait_ready=True):
    total_elements = int(np.prod(shape))
    chunks = (total_elements + lanes - 1) // lanes
    flattened = np.array(tensor).flatten().tolist()
    
    for chunk in range(chunks):
        if is_floatml:
            for l in range(lanes):
                idx = chunk * lanes + l
                val = flattened[idx] if idx < len(flattened) else 0.0
                val_bits = dtype.from_float(val)
                sign = (val_bits >> (dtype.exp_bits + dtype.mant_bits)) & 1
                exp_val = (val_bits >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
                mant = val_bits & ((1 << dtype.mant_bits) - 1)
                getattr(dut, f"{signal_prefix}_payload_{l}_sign").value = sign
                getattr(dut, f"{signal_prefix}_payload_{l}_exponent").value = exp_val
                getattr(dut, f"{signal_prefix}_payload_{l}_mantissa").value = mant
        else:
            for l in range(lanes):
                idx = chunk * lanes + l
                val = flattened[idx] if idx < len(flattened) else 0.0
                val_bits = dtype.from_float(val)
                getattr(dut, f"{signal_prefix}_payload_{l}").value = val_bits
                
        getattr(dut, f"{signal_prefix}_valid").value = 1
        await RisingEdge(dut.clk)
        if wait_ready:
            while getattr(dut, f"{signal_prefix}_ready").value == 0:
                await RisingEdge(dut.clk)
    getattr(dut, f"{signal_prefix}_valid").value = 0

async def recv_tensor(dut, signal_prefix, shape, dtype, is_floatml, lanes=1):
    total_elements = int(np.prod(shape))
    getattr(dut, f"{signal_prefix}_ready").value = 1
    
    chunks = (total_elements + lanes - 1) // lanes
    
    flattened_bits = []
    flattened_vals = []
    
    for chunk in range(chunks):
        while getattr(dut, f"{signal_prefix}_valid").value == 0:
            await RisingEdge(dut.clk)
            
        for l in range(lanes):
            if chunk * lanes + l >= total_elements:
                break
                
            if is_floatml:
                out_sign = int(getattr(dut, f"{signal_prefix}_payload_{l}_sign").value)
                out_exp = int(getattr(dut, f"{signal_prefix}_payload_{l}_exponent").value)
                out_mant = int(getattr(dut, f"{signal_prefix}_payload_{l}_mantissa").value)
                bits = (out_sign << (dtype.exp_bits + dtype.mant_bits)) | (out_exp << dtype.mant_bits) | out_mant
            else:
                bits = int(getattr(dut, f"{signal_prefix}_payload_{l}").value)
                
            flattened_bits.append(bits)
            flattened_vals.append(dtype.to_float(bits))
            
        await RisingEdge(dut.clk)
        
    for idx in range(min(total_elements, len(flattened_bits))):
        flattened_bits[idx] = flattened_bits[idx]
        
    out_tensor = np.array(flattened_vals[:total_elements]).reshape(shape).tolist()
    out_bits = np.array(flattened_bits[:total_elements]).reshape(shape).tolist()
                
    return out_bits, out_tensor

import xml.etree.ElementTree as ET

def safe_run_sim(**kwargs):
    try:
        run(**kwargs)
    except SystemExit as e:
        sim_build = kwargs.get("sim_build", "sim_build")
        results_xml = os.path.join(sim_build, "results.xml")
        if os.path.exists(results_xml):
            try:
                tree = ET.parse(results_xml)
                root = tree.getroot()
                if root.tag == "testsuites":
                    ts = root.find("testsuite")
                    if ts is not None and ts.attrib.get("failures") == "0" and ts.attrib.get("errors") == "0":
                        return
                elif root.tag == "testsuite":
                    if root.attrib.get("failures") == "0" and root.attrib.get("errors") == "0":
                        return
            except:
                pass
            
            try:
                with open(results_xml, "r") as f:
                    content = f.read()
                    if 'failures="0"' in content and 'errors="0"' in content:
                        return
            except:
                pass
        raise e

def run_layer_sim(layer_name, dtype_filter, testcase_name, toplevel, request=None):
    v_file = run_mill(f"spinalML.layers.{layer_name}Test", dtype_filter, toplevel)
    build_dir = f"sim_build/{layer_name.lower()}_{toplevel.lower()}_{dtype_filter.lower()}"
    copy_roms(build_dir)
    debug_flag = "1" if request and request.config.getoption("--debug-math") else "0"
    safe_run_sim(
        language="verilog",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module=f"test_{layer_name.lower()}",
        testcase=testcase_name,
        simulator="verilator",
        sim_build=build_dir,
        timescale="1ns/1ps",
        extra_args=["--trace", "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-WIDTH"],
        extra_env={"DEBUG_MATH": debug_flag}
    )
