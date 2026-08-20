import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np
import os

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16, I32
from golden_models.ops import conv1d_hw
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, run_layer_sim, DEFAULT_NUM_TRIALS
from utils.tb_utils import seed_random, SEED

seed_random()

# ==========================================
# CONV1D LAYER
# ==========================================
async def run_conv1d_test(dut, op_name, dtype_name, dtype, X, W, b, is_floatml, W_shape, W_lanes, X_shape, X_lanes, Y_shape, Y_lanes, collect=None):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_x_stream_valid.value = 0
    dut.io_w_stream_valid.value = 0
    dut.io_b_stream_valid.value = 0
    dut.io_y_stream_ready.value = 0
    
    # Hardware expects W to be streamed column-major
    W_T = [[W[i][j] for i in range(len(W))] for j in range(len(W[0]))]
    
    # Send W
    await send_tensor(dut, "io_w_stream", W_T, (W_shape[1], W_shape[0]), W_lanes, dtype, is_floatml)
    
    acc_dtype = dtype if is_floatml else I32
    # Send b (lanes is always 1 for bias)
    b_shape = (1, len(b[0]))
    await send_tensor(dut, "io_b_stream", b, b_shape, 1, acc_dtype, is_floatml)
    
    send_x = cocotb.start_soon(send_tensor(dut, "io_x_stream", X, X_shape, X_lanes, dtype, is_floatml))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_y_stream", Y_shape, acc_dtype, is_floatml, Y_lanes))
    
    Y_out_bits, Y_out = await recv_y
    await send_x
    
    # Exact HW Math
    Y_expected = conv1d_hw(X, W, b, dtype)
    
    # True Pure Math (for logging true error)
    L_in = len(X)
    C_in = len(X[0]) if isinstance(X[0], list) else 1
    K = len(W) // C_in
    L_out = L_in - K + 1
    
    X_np = np.array(X)
    W_np = np.array(W)
    b_np = np.array(b[0])
    
    Y_true = []
    for i in range(L_out):
        window = X_np[i:i+K].flatten()
        res = np.dot(window, W_np) + b_np
        Y_true.append(res.tolist())
    
    if collect is not None:
        collect["out"].append(Y_out)
        collect["true"].append(Y_true)
    else:
        log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, Y_out, Y_true)
        dut._log.info(log_msg)
    
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    for m in range(2):
        for n in range(Y_shape[1]):
            exp_val = Y_expected[m][n]
            exp_bits = acc_dtype.from_float(exp_val)
            out_bits = Y_out_bits[m][n]
            out_val = Y_out[m][n]
            
            if bit_width > 8 and is_floatml:
                assert abs(out_bits - exp_bits) <= 1, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {acc_dtype.to_float(exp_bits)}"
            else:
                assert out_bits == exp_bits, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {acc_dtype.to_float(exp_bits)}"

@cocotb.test()
async def cocotb_conv1d_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((3, 1), 10.0, True)
        W = get_random_tensor((2, 1), 10.0, True)
        b = get_random_tensor((1, 1), 10.0, True)
        await run_conv1d_test(dut, "Conv1D", "I8", I8, X, W, b, False, (2, 1), 2, (3, 1), 1, (2, 1), 1, collect=collect)
    details = f"X=3x1, W=2x1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Conv1D", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)
    
@cocotb.test()
async def cocotb_conv1d_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((3, 1), 5.0, False)
        W = get_random_tensor((2, 1), 5.0, False)
        b = get_random_tensor((1, 1), 5.0, False)
        await run_conv1d_test(dut, "Conv1D", "FP8", FP8_E4M3, X, W, b, True, (2, 1), 2, (3, 1), 1, (2, 1), 1, collect=collect)
    details = f"X=3x1, W=2x1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Conv1D", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_conv1d_i16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((3, 1), 100.0, True)
        W = get_random_tensor((2, 1), 10.0, True)
        b = get_random_tensor((1, 1), 100.0, True)
        await run_conv1d_test(dut, "Conv1D", "I16", I16, X, W, b, False, (2, 1), 2, (3, 1), 1, (2, 1), 1, collect=collect)
    details = f"X=3x1, W=2x1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Conv1D", "I16", I16, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_conv1d_bf16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((3, 1), 10.0, False)
        W = get_random_tensor((2, 1), 5.0, False)
        b = get_random_tensor((1, 1), 5.0, False)
        await run_conv1d_test(dut, "Conv1D", "BF16", BF16, X, W, b, True, (2, 1), 2, (3, 1), 1, (2, 1), 1, collect=collect)
    details = f"X=3x1, W=2x1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Conv1D", "BF16", BF16, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

# Multi-channel tests
@cocotb.test()
async def cocotb_conv1dmulti_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((3, 2), 10.0, True)
        W = get_random_tensor((4, 2), 10.0, True) # K=2, C_in=2 -> flattened K*C_in=4
        b = get_random_tensor((1, 2), 10.0, True)
        await run_conv1d_test(dut, "Conv1DMulti", "I8", I8, X, W, b, False, (4, 2), 4, (3, 2), 1, (2, 2), 1, collect=collect)
    details = f"X=3x2, W=4x2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Conv1DMulti", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_conv1dmulti_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((3, 2), 5.0, False)
        W = get_random_tensor((4, 2), 5.0, False)
        b = get_random_tensor((1, 2), 5.0, False)
        await run_conv1d_test(dut, "Conv1DMulti", "FP8", FP8_E4M3, X, W, b, True, (4, 2), 4, (3, 2), 1, (2, 2), 1, collect=collect)
    details = f"X=3x2, W=4x2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Conv1DMulti", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_conv1dmulti_i16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((3, 2), 100.0, True)
        W = get_random_tensor((4, 2), 10.0, True)
        b = get_random_tensor((1, 2), 100.0, True)
        await run_conv1d_test(dut, "Conv1DMulti", "I16", I16, X, W, b, False, (4, 2), 4, (3, 2), 1, (2, 2), 1, collect=collect)
    details = f"X=3x2, W=4x2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Conv1DMulti", "I16", I16, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_conv1dmulti_bf16(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((3, 2), 10.0, False)
        W = get_random_tensor((4, 2), 5.0, False)
        b = get_random_tensor((1, 2), 5.0, False)
        await run_conv1d_test(dut, "Conv1DMulti", "BF16", BF16, X, W, b, True, (4, 2), 4, (3, 2), 1, (2, 2), 1, collect=collect)
    details = f"X=3x2, W=4x2, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Conv1DMulti", "BF16", BF16, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_conv1d_i8(request): run_layer_sim("Conv1D", "I8", "cocotb_conv1d_i8", "Conv1DTestComp", request)
def test_pytest_conv1d_fp8(request): run_layer_sim("Conv1D", "FP8", "cocotb_conv1d_fp8", "Conv1DTestComp", request)
def test_pytest_conv1d_i16(request): run_layer_sim("Conv1D", "I16", "cocotb_conv1d_i16", "Conv1DTestComp", request)
def test_pytest_conv1d_bf16(request): run_layer_sim("Conv1D", "BF16", "cocotb_conv1d_bf16", "Conv1DTestComp", request)

def test_pytest_conv1dmulti_i8(request): run_layer_sim("Conv1D", "I8", "cocotb_conv1dmulti_i8", "Conv1DTestCompMulti", request)
def test_pytest_conv1dmulti_fp8(request): run_layer_sim("Conv1D", "FP8", "cocotb_conv1dmulti_fp8", "Conv1DTestCompMulti", request)
def test_pytest_conv1dmulti_i16(request): run_layer_sim("Conv1D", "I16", "cocotb_conv1dmulti_i16", "Conv1DTestCompMulti", request)
def test_pytest_conv1dmulti_bf16(request): run_layer_sim("Conv1D", "BF16", "cocotb_conv1dmulti_bf16", "Conv1DTestCompMulti", request)