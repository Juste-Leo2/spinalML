import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3, I16, BF16, I32
from golden_models.ops import conv1d_hw
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, run_layer_sim

# ==========================================
# CONV1D LAYER
# ==========================================
async def run_conv1d_test(dut, op_name, dtype_name, dtype, X, W, b, is_floatml, W_shape, W_lanes, X_shape, X_lanes, Y_shape, Y_lanes):
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
    # Send b (lanes is always 1 for bias)
    b_shape = (1, len(b[0]))
    await send_tensor(dut, "io_b_stream", b, b_shape, 1, dtype, is_floatml)
    
    send_x = cocotb.start_soon(send_tensor(dut, "io_x_stream", X, X_shape, X_lanes, dtype, is_floatml))
    acc_dtype = dtype if is_floatml else I32
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_y_stream", Y_shape, acc_dtype, is_floatml, Y_lanes))
    
    Y_out_bits, Y_out = await recv_y
    await send_x
    
    # Exact HW Math
    Y_expected = conv1d_hw(X, W, b, dtype)
    
    # True Pure Math (for logging true error)
    import numpy as np
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
    
    log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, Y_out, Y_true)
    dut._log.info(log_msg)
    
    # Exact HW Math
    Y_expected = conv1d_hw(X, W, b, dtype)
    
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
    X = get_random_tensor((3, 1), 10.0, True)
    W = get_random_tensor((2, 1), 10.0, True)
    b = get_random_tensor((1, 1), 10.0, True)
    from golden_models.dtypes import I8
    await run_conv1d_test(dut, "Conv1D", "I8", I8, X, W, b, False, (2, 1), 2, (3, 1), 1, (2, 1), 1)
    
@cocotb.test()
async def cocotb_conv1d_fp8(dut):
    X = get_random_tensor((3, 1), 5.0, False)
    W = get_random_tensor((2, 1), 5.0, False)
    b = get_random_tensor((1, 1), 5.0, False)
    from golden_models.dtypes import FP8_E4M3
    await run_conv1d_test(dut, "Conv1D", "FP8", FP8_E4M3, X, W, b, True, (2, 1), 2, (3, 1), 1, (2, 1), 1)

@cocotb.test()
async def cocotb_conv1d_i16(dut):
    X = get_random_tensor((3, 1), 100.0, True)
    W = get_random_tensor((2, 1), 10.0, True)
    b = get_random_tensor((1, 1), 100.0, True)
    await run_conv1d_test(dut, "Conv1D", "I16", I16, X, W, b, False, (2, 1), 2, (3, 1), 1, (2, 1), 1)

@cocotb.test()
async def cocotb_conv1d_bf16(dut):
    X = get_random_tensor((3, 1), 10.0, False)
    W = get_random_tensor((2, 1), 5.0, False)
    b = get_random_tensor((1, 1), 5.0, False)
    await run_conv1d_test(dut, "Conv1D", "BF16", BF16, X, W, b, True, (2, 1), 2, (3, 1), 1, (2, 1), 1)

# Multi-channel tests
@cocotb.test()
async def cocotb_conv1dmulti_i8(dut):
    X = get_random_tensor((3, 2), 10.0, True)
    W = get_random_tensor((4, 2), 10.0, True) # K=2, C_in=2 -> flattened K*C_in=4
    b = get_random_tensor((1, 2), 10.0, True)
    from golden_models.dtypes import I8
    await run_conv1d_test(dut, "Conv1DMulti", "I8", I8, X, W, b, False, (4, 2), 4, (3, 2), 1, (2, 2), 1)

@cocotb.test()
async def cocotb_conv1dmulti_fp8(dut):
    X = get_random_tensor((3, 2), 5.0, False)
    W = get_random_tensor((4, 2), 5.0, False)
    b = get_random_tensor((1, 2), 5.0, False)
    from golden_models.dtypes import FP8_E4M3
    await run_conv1d_test(dut, "Conv1DMulti", "FP8", FP8_E4M3, X, W, b, True, (4, 2), 4, (3, 2), 1, (2, 2), 1)

@cocotb.test()
async def cocotb_conv1dmulti_i16(dut):
    X = get_random_tensor((3, 2), 100.0, True)
    W = get_random_tensor((4, 2), 10.0, True)
    b = get_random_tensor((1, 2), 100.0, True)
    await run_conv1d_test(dut, "Conv1DMulti", "I16", I16, X, W, b, False, (4, 2), 4, (3, 2), 1, (2, 2), 1)

@cocotb.test()
async def cocotb_conv1dmulti_bf16(dut):
    X = get_random_tensor((3, 2), 10.0, False)
    W = get_random_tensor((4, 2), 5.0, False)
    b = get_random_tensor((1, 2), 5.0, False)
    await run_conv1d_test(dut, "Conv1DMulti", "BF16", BF16, X, W, b, True, (4, 2), 4, (3, 2), 1, (2, 2), 1)

def test_pytest_conv1d_i8(request): run_layer_sim("Conv1D", "I8", "cocotb_conv1d_i8", "Conv1DTestComp", request)
def test_pytest_conv1d_fp8(request): run_layer_sim("Conv1D", "FP8", "cocotb_conv1d_fp8", "Conv1DTestComp", request)
def test_pytest_conv1d_i16(request): run_layer_sim("Conv1D", "I16", "cocotb_conv1d_i16", "Conv1DTestComp", request)
def test_pytest_conv1d_bf16(request): run_layer_sim("Conv1D", "BF16", "cocotb_conv1d_bf16", "Conv1DTestComp", request)

def test_pytest_conv1dmulti_i8(request): run_layer_sim("Conv1D", "I8", "cocotb_conv1dmulti_i8", "Conv1DTestCompMulti", request)
def test_pytest_conv1dmulti_fp8(request): run_layer_sim("Conv1D", "FP8", "cocotb_conv1dmulti_fp8", "Conv1DTestCompMulti", request)
def test_pytest_conv1dmulti_i16(request): run_layer_sim("Conv1D", "I16", "cocotb_conv1dmulti_i16", "Conv1DTestCompMulti", request)
def test_pytest_conv1dmulti_bf16(request): run_layer_sim("Conv1D", "BF16", "cocotb_conv1dmulti_bf16", "Conv1DTestCompMulti", request)
