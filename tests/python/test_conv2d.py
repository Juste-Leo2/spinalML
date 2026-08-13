import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

from golden_models.dtypes import I8, FP8_E4M3
from golden_models.ops import conv2d_hw
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, run_layer_sim

# ==========================================
# CONV2D LAYER
# ==========================================
async def run_conv2d_test(dut, op_name, dtype_name, dtype, X, W, b, is_floatml):
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
    
    # H=3, W=3, K=2, Output=4 (Flattened)
    await send_tensor(dut, "io_w_stream", W, (4, 1), 4, dtype, is_floatml)
    await send_tensor(dut, "io_b_stream", b, (1, 1), 1, dtype, is_floatml)
    
    send_x = cocotb.start_soon(send_tensor(dut, "io_x_stream", X, (3, 3), 1, dtype, is_floatml))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_y_stream", (4, 1), dtype, is_floatml))
    
    Y_out_bits, Y_out = await recv_y
    await send_x
    
    # True Math
    X_np = np.array(X)
    W_np = np.array([w[0] for w in W]).reshape(2, 2)
    Y_true = []
    for i in range(2):
        for j in range(2):
            window = X_np[i:i+2, j:j+2]
            Y_true.append([float(np.sum(window * W_np) + b[0][0])])
            
    log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, Y_out, Y_true)
    dut._log.info(log_msg)
    
    # Exact HW Math
    Y_expected = conv2d_hw(X, W, b, dtype)
    
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    for m in range(4):
        for n in range(1):
            exp_val = Y_expected[m][n]
            exp_bits = dtype.from_float(exp_val)
            out_bits = Y_out_bits[m][n]
            out_val = Y_out[m][n]
            if bit_width > 8 and is_floatml:
                assert abs(out_bits - exp_bits) <= 1, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {dtype.to_float(exp_bits)}"
            else:
                assert out_bits == exp_bits, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {dtype.to_float(exp_bits)}"

@cocotb.test()
async def cocotb_conv2d_i8(dut):
    X = get_random_tensor((3, 3), 10.0, True)
    W = get_random_tensor((4, 1), 10.0, True)
    b = get_random_tensor((1, 1), 10.0, True)
    await run_conv2d_test(dut, "Conv2D", "I8", I8, X, W, b, False)
    
@cocotb.test()
async def cocotb_conv2d_fp8(dut):
    X = get_random_tensor((3, 3), 5.0, False)
    W = get_random_tensor((4, 1), 5.0, False)
    b = get_random_tensor((1, 1), 5.0, False)
    await run_conv2d_test(dut, "Conv2D", "FP8", FP8_E4M3, X, W, b, True)

def test_pytest_conv2d_i8(request): run_layer_sim("Conv2D", "I8", "cocotb_conv2d_i8", "Conv2DTestComp", request)
def test_pytest_conv2d_fp8(request): run_layer_sim("Conv2D", "FP8", "cocotb_conv2d_fp8", "Conv2DTestComp", request)
