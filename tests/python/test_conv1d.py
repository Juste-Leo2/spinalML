import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

from golden_models.dtypes import I16, BF16
from golden_models.ops import conv1d_hw
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, run_layer_sim

# ==========================================
# CONV1D LAYER
# ==========================================
async def run_conv1d_test(dut, op_name, dtype_name, dtype, X, W, b, is_floatml):
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
    
    # L_in = 3, K = 2, L_out = 2
    # Send W
    await send_tensor(dut, "io_w_stream", W, (2, 1), 2, dtype, is_floatml)
    # Send b
    await send_tensor(dut, "io_b_stream", b, (1, 1), 1, dtype, is_floatml)
    
    send_x = cocotb.start_soon(send_tensor(dut, "io_x_stream", X, (3, 1), 1, dtype, is_floatml))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_y_stream", (2, 1), dtype, is_floatml))
    
    Y_out_bits, Y_out = await recv_y
    await send_x
    
    # True Math
    X_np = np.array([x[0] for x in X])
    W_np = np.array([w[0] for w in W])
    Y_true = [[float(np.dot(X_np[i:i+2], W_np) + b[0][0])] for i in range(2)]
    
    log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, Y_out, Y_true)
    dut._log.info(log_msg)
    
    # Exact HW Math
    Y_expected = conv1d_hw(X, W, b, dtype)
    
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    for m in range(2):
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
async def cocotb_conv1d_i16(dut):
    X = get_random_tensor((3, 1), 100.0, True)
    W = get_random_tensor((2, 1), 10.0, True)
    b = get_random_tensor((1, 1), 100.0, True)
    await run_conv1d_test(dut, "Conv1D", "I16", I16, X, W, b, False)

@cocotb.test()
async def cocotb_conv1d_bf16(dut):
    X = get_random_tensor((3, 1), 10.0, False)
    W = get_random_tensor((2, 1), 5.0, False)
    b = get_random_tensor((1, 1), 5.0, False)
    await run_conv1d_test(dut, "Conv1D", "BF16", BF16, X, W, b, True)

def test_pytest_conv1d_i16(request): run_layer_sim("Conv1D", "I16", "cocotb_conv1d_i16", "Conv1DTestComp", request)
def test_pytest_conv1d_bf16(request): run_layer_sim("Conv1D", "BF16", "cocotb_conv1d_bf16", "Conv1DTestComp", request)
