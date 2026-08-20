import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np
import os

from golden_models.dtypes import I8, FP8_E4M3, I32
from golden_models.ops import linear_hw
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, run_layer_sim, DEFAULT_NUM_TRIALS
from utils.tb_utils import seed_random, SEED

seed_random()

# ==========================================
# LINEAR LAYER
# ==========================================
async def run_linear_test(dut, op_name, dtype_name, dtype, A, W, b, is_floatml, A_shape=(1, 2), A_lanes=2, W_shape=(2, 1), W_lanes=2, b_shape=(1, 1), b_lanes=1, Y_shape=(1, 1), Y_lanes=1, collect=None):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_a_stream_valid.value = 0
    dut.io_w_stream_valid.value = 0
    dut.io_b_stream_valid.value = 0
    dut.io_y_stream_ready.value = 0
    
    # Send W (transposed because Linear/Matmul expects column-major weight streaming)
    W_T = np.array(W).T.tolist()
    await send_tensor(dut, "io_w_stream", W_T, (W_shape[1], W_shape[0]), W_lanes, dtype, is_floatml)
    
    acc_dtype = dtype if is_floatml else I32
    b_shape = (1, len(b[0]))
    await send_tensor(dut, "io_b_stream", b, b_shape, b_lanes, acc_dtype, is_floatml)
    
    send_a = cocotb.start_soon(send_tensor(dut, "io_a_stream", A, A_shape, A_lanes, dtype, is_floatml))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_y_stream", Y_shape, acc_dtype, is_floatml, Y_lanes))
    
    Y_out_bits, Y_out = await recv_y
    await send_a
    
    # True Math
    A_np = np.array(A)
    W_np = np.array(W)
    b_np = np.array(b)
    Y_true = np.matmul(A_np, W_np) + b_np
    
    if collect is not None:
        collect["out"].append(Y_out)
        collect["true"].append(Y_true.tolist())
    else:
        log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, Y_out, Y_true.tolist())
        dut._log.info(log_msg)
    
    # Exact HW Math
    Y_expected = linear_hw(A, W, b, dtype)
    
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    for m in range(Y_shape[0]):
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
async def cocotb_linear_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        A = get_random_tensor((1, 2), 10.0, True)
        W = get_random_tensor((2, 1), 10.0, True)
        b = get_random_tensor((1, 1), 10.0, True)
        await run_linear_test(dut, "Linear", "I8", I8, A, W, b, False, collect=collect)
    details = f"A=1x2, W=2x1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Linear", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_linear_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        A = get_random_tensor((1, 2), 5.0, False)
        W = get_random_tensor((2, 1), 5.0, False)
        b = get_random_tensor((1, 1), 5.0, False)
        await run_linear_test(dut, "Linear", "FP8", FP8_E4M3, A, W, b, True, collect=collect)
    details = f"A=1x2, W=2x1, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("Linear", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_linearmulti_i8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        A = get_random_tensor((2, 3), 10.0, True)
        W = get_random_tensor((3, 4), 10.0, True)
        b = get_random_tensor((1, 4), 10.0, True)
        await run_linear_test(dut, "LinearMulti", "I8", I8, A, W, b, False, (2,3), 3, (3,4), 3, (1,4), 1, (2,4), 1, collect=collect)
    details = f"A=2x3, W=3x4, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LinearMulti", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_linearmulti_fp8(dut):
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        A = get_random_tensor((2, 3), 5.0, False)
        W = get_random_tensor((3, 4), 5.0, False)
        b = get_random_tensor((1, 4), 5.0, False)
        await run_linear_test(dut, "LinearMulti", "FP8", FP8_E4M3, A, W, b, True, (2,3), 3, (3,4), 3, (1,4), 1, (2,4), 1, collect=collect)
    details = f"A=2x3, W=3x4, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("LinearMulti", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_linear_i8(request): run_layer_sim("Linear", "I8", "cocotb_linear_i8", "LinearTestComp", request)
def test_pytest_linear_fp8(request): run_layer_sim("Linear", "FP8", "cocotb_linear_fp8", "LinearTestComp", request)
def test_pytest_linearmulti_i8(request): run_layer_sim("Linear", "I8", "cocotb_linearmulti_i8", "LinearTestCompMulti", request)
def test_pytest_linearmulti_fp8(request): run_layer_sim("Linear", "FP8", "cocotb_linearmulti_fp8", "LinearTestCompMulti", request)