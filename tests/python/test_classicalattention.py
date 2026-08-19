import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np
import pytest

from golden_models.dtypes import FP8_E4M3, I8, I16, BF16
from golden_models.ops import classical_attention_hw
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, run_layer_sim

async def run_attention_test(dut, op_name, dtype_name, dtype, X, Wq, Wk, Wv, Wo, is_floatml, seqLen, embedDim, xLanes, wLanes):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_x_stream_valid.value = 0
    dut.io_wq_stream_valid.value = 0
    dut.io_wk_stream_valid.value = 0
    dut.io_wv_stream_valid.value = 0
    dut.io_wo_stream_valid.value = 0
    dut.io_y_stream_ready.value = 0
    
    # Transpose weights for Matmul
    Wq_T = np.array(Wq).T.tolist()
    Wk_T = np.array(Wk).T.tolist()
    Wv_T = np.array(Wv).T.tolist()
    Wo_T = np.array(Wo).T.tolist()
    
    await send_tensor(dut, "io_wq_stream", Wq_T, (embedDim, embedDim), wLanes, dtype, is_floatml)
    await send_tensor(dut, "io_wk_stream", Wk_T, (embedDim, embedDim), wLanes, dtype, is_floatml)
    await send_tensor(dut, "io_wv_stream", Wv_T, (embedDim, embedDim), wLanes, dtype, is_floatml)
    await send_tensor(dut, "io_wo_stream", Wo_T, (embedDim, embedDim), wLanes, dtype, is_floatml)
    
    send_x = cocotb.start_soon(send_tensor(dut, "io_x_stream", X, (seqLen, embedDim), xLanes, dtype, is_floatml))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_y_stream", (seqLen, embedDim), dtype, is_floatml, 1))
    
    Y_out_bits, Y_out = await recv_y
    await send_x
    
    # Hardware exact math
    Y_expected = classical_attention_hw(X, Wq, Wk, Wv, Wo, dtype)
    
    log_msg = log_true_math_error("ClassicalAttention", dtype_name, dtype, is_floatml, Y_out, Y_expected)
    print(log_msg)
    
    # We allow a relaxed margin for Softmax PWL approximations
    for m in range(seqLen):
        for n in range(embedDim):
            out_bits = Y_out_bits[m][n]
            out_val = Y_out[m][n]
            exp_val = float(Y_expected[m][n])
            
            if is_floatml:
                err = abs(out_val - exp_val)
                # PWL introduces approximations up to 10-15% in extreme cases, allow ~0.25 diff for small values
                assert err < 0.25, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {exp_val}"
            else:
                fs_val = (1 << (getattr(dtype, 'bit_width', 8) - 1)) - 1
                err = abs(out_val - exp_val) / fs_val
                assert err < 0.15, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {exp_val}"


@cocotb.test()
async def cocotb_attention_fp8(dut):
    # Dimensions
    seqLen = 2
    embedDim = 2
    xLanes = 2
    wLanes = 2
    
    # Attention requires scale to not explode the Exp LUT. We use very small random weights.
    X = get_random_tensor((seqLen, embedDim), 1.0, False)
    Wq = get_random_tensor((embedDim, embedDim), 0.5, False)
    Wk = get_random_tensor((embedDim, embedDim), 0.5, False)
    Wv = get_random_tensor((embedDim, embedDim), 0.5, False)
    Wo = get_random_tensor((embedDim, embedDim), 0.5, False)
    
    await run_attention_test(dut, "ClassicalAttention", "FP8", FP8_E4M3, X, Wq, Wk, Wv, Wo, True, seqLen, embedDim, xLanes, wLanes)

def test_pytest_attention_fp8(request): run_layer_sim("ClassicalAttention", "FP8", "cocotb_attention_fp8", "AttentionTestComp", request)

@cocotb.test()
async def cocotb_attention_i8(dut):
    seqLen = 2
    embedDim = 2
    xLanes = 2
    wLanes = 2
    
    X = get_random_tensor((seqLen, embedDim), 1.0, True)
    Wq = get_random_tensor((embedDim, embedDim), 1.0, True)
    Wk = get_random_tensor((embedDim, embedDim), 1.0, True)
    Wv = get_random_tensor((embedDim, embedDim), 1.0, True)
    Wo = get_random_tensor((embedDim, embedDim), 1.0, True)
    
    await run_attention_test(dut, "ClassicalAttention", "I8", I8, X, Wq, Wk, Wv, Wo, False, seqLen, embedDim, xLanes, wLanes)

def test_pytest_attention_i8(request): run_layer_sim("ClassicalAttention", "I8", "cocotb_attention_i8", "AttentionTestComp", request)

@cocotb.test()
async def cocotb_attention_i16(dut):
    seqLen = 2
    embedDim = 2
    xLanes = 2
    wLanes = 2
    
    X = get_random_tensor((seqLen, embedDim), 3.0, True)
    Wq = get_random_tensor((embedDim, embedDim), 3.0, True)
    Wk = get_random_tensor((embedDim, embedDim), 3.0, True)
    Wv = get_random_tensor((embedDim, embedDim), 3.0, True)
    Wo = get_random_tensor((embedDim, embedDim), 3.0, True)
    
    await run_attention_test(dut, "ClassicalAttention", "I16", I16, X, Wq, Wk, Wv, Wo, False, seqLen, embedDim, xLanes, wLanes)

def test_pytest_attention_i16(request): run_layer_sim("ClassicalAttention", "I16", "cocotb_attention_i16", "AttentionTestComp", request)

@cocotb.test()
async def cocotb_attention_bf16(dut):
    seqLen = 2
    embedDim = 2
    xLanes = 2
    wLanes = 2
    
    X = get_random_tensor((seqLen, embedDim), 1.0, False)
    Wq = get_random_tensor((embedDim, embedDim), 1.0, False)
    Wk = get_random_tensor((embedDim, embedDim), 1.0, False)
    Wv = get_random_tensor((embedDim, embedDim), 1.0, False)
    Wo = get_random_tensor((embedDim, embedDim), 1.0, False)
    
    await run_attention_test(dut, "ClassicalAttention", "BF16", BF16, X, Wq, Wk, Wv, Wo, True, seqLen, embedDim, xLanes, wLanes)

def test_pytest_attention_bf16(request): run_layer_sim("ClassicalAttention", "BF16", "cocotb_attention_bf16", "AttentionTestComp", request)
