import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np
import pytest
import os

from golden_models.dtypes import FP8_E4M3, I8, I16, BF16
from golden_models.ops import multi_head_attention_hw
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, run_layer_sim, DEFAULT_NUM_TRIALS
from utils.tb_utils import seed_random, SEED

seed_random()

async def run_multihead_test(dut, op_name, dtype_name, dtype, X, Wq, Wk, Wv, Wo, is_floatml, seqLen, embedDim, numHeads, xLanes, wLanes, collect=None):
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
    Y_expected = multi_head_attention_hw(X, Wq, Wk, Wv, Wo, dtype, numHeads)
    
    if collect is not None:
        collect["out"].append(Y_out)
        collect["true"].append(Y_expected)
    else:
        log_msg = log_true_math_error(op_name, dtype_name, dtype, is_floatml, collect["out"] if collect else Y_out, collect["true"] if collect else Y_expected)
        dut._log.info(log_msg)
    
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
async def cocotb_multihead_fp8(dut):
    # Dimensions
    seqLen = 4
    embedDim = 4
    numHeads = 2
    xLanes = 4
    wLanes = 4
    
    # Attention requires scale to not explode the Exp LUT. We use very small random weights.
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((seqLen, embedDim), 1.0, False)
        Wq = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wk = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wv = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wo = get_random_tensor((embedDim, embedDim), 0.5, False)
        await run_multihead_test(dut, "MultiHeadAttention", "FP8", FP8_E4M3, X, Wq, Wk, Wv, Wo, True, seqLen, embedDim, numHeads, xLanes, wLanes, collect=collect)
    
    details = f"seq={seqLen}, emb={embedDim}, heads={numHeads}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MultiHeadAttention", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_multihead_fp8(request): run_layer_sim("MultiHeadAttention", "FP8", "cocotb_multihead_fp8", "MultiHeadAttentionTestComp", request)

@cocotb.test()
async def cocotb_multihead_i8(dut):
    seqLen = 4
    embedDim = 4
    numHeads = 2
    xLanes = 4
    wLanes = 4
    
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((seqLen, embedDim), 1.0, True)
        Wq = get_random_tensor((embedDim, embedDim), 1.0, True)
        Wk = get_random_tensor((embedDim, embedDim), 1.0, True)
        Wv = get_random_tensor((embedDim, embedDim), 1.0, True)
        Wo = get_random_tensor((embedDim, embedDim), 1.0, True)
        await run_multihead_test(dut, "MultiHeadAttention", "I8", I8, X, Wq, Wk, Wv, Wo, False, seqLen, embedDim, numHeads, xLanes, wLanes, collect=collect)
    
    details = f"seq={seqLen}, emb={embedDim}, heads={numHeads}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MultiHeadAttention", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_multihead_i8(request): run_layer_sim("MultiHeadAttention", "I8", "cocotb_multihead_i8", "MultiHeadAttentionTestComp", request)

@cocotb.test()
async def cocotb_multihead_i16(dut):
    seqLen = 4
    embedDim = 4
    numHeads = 2
    xLanes = 4
    wLanes = 4
    
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((seqLen, embedDim), 3.0, True)
        Wq = get_random_tensor((embedDim, embedDim), 3.0, True)
        Wk = get_random_tensor((embedDim, embedDim), 3.0, True)
        Wv = get_random_tensor((embedDim, embedDim), 3.0, True)
        Wo = get_random_tensor((embedDim, embedDim), 3.0, True)
        await run_multihead_test(dut, "MultiHeadAttention", "I16", I16, X, Wq, Wk, Wv, Wo, False, seqLen, embedDim, numHeads, xLanes, wLanes, collect=collect)
    
    details = f"seq={seqLen}, emb={embedDim}, heads={numHeads}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MultiHeadAttention", "I16", I16, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_multihead_i16(request): run_layer_sim("MultiHeadAttention", "I16", "cocotb_multihead_i16", "MultiHeadAttentionTestComp", request)

@cocotb.test()
async def cocotb_multihead_bf16(dut):
    seqLen = 4
    embedDim = 4
    numHeads = 2
    xLanes = 4
    wLanes = 4
    
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((seqLen, embedDim), 1.0, False)
        Wq = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wk = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wv = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wo = get_random_tensor((embedDim, embedDim), 0.5, False)
        await run_multihead_test(dut, "MultiHeadAttention", "BF16", BF16, X, Wq, Wk, Wv, Wo, True, seqLen, embedDim, numHeads, xLanes, wLanes, collect=collect)
    
    details = f"seq={seqLen}, emb={embedDim}, heads={numHeads}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("MultiHeadAttention", "BF16", BF16, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_multihead_bf16(request): run_layer_sim("MultiHeadAttention", "BF16", "cocotb_multihead_bf16", "MultiHeadAttentionTestComp", request)
