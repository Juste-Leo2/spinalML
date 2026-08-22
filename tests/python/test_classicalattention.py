import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np
import pytest
import os

from golden_models.dtypes import FP8_E4M3, I8, I16, BF16, I4, FP4_E2M1
from golden_models.ops import classical_attention_hw, classical_attention_hw_wxay, dequant_hw, matmul_hw, softmax
from utils.test_layers_utils import get_random_tensor, send_tensor, recv_tensor, log_true_math_error, run_layer_sim, DEFAULT_NUM_TRIALS
from utils.math_metrics import log_math_line
from utils.tb_utils import seed_random, SEED

seed_random()

async def run_attention_test(dut, op_name, dtype_name, dtype, X, Wq, Wk, Wv, Wo, is_floatml, seqLen, embedDim, xLanes, wLanes, collect=None):
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
async def cocotb_attention_fp8(dut):
    # Dimensions
    seqLen = 2
    embedDim = 2
    xLanes = 2
    wLanes = 2
    
    # Attention requires scale to not explode the Exp LUT. We use very small random weights.
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((seqLen, embedDim), 1.0, False)
        Wq = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wk = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wv = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wo = get_random_tensor((embedDim, embedDim), 0.5, False)
        await run_attention_test(dut, "ClassicalAttention", "FP8", FP8_E4M3, X, Wq, Wk, Wv, Wo, True, seqLen, embedDim, xLanes, wLanes, collect=collect)
    
    details = f"seq={seqLen}, emb={embedDim}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("ClassicalAttention", "FP8", FP8_E4M3, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_attention_fp8(request): run_layer_sim("ClassicalAttention", "FP8", "cocotb_attention_fp8", "AttentionTestComp", request)

@cocotb.test()
async def cocotb_attention_i8(dut):
    seqLen = 2
    embedDim = 2
    xLanes = 2
    wLanes = 2
    
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((seqLen, embedDim), 1.0, True)
        Wq = get_random_tensor((embedDim, embedDim), 1.0, True)
        Wk = get_random_tensor((embedDim, embedDim), 1.0, True)
        Wv = get_random_tensor((embedDim, embedDim), 1.0, True)
        Wo = get_random_tensor((embedDim, embedDim), 1.0, True)
        await run_attention_test(dut, "ClassicalAttention", "I8", I8, X, Wq, Wk, Wv, Wo, False, seqLen, embedDim, xLanes, wLanes, collect=collect)
    
    details = f"seq={seqLen}, emb={embedDim}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("ClassicalAttention", "I8", I8, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_attention_i8(request): run_layer_sim("ClassicalAttention", "I8", "cocotb_attention_i8", "AttentionTestComp", request)

@cocotb.test()
async def cocotb_attention_i16(dut):
    seqLen = 2
    embedDim = 2
    xLanes = 2
    wLanes = 2
    
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((seqLen, embedDim), 3.0, True)
        Wq = get_random_tensor((embedDim, embedDim), 3.0, True)
        Wk = get_random_tensor((embedDim, embedDim), 3.0, True)
        Wv = get_random_tensor((embedDim, embedDim), 3.0, True)
        Wo = get_random_tensor((embedDim, embedDim), 3.0, True)
        await run_attention_test(dut, "ClassicalAttention", "I16", I16, X, Wq, Wk, Wv, Wo, False, seqLen, embedDim, xLanes, wLanes, collect=collect)
    
    details = f"seq={seqLen}, emb={embedDim}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("ClassicalAttention", "I16", I16, False, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_attention_i16(request): run_layer_sim("ClassicalAttention", "I16", "cocotb_attention_i16", "AttentionTestComp", request)

@cocotb.test()
async def cocotb_attention_bf16(dut):
    seqLen = 2
    embedDim = 2
    xLanes = 2
    wLanes = 2
    
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((seqLen, embedDim), 1.0, False)
        Wq = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wk = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wv = get_random_tensor((embedDim, embedDim), 0.5, False)
        Wo = get_random_tensor((embedDim, embedDim), 0.5, False)
        await run_attention_test(dut, "ClassicalAttention", "BF16", BF16, X, Wq, Wk, Wv, Wo, True, seqLen, embedDim, xLanes, wLanes, collect=collect)
    
    details = f"seq={seqLen}, emb={embedDim}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("ClassicalAttention", "BF16", BF16, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

def test_pytest_attention_bf16(request): run_layer_sim("ClassicalAttention", "BF16", "cocotb_attention_bf16", "AttentionTestComp", request)

# ==========================================
# CLASSICAL ATTENTION — WEIGHT-ONLY QUANTIZATION (wXaY)
# SInt weights (I4/I8) + shared compile-time scale(s), float activations
# ==========================================
QUANT_COMBOS = {
    "w8a16": (I8, 8, BF16),
    "w4a16": (I4, 4, BF16),
    "w8a8": (I8, 8, FP8_E4M3),
    "w4a8": (I4, 4, FP8_E4M3),
    "w8a4": (I8, 8, FP4_E2M1),
    "w4a4": (I4, 4, FP4_E2M1),
}

_PHANTOM_STATE = {}

def _golden_probs(X, Wq_int, Wk_int, act_dtype, scales, w_bits):
    Wq = np.array(dequant_hw(Wq_int, scales, act_dtype, weight_bits=w_bits))
    Wk = np.array(dequant_hw(Wk_int, scales, act_dtype, weight_bits=w_bits))
    Q = np.array(matmul_hw(X, Wq.tolist(), act_dtype))
    K_T = np.array(matmul_hw(X, Wk.tolist(), act_dtype)).T
    S = matmul_hw(Q.tolist(), K_T.tolist(), act_dtype)
    return np.array([softmax(np.array(row), act_dtype) for row in S])

def _phantom_predictions(X, Wq_int, Wk_int, Wv_int, Wo_int, act_dtype, scales, w_bits, P_prev):
    Wv = np.array(dequant_hw(Wv_int, scales, act_dtype, weight_bits=w_bits))
    Wo = np.array(dequant_hw(Wo_int, scales, act_dtype, weight_bits=w_bits))
    Vc = np.array(matmul_hw(X, Wv.tolist(), act_dtype))
    P_cur = _golden_probs(X, Wq_int, Wk_int, act_dtype, scales, w_bits)
    P_shift = np.vstack([P_prev[-1:, :], P_cur[:-1, :]])
    Y_shift = P_shift @ Vc @ Wo
    Y_norm = P_cur @ Vc @ Wo
    return P_cur, Y_shift, Y_norm

async def run_attention_quant_test(dut, op_name, combo_name, w_dtype, w_bits, act_dtype,
                                   X, Wq, Wk, Wv, Wo, seqLen, embedDim, xLanes, wLanes,
                                   scales=(1.0,), collect=None):
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

    # Transpose weights for Matmul, sent as raw SInt bits
    Wq_T = np.array(Wq).T.tolist()
    Wk_T = np.array(Wk).T.tolist()
    Wv_T = np.array(Wv).T.tolist()
    Wo_T = np.array(Wo).T.tolist()

    await send_tensor(dut, "io_wq_stream", Wq_T, (embedDim, embedDim), wLanes, w_dtype, False)
    await send_tensor(dut, "io_wk_stream", Wk_T, (embedDim, embedDim), wLanes, w_dtype, False)
    await send_tensor(dut, "io_wv_stream", Wv_T, (embedDim, embedDim), wLanes, w_dtype, False)
    await send_tensor(dut, "io_wo_stream", Wo_T, (embedDim, embedDim), wLanes, w_dtype, False)

    send_x = cocotb.start_soon(send_tensor(dut, "io_x_stream", X, (seqLen, embedDim), xLanes, act_dtype, True))
    recv_y = cocotb.start_soon(recv_tensor(dut, "io_y_stream", (seqLen, embedDim), act_dtype, True, 1))

    Y_out_bits, Y_out = await recv_y
    await send_x

    # Hardware exact math (weights dequantized to the activation float domain)
    Y_expected = classical_attention_hw_wxay(X, Wq, Wk, Wv, Wo, act_dtype, scales, weight_bits=w_bits)

    # TEMP DEBUG: per-trial dump + phantom-row hypothesis check
    trial_idx = len(collect["out"]) if collect is not None else 0
    log_math_line(f"ATTDBG t{trial_idx} {combo_name} X={[[round(x, 3) for x in r] for r in X]}")
    log_math_line(f"ATTDBG t{trial_idx} Y_hw={[[round(x, 4) for x in r] for r in Y_out]}")
    log_math_line(f"ATTDBG t{trial_idx} Y_gd={[[round(x, 4) for x in r] for r in Y_expected]}")
    errs = [[round(abs(Y_out[m][n] - float(Y_expected[m][n])), 4) for n in range(embedDim)] for m in range(seqLen)]
    log_math_line(f"ATTDBG t{trial_idx} abs_err={errs}")

    P_prev = _PHANTOM_STATE.get(combo_name)
    if P_prev is not None and trial_idx >= 1:
        P_cur, Y_shift, Y_norm = _phantom_predictions(
            X, Wq, Wk, Wv, Wo, act_dtype, scales, w_bits, P_prev)
        d_shift = float(np.max(np.abs(np.array(Y_out) - Y_shift)))
        d_norm = float(np.max(np.abs(np.array(Y_out) - Y_norm)))
        log_math_line(f"ATTDBG t{trial_idx} PHANTOM max|Yhw-Yshift|={d_shift:.4f} max|Yhw-Ynorm|={d_norm:.4f}")
        log_math_line(f"ATTDBG t{trial_idx} Y_shift_pred={[[round(x, 4) for x in r] for r in Y_shift.tolist()]}")
    _PHANTOM_STATE[combo_name] = _golden_probs(X, Wq, Wk, act_dtype, scales, w_bits)

    if collect is not None:
        collect["out"].append(Y_out)
        collect["true"].append(Y_expected)

    # Same relaxed margins as the uniform float path (softmax PWL approximations)
    for m in range(seqLen):
        for n in range(embedDim):
            out_val = Y_out[m][n]
            exp_val = float(Y_expected[m][n])
            if np.isinf(out_val) or np.isinf(exp_val):
                assert np.isinf(out_val) and np.isinf(exp_val) and np.signbit(out_val) == np.signbit(exp_val), \
                    f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {exp_val}"
            else:
                err = abs(out_val - exp_val)
                assert err < 0.25, f"HW Mismatch at Y[{m}][{n}]: got {out_val} instead of {exp_val}"

async def _run_attention_quant_combo(dut, combo_name, scales=(1.0,)):
    w_dt, w_bits, a_dt = QUANT_COMBOS[combo_name]
    seqLen, embedDim, xLanes, wLanes = 2, 2, 2, 2
    collect = {"out": [], "true": []}
    for _ in range(DEFAULT_NUM_TRIALS):
        X = get_random_tensor((seqLen, embedDim), 1.0, False)
        Wq = get_random_tensor((embedDim, embedDim), 3.0, True)
        Wk = get_random_tensor((embedDim, embedDim), 3.0, True)
        Wv = get_random_tensor((embedDim, embedDim), 3.0, True)
        Wo = get_random_tensor((embedDim, embedDim), 3.0, True)
        await run_attention_quant_test(dut, "ClassicalAttention", combo_name, w_dt, w_bits, a_dt,
                                       X, Wq, Wk, Wv, Wo, seqLen, embedDim, xLanes, wLanes,
                                       scales=scales, collect=collect)
    details = f"seq={seqLen}, emb={embedDim}, scales={scales}, trials={DEFAULT_NUM_TRIALS}, seed={int(os.environ.get('SPINALML_SEED', SEED))}"
    log_msg = log_true_math_error("ClassicalAttentionQuant", combo_name, a_dt, True, collect["out"], collect["true"], details=details)
    dut._log.info(log_msg)

@cocotb.test()
async def cocotb_attention_quant_w8a16(dut): await _run_attention_quant_combo(dut, "w8a16")

@cocotb.test()
async def cocotb_attention_quant_w4a16(dut): await _run_attention_quant_combo(dut, "w4a16")

@cocotb.test()
async def cocotb_attention_quant_w8a8(dut): await _run_attention_quant_combo(dut, "w8a8")

@cocotb.test()
async def cocotb_attention_quant_w4a8(dut): await _run_attention_quant_combo(dut, "w4a8")

@cocotb.test()
async def cocotb_attention_quant_w8a4(dut): await _run_attention_quant_combo(dut, "w8a4")

@cocotb.test()
async def cocotb_attention_quant_w4a4(dut): await _run_attention_quant_combo(dut, "w4a4")

@cocotb.test()
async def cocotb_attention_quant_perchannel(dut):
    # Per-channel scales: one weight scale per output column (embedDim=2)
    await _run_attention_quant_combo(dut, "w8a16", scales=(0.5, 2.0))

def test_pytest_attention_quant_w8a16(request): run_layer_sim("ClassicalAttention", "w8a16", "cocotb_attention_quant_w8a16", "AttentionQuantTestComp", request)
def test_pytest_attention_quant_w4a16(request): run_layer_sim("ClassicalAttention", "w4a16", "cocotb_attention_quant_w4a16", "AttentionQuantTestComp", request)
def test_pytest_attention_quant_w8a8(request): run_layer_sim("ClassicalAttention", "w8a8", "cocotb_attention_quant_w8a8", "AttentionQuantTestComp", request)
def test_pytest_attention_quant_w4a8(request): run_layer_sim("ClassicalAttention", "w4a8", "cocotb_attention_quant_w4a8", "AttentionQuantTestComp", request)
def test_pytest_attention_quant_w8a4(request): run_layer_sim("ClassicalAttention", "w8a4", "cocotb_attention_quant_w8a4", "AttentionQuantTestComp", request)
def test_pytest_attention_quant_w4a4(request): run_layer_sim("ClassicalAttention", "w4a4", "cocotb_attention_quant_w4a4", "AttentionQuantTestComp", request)
def test_pytest_attention_quant_perchannel(request): run_layer_sim("ClassicalAttention", "PerChannel", "cocotb_attention_quant_perchannel", "AttentionQuantTestComp", request)
