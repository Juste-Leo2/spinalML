import os

import numpy as np

_MATH_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "true_math_errors.log")

def _flatten(values):
    arr = np.array(values, dtype=np.float64)
    return arr.flatten()

def full_scale_value(dtype):
    bit_width = getattr(dtype, 'bit_width', getattr(dtype, 'exp_bits', 0) + getattr(dtype, 'mant_bits', 0))
    return (1 << (bit_width - 1)) - 1

def compute_metrics(out_vals, true_vals, is_floatml, dtype):
    """Compute MAPE (or MAE %FS for integers), NMSE and Cosine similarity.

    out_vals / true_vals: nested lists or flat lists, same shape, element-wise comparable.
    Returns a dict with keys: mae, mape (float|None), maefs (float|None), nmse, cosine.
    """
    out = _flatten(out_vals)
    true = _flatten(true_vals)

    if out.size == 0:
        return {"mae": None, "mape": None, "maefs": None, "nmse": None, "cosine": None}

    diff = out - true
    metrics = {"mae": float(np.mean(np.abs(diff)))}

    if is_floatml:
        nonzero = true != 0
        if np.any(nonzero):
            metrics["mape"] = float(np.mean(np.abs(diff[nonzero] / true[nonzero])) * 100)
        else:
            metrics["mape"] = None
        metrics["maefs"] = None
    else:
        fs_val = full_scale_value(dtype)
        metrics["mape"] = None
        metrics["maefs"] = float(np.mean(np.abs(diff)) / fs_val * 100)

    mse = float(np.mean(diff ** 2))
    var_true = float(np.var(true))
    if var_true < 1e-12:
        metrics["nmse"] = 0.0 if mse < 1e-12 else 999.99
    else:
        metrics["nmse"] = mse / var_true

    denom = float(np.linalg.norm(out) * np.linalg.norm(true))
    if denom < 1e-12:
        metrics["cosine"] = 0.0
    else:
        metrics["cosine"] = float(np.dot(out, true) / denom)

    return metrics

def format_metrics_line(op_name, dtype_name, metrics, is_floatml, label="Test", details=""):
    first_metric = ""
    if is_floatml:
        if metrics["mape"] is not None:
            first_metric = f"MAPE: {metrics['mape']:.2f}%"
        else:
            first_metric = "MAPE: n/a"
    else:
        if metrics["maefs"] is not None:
            first_metric = f"MAE %FS: {metrics['maefs']:.2f}%"
        else:
            first_metric = "MAE %FS: n/a"

    nmse = "n/a" if metrics["nmse"] is None else f"{metrics['nmse']:.3f}"
    cosine = "n/a" if metrics["cosine"] is None else f"{metrics['cosine']:.3f}"
    mae = "n/a" if metrics["mae"] is None else f"{metrics['mae']:.4f}"

    line = f"[{op_name}][{dtype_name}] {label:<10}| MAE: {mae} | {first_metric}  | NMSE: {nmse} | Cosine: {cosine}"
    if details:
        line += f" ({details})"
    return line

def log_math_line(line):
    """Write a line to the centralized math log (only when DEBUG_MATH=1)."""
    if os.environ.get("DEBUG_MATH") == "1":
        with open(_MATH_LOG_PATH, "a") as f:
            f.write(line + "\n")
    return line

def clear_math_log():
    """Truncate the centralized math log (only when DEBUG_MATH=1)."""
    if os.environ.get("DEBUG_MATH") == "1":
        with open(_MATH_LOG_PATH, "w"):
            pass