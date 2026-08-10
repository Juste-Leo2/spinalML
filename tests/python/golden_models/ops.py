import numpy as np
from golden_models.dtypes import FloatML

def floatml_add(a: float, b: float, dtype: FloatML) -> float:
    """Golden model for FloatML hardware addition."""
    # hardware logic: converts to float, adds, then re-quantizes
    res = a + b
    res_bits = dtype.from_float(res)
    return dtype.to_float(res_bits)

def floatml_mul(a: float, b: float, dtype: FloatML) -> float:
    """Golden model for FloatML hardware multiplication."""
    res = a * b
    res_bits = dtype.from_float(res)
    return dtype.to_float(res_bits)

def rsqrt(x: float, dtype: FloatML = None) -> float:
    """Golden model for Inverse Square Root. Includes PWL approximation error if dtype is provided."""
    if x <= 0:
        return 0.0 # Or hardware default
    
    # Exact math (matching SpinalHDL 1e-5 offset to avoid div by zero)
    exact = 1.0 / np.sqrt(abs(x) + 1e-5)
    
    if dtype is None:
        return exact
        
    # Re-quantize to simulate hardware rounding
    return dtype.to_float(dtype.from_float(exact))

def exp(x: float, dtype: FloatML = None) -> float:
    """Golden model for Exponential."""
    exact = np.exp(x)
    if dtype is None:
        return exact
    return dtype.to_float(dtype.from_float(exact))

def softmax(x: np.ndarray, dtype: FloatML = None) -> np.ndarray:
    """Golden model for Softmax1D pipeline."""
    # 1. Max-Tree
    max_val = np.max(x)
    if dtype: max_val = dtype.to_float(dtype.from_float(max_val))
    
    # 2. Subtract
    shifted = x - max_val
    if dtype: shifted = np.array([dtype.to_float(dtype.from_float(v)) for v in shifted])
    
    # 3. Exp
    exp_vals = np.exp(shifted)
    if dtype: exp_vals = np.array([dtype.to_float(dtype.from_float(v)) for v in exp_vals])
    
    # 4. Sum
    sum_val = np.sum(exp_vals)
    if dtype: sum_val = dtype.to_float(dtype.from_float(sum_val))
    
    # 5. Reciprocal
    recip = 1.0 / sum_val if sum_val != 0 else 0.0
    if dtype: recip = dtype.to_float(dtype.from_float(recip))
    
    # 6. Final Multiply
    final = exp_vals * recip
    if dtype: final = np.array([dtype.to_float(dtype.from_float(v)) for v in final])
    
    return final

import math
def pwl_rsqrt_int(x_val: float, bit_width: int, index_bits: int = 8) -> int:
    """Golden model reproduisant exactement l'approximation linéaire (PWL) matérielle pour les entiers."""
    x_int = int(x_val)
    if x_int < 0:
        x_int += (1 << bit_width)
        
    shift = bit_width - index_bits
    segment_index = (x_int >> shift) & ((1 << index_bits) - 1)
    
    i_start = segment_index << shift
    i_end = ((segment_index + 1) << shift) - 1
    
    def intValFn(i):
        halfVal = 1 << (bit_width - 1)
        maxVal = 1 << bit_width
        return float(i - maxVal) if i >= halfVal else float(i)
        
    xs = intValFn(i_start)
    xe = intValFn(i_end)
    
    ys = 1.0 / np.sqrt(abs(xs) + 1e-5)
    ye = 1.0 / np.sqrt(abs(xe) + 1e-5)
    
    a = (ye - ys) / (xe - xs) if xe != xs else 0.0
    b = ys - a * xs
    
    def intEncodeFn(y):
        minVal = -(1 << (bit_width - 1))
        maxVal = (1 << (bit_width - 1)) - 1
        return int(max(float(minVal), min(float(maxVal), math.floor(y + 0.5))))
        
    a_enc = intEncodeFn(a)
    b_enc = intEncodeFn(b)
    
    # Simulation du matériel: vx * va + bCoef
    vx = intValFn(x_int)
    p = int(vx) * a_enc
    res = p + b_enc
    
    # Truncate to bit_width (SInt behavior)
    res = res & ((1 << bit_width) - 1)
    if res >= (1 << (bit_width - 1)):
        res -= (1 << bit_width)
    return res
