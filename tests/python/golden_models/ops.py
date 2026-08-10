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
def pwl_int(x_val: float, bit_width: int, math_fn, index_bits: int = 8) -> int:
    """Golden model reproduisant exactement l'approximation linéaire (PWL) matérielle pour les entiers, pour n'importe quelle fonction."""
    # The input bits exactly as injected by Cocotb
    from golden_models.dtypes import SIntML
    x_int = SIntML(bit_width).from_float(x_val)
    # Si x_int représente un nombre négatif en complément à 2, x_int > 0 dans sa version binaire non-signée
    # Mais le segment index a besoin des bits bruts
    x_bits = x_int
        
    shift = bit_width - index_bits
    segment_index = (x_bits >> shift) & ((1 << index_bits) - 1)
    
    i_start = segment_index << shift
    i_end = ((segment_index + 1) << shift) - 1
    
    def intValFn(i):
        halfVal = 1 << (bit_width - 1)
        maxVal = 1 << bit_width
        return float(i - maxVal) if i >= halfVal else float(i)
        
    xs = intValFn(i_start)
    xe = intValFn(i_end)
    
    # Handle overflow in math_fn (e.g. math.exp on large numbers)
    try:
        ys = math_fn(xs)
    except OverflowError:
        ys = float('inf')
    try:
        ye = math_fn(xe)
    except OverflowError:
        ye = float('inf')
        
    a = (ye - ys) / (xe - xs) if xe != xs else 0.0
    # math.inf - math.inf is nan, so we protect b
    b = ys - a * xs
    
    if math.isnan(a) or math.isinf(a): a = 0.0
    if math.isnan(b) or math.isinf(b): b = ys if not math.isinf(ys) else 0.0
    
    def intEncodeFn(y):
        minVal = -(1 << (bit_width - 1))
        maxVal = (1 << (bit_width - 1)) - 1
        return int(max(float(minVal), min(float(maxVal), math.floor(y + 0.5))))
        
    a_enc = intEncodeFn(a)
    b_enc = intEncodeFn(b)
    
    # Simulation du matériel: vx * va + bCoef
    vx = intValFn(x_bits)
    p = int(vx) * a_enc
    res = p + b_enc
    
    # Truncate to bit_width (unsigned bits behavior to match FPGA io)
    res = res & ((1 << bit_width) - 1)
    return res

def pwl_rsqrt_int(x_val: float, bit_width: int, index_bits: int = 8) -> int:
    def rsqrt_fn(x):
        return 1.0 / np.sqrt(abs(x) + 1e-5)
    return pwl_int(x_val, bit_width, rsqrt_fn, index_bits)

def pwl_exp_int(x_val: float, bit_width: int, index_bits: int = 8) -> int:
    def exp_fn(x):
        return math.exp(x)
    return pwl_int(x_val, bit_width, exp_fn, index_bits)

def pwl_float(x_val: float, dtype, math_fn, index_bits: int = 8) -> int:
    """Golden model reproduisant exactement l'approximation linéaire (PWL) matérielle pour les flottants."""
    bit_width = dtype.exp_bits + dtype.mant_bits + 1
    
    x_bits = dtype.from_float(x_val)
    shift = bit_width - index_bits
    segment_index = (x_bits >> shift) & ((1 << index_bits) - 1)
    
    i_start = segment_index << shift
    i_end = ((segment_index + 1) << shift) - 1
    
    xs = dtype.to_float(i_start)
    xe = dtype.to_float(i_end)
    
    # Handle overflow in math_fn
    try:
        ys = math_fn(xs)
    except OverflowError:
        ys = float('inf')
    try:
        ye = math_fn(xe)
    except OverflowError:
        ye = float('inf')
        
    a = (ye - ys) / (xe - xs) if xe != xs else 0.0
    b = ys - a * xs
    
    if math.isnan(a) or math.isinf(a): a = 0.0
    if math.isnan(b) or math.isinf(b): b = ys if not math.isinf(ys) else 0.0
    
    a_bits = dtype.from_float(a)
    b_bits = dtype.from_float(b)
    
    a_hw = dtype.to_float(a_bits)
    b_hw = dtype.to_float(b_bits)
    
    # L'architecture matérielle (SpinalHDL Float) utilise une troncature (Round Towards Zero) pour économiser de la logique
    def float_mul_hw(v1: float, v2: float) -> float:
        b1 = dtype.from_float(v1); b2 = dtype.from_float(v2)
        s1 = (b1 >> (dtype.exp_bits + dtype.mant_bits)) & 1
        e1 = (b1 >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
        m1 = (b1 & ((1 << dtype.mant_bits) - 1)) | (1 << dtype.mant_bits) if e1 > 0 else 0
        s2 = (b2 >> (dtype.exp_bits + dtype.mant_bits)) & 1
        e2 = (b2 >> dtype.mant_bits) & ((1 << dtype.exp_bits) - 1)
        m2 = (b2 & ((1 << dtype.mant_bits) - 1)) | (1 << dtype.mant_bits) if e2 > 0 else 0
        
        if e1 == 0 or e2 == 0: return 0.0
        
        sign = s1 ^ s2
        exp = e1 + e2 - dtype.bias
        mant_prod = m1 * m2
        
        if (mant_prod & (1 << (2 * dtype.mant_bits + 1))) != 0:
            exp += 1
            mant_trunc = (mant_prod >> (dtype.mant_bits + 1)) & ((1 << dtype.mant_bits) - 1)
        else:
            mant_trunc = (mant_prod >> dtype.mant_bits) & ((1 << dtype.mant_bits) - 1)
            
        if exp >= ((1 << dtype.exp_bits) - 1): return dtype.to_float((sign << (dtype.exp_bits + dtype.mant_bits)) | (((1 << dtype.exp_bits) - 1) << dtype.mant_bits))
        if exp <= 0: return 0.0
        return dtype.to_float((sign << (dtype.exp_bits + dtype.mant_bits)) | (exp << dtype.mant_bits) | mant_trunc)

    def float_add_hw(v1: float, v2: float) -> float:
        # Pour simplifier, on utilise le float Python mais on force la troncature de la mantisse
        import math
        res = v1 + v2
        if res == 0.0: return 0.0
        mant, exp = math.frexp(abs(res))
        mant = mant * 2.0 - 1.0
        mant_val = int(mant * (1 << dtype.mant_bits)) # Int cast truncates (Round Towards Zero)
        exp_val = exp - 1 + dtype.bias
        if exp_val >= ((1 << dtype.exp_bits) - 1): exp_val = (1 << dtype.exp_bits) - 1; mant_val = 0
        if exp_val <= 0: return 0.0
        sign = 1 if res < 0 else 0
        return dtype.to_float((sign << (dtype.exp_bits + dtype.mant_bits)) | (exp_val << dtype.mant_bits) | mant_val)

    # p = a * x (tronqué HW)
    p_hw = float_mul_hw(a_hw, x_val)
    # res = p + b (tronqué HW)
    res_hw = float_add_hw(p_hw, b_hw)
    
    return dtype.from_float(res_hw)

def pwl_exp_float(x_val: float, dtype, index_bits: int = 8) -> int:
    def exp_fn(x):
        return math.exp(x)
    return pwl_float(x_val, dtype, exp_fn, index_bits)

def pwl_rsqrt_float(x_val: float, dtype, index_bits: int = 8) -> int:
    def rsqrt_fn(x):
        return 1.0 / np.sqrt(abs(x) + 1e-5)
    return pwl_float(x_val, dtype, rsqrt_fn, index_bits)
