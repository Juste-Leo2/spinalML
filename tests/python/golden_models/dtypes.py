# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import numpy as np
import os


def resolve_rounding(rounding=None):
    """Shared rounding switch: None follows SPINALML_ROUNDING (default RNE).

    Mirrors the RTL elaboration switch (`RoundingConfig.current`): 'rne'
    (half-even) vs 'trunc' (legacy half-up / truncation bit-exact).
    """
    if rounding is None:
        raw = os.environ.get("SPINALML_ROUNDING", "")
        rounding = "trunc" if raw.strip().lower() in ("trunc", "truncate", "floor") else "rne"
    return rounding

class FloatML:
    def __init__(self, exp_bits: int, mant_bits: int):
        self.exp_bits = exp_bits
        self.mant_bits = mant_bits
        self.bias = (1 << (exp_bits - 1)) - 1
        
    def to_float(self, bits: int) -> float:
        """Convert the integer representation of the FloatML into a Python float"""
        if bits == 0:
            return 0.0
            
        sign_bit = (bits >> (self.exp_bits + self.mant_bits)) & 1
        exp_val = (bits >> self.mant_bits) & ((1 << self.exp_bits) - 1)
        mant_val = bits & ((1 << self.mant_bits) - 1)
        
        # Handle zero
        if exp_val == 0:
            return 0.0 # Subnormals are omitted in hardware
            
        # Handle infinity
        if exp_val == ((1 << self.exp_bits) - 1):
            return float('-inf') if sign_bit else float('inf')
            
        # Normal value
        fraction = 1.0 + (mant_val / (1 << self.mant_bits))
        value = fraction * (2 ** (exp_val - self.bias))
        return -value if sign_bit else value

    def from_float(self, value: float, rounding=None) -> int:
        """Convert a Python float into the integer bit representation of the FloatML.

        The mantissa quantizes half-even (Python round, like the RTL roundRNE)
        under RNE, or legacy half-up (floor(x + 0.5), like Math.round) under
        trunc. None follows SPINALML_ROUNDING.
        """
        rounding = resolve_rounding(rounding)
        if value == 0.0 or np.isnan(value):
            return 0
            
        sign_bit = 1 if value < 0 else 0
        value = abs(value)
        
        # Handle infinity and overflow (same formula as Scala
        # doubleToFields). E4M3 has no infinity: it tops out at max finite
        # 448 (exp field 15, mantissa 6); the mantissa-7 slot is NaN.
        is_e4m3 = self.exp_bits == 4 and self.mant_bits == 3
        if is_e4m3:
            max_val = 448.0
        else:
            max_exp = ((1 << self.exp_bits) - 2)
            max_mant = ((1 << self.mant_bits) - 1)
            max_val = (1.0 + max_mant / (1 << self.mant_bits)) * (2 ** (max_exp - self.bias))

        if value > max_val or np.isinf(value):
            # Saturate as per Scala logic: 448 for E4M3, canonical
            # infinity (exp all-ones, mant 0) for other formats.
            if is_e4m3:
                exp_val = (1 << self.exp_bits) - 1
                mant_val = (1 << self.mant_bits) - 2
            else:
                exp_val = (1 << self.exp_bits) - 1
                mant_val = 0
            return (sign_bit << (self.exp_bits + self.mant_bits)) | (exp_val << self.mant_bits) | mant_val
            
        # Extract mantissa and exponent
        # Python floats are doubles (FP64), so we can just use math.frexp
        import math
        mant, exp = math.frexp(value) # mant is [0.5, 1.0)
        # We need mant in [1.0, 2.0)
        mant *= 2.0
        exp -= 1
        
        # Adjust exponent with bias
        exp_val = exp + self.bias
        if rounding != "trunc":
            mant_val = int(round((mant - 1.0) * (1 << self.mant_bits)))
        else:
            mant_val = int(math.floor((mant - 1.0) * (1 << self.mant_bits) + 0.5))
        
        # Handle mantissa rounding overflow FIRST (can rescue an underflow)
        if mant_val >= (1 << self.mant_bits):
            mant_val = 0
            exp_val += 1
            
        # NOW check for Saturation / Underflow. E4M3 keeps finite field-15
        # values (256..448); only past-the-max or the NaN slot saturates.
        exp_max = (1 << self.exp_bits) - 1
        if is_e4m3:
            needs_sat = exp_val > exp_max or (exp_val == exp_max and mant_val == (1 << self.mant_bits) - 1)
        else:
            needs_sat = exp_val >= exp_max
        if needs_sat:
            if is_e4m3:
                exp_val = (1 << self.exp_bits) - 1
                mant_val = (1 << self.mant_bits) - 2
            else:
                exp_val = (1 << self.exp_bits) - 1
                mant_val = 0
        elif exp_val <= 0:
            exp_val = 0
            mant_val = 0
                
        return (sign_bit << (self.exp_bits + self.mant_bits)) | (exp_val << self.mant_bits) | mant_val

FP4_E2M1 = FloatML(2, 1)
FP8_E4M3 = FloatML(4, 3)
BF16 = FloatML(8, 7)
FP32 = FloatML(8, 23)

class SIntML:
    def __init__(self, bit_width: int):
        self.bit_width = bit_width
        self.max_val = (1 << (bit_width - 1)) - 1
        self.min_val = -(1 << (bit_width - 1))
        
    def to_float(self, bits: int) -> float:
        if bits >= (1 << (self.bit_width - 1)):
            return float(bits - (1 << self.bit_width))
        return float(bits)
        
    def from_float(self, value: float, rounding=None) -> int:
        """Integer quantize (2's complement bits): half-even under
        RNE (like the RTL roundRNE), legacy half-up under trunc. None
        follows SPINALML_ROUNDING."""
        if np.isnan(value):
            return 0
        import math
        rounding = resolve_rounding(rounding)
        if rounding != "trunc":
            q = round(value)
        else:
            # Java's Math.round equivalent
            q = math.floor(value + 0.5)
        clamped = max(float(self.min_val), min(float(self.max_val), q))
        int_val = int(clamped)
        if int_val < 0:
            return int_val + (1 << self.bit_width)
        return int_val

I4 = SIntML(4)
I8 = SIntML(8)
I16 = SIntML(16)
I32 = SIntML(32)
