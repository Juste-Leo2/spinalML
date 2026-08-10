import numpy as np

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

    def from_float(self, value: float) -> int:
        """Convert a Python float into the integer bit representation of the FloatML"""
        if value == 0.0 or np.isnan(value):
            return 0
            
        sign_bit = 1 if value < 0 else 0
        value = abs(value)
        
        # Handle infinity and overflow
        max_exp = ((1 << self.exp_bits) - 2)
        max_mant = ((1 << self.mant_bits) - 1)
        max_val = (1.0 + max_mant / (1 << self.mant_bits)) * (2 ** (max_exp - self.bias))
        
        if value > max_val or np.isinf(value):
            # Saturate to infinity as per Scala logic
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
        mant_val = int(round((mant - 1.0) * (1 << self.mant_bits)))
        
        # Handle mantissa rounding overflow FIRST (can rescue an underflow)
        if mant_val >= (1 << self.mant_bits):
            mant_val = 0
            exp_val += 1
            
        # NOW check for Saturation / Underflow
        if exp_val >= ((1 << self.exp_bits) - 1):
            exp_val = (1 << self.exp_bits) - 1
            mant_val = 0
        elif exp_val <= 0:
            exp_val = 0
            mant_val = 0
                
        return (sign_bit << (self.exp_bits + self.mant_bits)) | (exp_val << self.mant_bits) | mant_val

FP4_E2M1 = FloatML(2, 1)
FP8_E4M3 = FloatML(4, 3)
BF16 = FloatML(8, 7)

class SIntML:
    def __init__(self, bit_width: int):
        self.bit_width = bit_width
        self.max_val = (1 << (bit_width - 1)) - 1
        self.min_val = -(1 << (bit_width - 1))
        
    def to_float(self, bits: int) -> float:
        if bits >= (1 << (self.bit_width - 1)):
            return float(bits - (1 << self.bit_width))
        return float(bits)
        
    def from_float(self, value: float) -> int:
        if np.isnan(value):
            return 0
        import math
        # Java's Math.round equivalent
        clamped = max(float(self.min_val), min(float(self.max_val), math.floor(value + 0.5)))
        int_val = int(clamped)
        if int_val < 0:
            return int_val + (1 << self.bit_width)
        return int_val

I4 = SIntML(4)
I8 = SIntML(8)
I16 = SIntML(16)
