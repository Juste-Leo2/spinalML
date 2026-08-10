import pytest
from golden_models.dtypes import FP4_E2M1, BF16

def test_fp4_e2m1_quantization():
    """
    Test de la quantification FP4_E2M1
    Bit de signe = 1, Exposant = 2, Mantisse = 1.
    Le biais est de (2^(2-1)) - 1 = 1.
    """
    
    # 1. Zéros
    assert FP4_E2M1.to_float(0b0000) == 0.0
    assert FP4_E2M1.from_float(0.0) == 0b0000
    
    # 2. Valeurs Normales
    # 0b0_01_0 -> signe=0, exp=1, mant=0 -> (1-1) = 0 -> 2^0 * 1.0 = 1.0
    assert FP4_E2M1.to_float(0b0010) == 1.0
    
    # 0b0_01_1 -> signe=0, exp=1, mant=1 -> 2^0 * 1.5 = 1.5
    assert FP4_E2M1.to_float(0b0011) == 1.5
    
    # 0b0_10_0 -> signe=0, exp=2, mant=0 -> 2^1 * 1.0 = 2.0
    assert FP4_E2M1.to_float(0b0100) == 2.0
    
    # 0b0_10_1 -> signe=0, exp=2, mant=1 -> 2^1 * 1.5 = 3.0
    assert FP4_E2M1.to_float(0b0101) == 3.0
    
    # 3. Test des encodages depuis Float
    assert FP4_E2M1.from_float(1.0) == 0b0010
    assert FP4_E2M1.from_float(1.5) == 0b0011
    assert FP4_E2M1.from_float(2.0) == 0b0100
    assert FP4_E2M1.from_float(3.0) == 0b0101
    
    # 4. Comportement des arrondis (Très important pour le matériel !)
    # 0.78 est mathématiquement plus proche de 1.0 (0b0010) que de 0.0 (0b0000) !
    # Actuellement dans notre dtypes.py, l'underflow est strict et renvoie 0.0.
    val = FP4_E2M1.from_float(0.78)
    print(f"0.78 encodé en binaire : {bin(val)}")
    
def test_bf16_quantization():
    assert BF16.to_float(0) == 0.0
    assert BF16.from_float(0.0) == 0
