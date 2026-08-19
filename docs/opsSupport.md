# Machine Learning Operations

This document lists the machine learning operations that spinalML aims to support natively in hardware.

## Hardware Memory Management Guidelines
To ensure optimal synthesis on FPGA, operations must follow these memory guidelines:
- **Streaming & Element-wise Ops**: Do not instantiate `Mem`. Use pure combinational logic or simple `Vec(Reg(dataType))` for 1D sliding windows (e.g., `seq2col`, `MaxPool1D`). These map efficiently to Shift Register LUTs (SRLs).
- **Stateful Ops (Large Buffers & 2D Windows)**: When storing large tiles or full tensors (e.g., `matmul` weights, `transpose` buffers, `im2col` line buffers), you MUST use `Mem` with **`readSync`** (synchronous read). This introduces a 1-cycle latency that must be pipelined, but guarantees the synthesizer will infer **Block RAM (BRAM)**. Using `readAsync` forces the synthesizer to use distributed LUTRAM, which consumes massive logic resources and ruins Fmax.
- **FloatML Non-Linear Ops (`Exp`, `Reciprocal`, `Sqrt`)**: Instead of generic Piece-Wise Linear (PWL) approximations which perform poorly on floating point numbers, we implement **Algebraic Separation** (Exponent/Mantissa Split). For example, $e^X = 2^{X \log_2(e)}$ decomposes the exponent into integer and fractional parts, allowing a small, fast 1D LUT (e.g., 256 entries for `BF16` mantissa) to compute the fractional mantissa with near-perfect accuracy without DSP multipliers.

## Basic Arithmetic
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Symbolically Verified | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Add` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Element-wise addition of two tensors. |
| `Sub` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Element-wise subtraction of two tensors. |
| `Mul` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Element-wise multiplication (Hadamard product). |
| `Div` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Element-wise division (Mul + Reciprocal). |
| `BiasAdd` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Broadcast add of a bias vector over the last dimension (columns). |
| `Exp` | [⚠️](#methodology-notes) (LUT) | [⚠️](#methodology-notes) (PWL) | ✅ (LUT) | ✅ ([Alg+LUT](#methodology-notes)) | ✅ | ✅ | Exponential. |
| `Log` | [⚠️](#methodology-notes) (LUT) | [⚠️](#methodology-notes) (PWL) | ✅ (LUT) | ✅ ([Alg+LUT](#methodology-notes)) | ✅ | ✅ | Element-wise logarithm, compile-time base (default `e` = ln, `10` = log10). Domain `x <= 0 -> 0` (industry convention, like `Rsqrt`). |
| `Abs` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Element-wise absolute value. |
| `Reciprocal` | [⚠️](#methodology-notes) (LUT) | [⚠️](#methodology-notes) (PWL) | ✅ (LUT) | ✅ ([Alg+LUT](#methodology-notes)) | ✅ | ✅ | Reciprocal (1/X). |
| `Rsqrt` | [⚠️](#methodology-notes) (LUT) | [⚠️](#methodology-notes) (PWL) | ✅ (LUT) | ✅ ([Alg+LUT](#methodology-notes)) | ✅ | ✅ | Inverse Square Root. |
| `Sqrt` | ✅ (LUT) | ✅ (PWL) | ✅ (LUT) | ✅ ([Alg+LUT](#methodology-notes)) | ✅ | ✅ | Square Root. |
| `Scale Add` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Fused Multiply-Add (A*X + B). Mapped to DSP48. |
| `CumSum` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Cumulative Sum over sequence dimension (L). |

## Matrix and Vector Operations
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Symbolically Verified | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `MatMul` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Matrix multiplication of 2D tensors. *(Requires BRAM)* |
| `Dot` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Dot product of two 1D vectors (wrapper over `matmul`). |

## Tensor Manipulations
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Symbolically Verified | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Reshape` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Change the dimensions of a tensor without changing its data. |
| `Transpose` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Permute the dimensions of a tensor. *(Requires BRAM)* |
| `Concatenate` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Join a sequence of tensors along an existing axis. |
| `Slice` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Extract a subset of elements from a tensor. |
| `Flatten` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Flatten a multi-dimensional tensor into a 1D tensor. |
| `Seq2Col` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Convert a 1D sequence into sliding windows. |
| `Im2Col` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Convert a 2D image into sliding windows. *(Requires BRAM)* |
| `Repack` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Hardware gearbox to alter the number of lanes (e.g., from 1 to 4 lanes). |

## Data Conversion
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Symbolically Verified | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Cast` | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | SInt -> FloatML conversion (any SInt width, any float format). Only one direction for now; UInt not supported yet. |
| `Requantize` | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Shift + saturate larger SInt (e.g. I32) to smaller SInt (I8/I16). SInt -> SInt only; no float, no UInt. |

## Neural Network Layers
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Symbolically Verified | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Linear` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Fully connected dense layer. |
| `Conv1D` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 1D Convolution over sequence. |
| `Conv2D` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 2D Convolution over image. *(Requires BRAM)* |
| `DepthwiseConv` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Depthwise separable convolution. *(Requires BRAM)* |

## Attention Mechanisms
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Symbolically Verified | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Classical Attention` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | Scaled dot-product attention (Q, K, V). |
| `Multi-Head Attention`| ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Multiple parallel attention heads. |
| `Mamba2` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Advanced state space models for sequence processing. |

## Activation Functions
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Symbolically Verified | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ReLU` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Rectified Linear Unit. |
| `LeakyReLU` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Leaky Rectified Linear Unit. |
| `Sigmoid` | [⚠️](#methodology-notes) (LUT) | [⚠️](#methodology-notes) (PWL) | ✅ (LUT) | ✅ ([Alg+LUT](#methodology-notes)) | ✅ | ✅ | Sigmoid = 1/(1+e^(-x)). Composition of Negation -> Exp -> +1 -> Reciprocal (ints ⚠️: PWL chain, degenerate for large |x|). |
| `Tanh` | [⚠️](#methodology-notes) (LUT) | [⚠️](#methodology-notes) (PWL) | ✅ (LUT) | ✅ ([Alg+LUT](#methodology-notes)) | ✅ | ✅ | Hyperbolic tangent = 2·sigmoid(2x) - 1. Composition of Mul(×2) -> Sigmoid -> ×2 - 1. |
| `Softmax` | [⚠️](#methodology-notes) (LUT) | [⚠️](#methodology-notes) (PWL) | ✅ (LUT) | ✅ ([Alg+LUT](#methodology-notes)) | ✅ | ✅ | Softmax function (uses Max-Tree, Exp, Adder-Tree, Reciprocal). |

## Normalization
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Symbolically Verified | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BatchNorm1D` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Inference-only (Scale & Shift via DSP). |
| `LayerNorm1D` | [⚠️](#methodology-notes) | [⚠️](#methodology-notes) | ✅ | ✅ | ✅ | ✅ | Pipelined Adder Tree for Mean/Var, LUT/Alg+LUT for Rsqrt. |

## Pooling Operations
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Symbolically Verified | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `MaxPool1D` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 1D max pooling. |
| `MaxPool2D` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 2D max pooling. *(Requires BRAM)* |
| `AvgPool1D` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 1D average pooling. |
| `AvgPool2D` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 2D average pooling. *(Requires BRAM)* |

## Infrastructure Modules

Non-operator modules (streaming, memory, numeric units) are verified with the same dual approach (Python/Cocotb co-sim + Scala formal proofs).

| Module | Python Co-sim | Symbolically Verified | Notes |
| :--- | :---: | :---: | :--- |
| `StreamDoubleBuffer` | ✅ | ✅ | No data loss / no deadlock under any valid/ready pattern. |
| `DoubleBufferStreamer` | ✅ | ✅ | Double-buffered streaming engine. |
| `DMAReader` | ✅ | ✅ | AXI-style reader (BMC reachability). |
| `DMAReader2D` | ✅ | ✅ | 2D tiled reader over external memory. |
| `Tensor` (pack/unpack) | ✅ | ✅ | Tensor framing + lane packaging contract. |
| `Float` unit (fromSInt) | — | ✅ | `FloatFormal` : SInt -> FloatML conversion (FP8 E4M3), plus `Float.zero`. |
| DTypes (I4/I8/I16/I32/U4/U8/FP4) | ✅ | ✅ | Quantization round-trip per dtype. |
| PWL / Math LUTs | ✅ | ✅ | `PWLFormal`, `math_lutsFormal` (exp/rsqrt/sqrt/reciprocal units). |

## Methodology Notes

* **Alg+LUT**: **Algebraic Separation + LUT** (exact math for exponent, LUT for mantissa). `Log` uses the same idea: `log_b(x) = log2(x) * ln(2)/ln(b)` with a fixed-point `Q8.8` mantissa LUT, a `Q0.16` constant (`ln(2)/ln(b)`), and LZD re-quantization.
* **⚠️**: Supported via PWL or LUT approximation, but mathematically meaningless for unquantized integers (results in 0 or overflow).

