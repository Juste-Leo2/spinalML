# Machine Learning Operations

This document lists the machine learning operations that spinalML aims to support natively in hardware.

## Hardware Memory Management Guidelines
To ensure optimal synthesis on FPGA, operations must follow these memory guidelines:
- **Streaming & Element-wise Ops**: Do not instantiate `Mem`. Use pure combinational logic or simple `Vec(Reg(dataType))` for 1D sliding windows (e.g., `seq2col`, `MaxPool1D`). These map efficiently to Shift Register LUTs (SRLs).
- **Stateful Ops (Large Buffers & 2D Windows)**: When storing large tiles or full tensors (e.g., `matmul` weights, `transpose` buffers, `im2col` line buffers), you MUST use `Mem` with **`readSync`** (synchronous read). This introduces a 1-cycle latency that must be pipelined, but guarantees the synthesizer will infer **Block RAM (BRAM)**. Using `readAsync` forces the synthesizer to use distributed LUTRAM, which consumes massive logic resources and ruins Fmax.

## Basic Arithmetic
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Add` | ✅ | ✅ | ✅ | ✅ | ❌ | Element-wise addition of two tensors. |
| `Sub` | ✅ | ✅ | ✅ | ✅ | ❌ | Element-wise subtraction of two tensors. |
| `Mul` | ✅ | ✅ | ✅ | ✅ | ❌ | Element-wise multiplication (Hadamard product). |
| `Div` | ✅ | ✅ | ✅ | ✅ | ❌ | Element-wise division (Mul + Reciprocal). |
| `Exp` | ✅ (LUT) | ✅ (PWL) | ✅ (LUT) | ✅ (PWL) | ✅ | Element-wise exponential. |
| `Log` | ❌ | ❌ | ❌ | ❌ | ❌ | Element-wise natural logarithm. |
| `Abs` | ✅ | ✅ | ✅ | ✅ | ❌ | Element-wise absolute value. |
| `sqrt`| ✅ (LUT) | ✅ (PWL) | ✅ (LUT) | ✅ (PWL) | ❌ | Square Root. |
| `reciprocal` | ✅ (LUT) | ✅ (PWL) | ✅ (LUT) | ✅ (PWL) | ❌ | Inverse (1/x). |
| `rsqrt` | ✅ (LUT) | ✅ (PWL) | ✅ (LUT) | ✅ (PWL) | ✅ | Inverse Square Root (1/sqrt(x)). |
| `scale_add` | ✅ | ✅ | ✅ | ✅ | ❌ | Fast Fused MAC (A*X + B). Mapped to DSP48. |

## Matrix and Vector Operations
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `MatMul` | ✅ | ✅ | ✅ | ✅ | ❌ | Matrix multiplication of 2D tensors. *(Requires BRAM)* |
| `Dot` | ❌ | ❌ | ❌ | ❌ | ❌ | Dot product of two 1D vectors. |

## Tensor Manipulations
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Reshape` | ✅ | ✅ | ✅ | ✅ | ❌ | Change the dimensions of a tensor without changing its data. |
| `Transpose` | ✅ | ✅ | ✅ | ✅ | ❌ | Permute the dimensions of a tensor. *(Requires BRAM)* |
| `Concatenate` | ✅ | ✅ | ✅ | ✅ | ❌ | Join a sequence of tensors along an existing axis. |
| `Slice` | ✅ | ✅ | ✅ | ✅ | ❌ | Extract a subset of elements from a tensor. |
| `Flatten` | ✅ | ✅ | ✅ | ✅ | ❌ | Flatten a multi-dimensional tensor into a 1D tensor. |
| `Seq2Col` | ✅ | ✅ | ✅ | ✅ | ❌ | Convert a 1D sequence into sliding windows. |
| `Im2Col` | ✅ | ✅ | ✅ | ✅ | ❌ | Convert a 2D image into sliding windows. *(Requires BRAM)* |

## Neural Network Layers
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Linear` (Dense) | ✅ | ✅ | ✅ | ✅ | ❌ | Fully connected layer. *(Requires BRAM)* |
| `Conv1D` | ✅ | ✅ | ✅ | ✅ | ❌ | 1D convolution over an input signal. |
| `Conv2D` | ✅ | ✅ | ✅ | ✅ | ❌ | 2D convolution over an input image. *(Requires BRAM)* |
| `DepthwiseConv` | ❌ | ❌ | ❌ | ❌ | ❌ | Depthwise separable convolution. *(Requires BRAM)* |

## Attention Mechanisms
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Classical Attention` | ❌ | ❌ | ❌ | ❌ | ❌ | Scaled dot-product attention (Q, K, V). |
| `Multi-Head Attention`| ❌ | ❌ | ❌ | ❌ | ❌ | Multiple parallel attention heads. |
| `Mamba2` | ❌ | ❌ | ❌ | ❌ | ❌ | Advanced state space models for sequence processing. |

## Activation Functions
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `ReLU` | ✅ | ✅ | ✅ | ✅ | ❌ | Rectified Linear Unit. |
| `LeakyReLU` | ✅ | ✅ | ✅ | ✅ | ❌ | Leaky Rectified Linear Unit. |
| `Sigmoid` | ❌ | ❌ | ❌ | ❌ | ❌ | Sigmoid activation function. |
| `Tanh` | ❌ | ❌ | ❌ | ❌ | ❌ | Hyperbolic tangent activation function. |
| `Softmax` | ✅ | ✅ | ✅ | ✅ | ❌ | Softmax function (uses Max-Tree, Exp, Adder-Tree, Reciprocal). |

## Normalization
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `BatchNorm1D` | ✅ | ✅ | ✅ | ✅ | ❌ | Inference-only (Scale & Shift via DSP). |
| `LayerNorm1D` | ✅ | ✅ | ✅ | ✅ | ❌ | Pipelined Adder Tree for Mean/Var, LUT/PWL for Rsqrt. |

## Pooling Operations
| Operation | I4 / I8 | I16 / I32 | FP4 / FP8 | BF16 / FP32 | Math Validated | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `MaxPool1D` | ✅ | ✅ | ✅ | ✅ | ❌ | 1D max pooling. |
| `MaxPool2D` | ❌ | ❌ | ❌ | ❌ | ❌ | 2D max pooling. *(Requires BRAM)* |
| `AvgPool1D` | ✅ | ✅ | ✅ | ✅ | ❌ | 1D average pooling. |
| `AvgPool2D` | ❌ | ❌ | ❌ | ❌ | ❌ | 2D average pooling. *(Requires BRAM)* |
