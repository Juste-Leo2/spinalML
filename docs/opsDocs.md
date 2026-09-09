# SpinalML Operations Reference

This document provides a concise reference for all hardware operations and neural network layers available in **SpinalML**.

All operations process data through the `Tensor[T]` abstraction, which wraps hardware streams with standard `valid` / `ready` handshakes.

---

## 1. Basic Arithmetic & Element-wise Ops

| Operation | Syntax | Description | Hardware Notes |
| :--- | :--- | :--- | :--- |
| **Add / Sub** | `add(a, b)` / `sub(a, b)` | Element-wise addition and subtraction | Fully pipelined, saturating or wrapping |
| **Mul** | `mul(a, b)` | Element-wise multiplication | Inferred multipliers (LUT or pipelined DSP) |
| **Div** | `div(a, b)` | Element-wise division | Radix-2 / iterative shift-divider |
| **Abs** | `abs(a)` | Absolute value | Combinational sign inversion |
| **BiasAdd** | `bias_add(a, bias)` | Broadcast add of a 1D vector over the inner dimension | Zero-overhead streaming broadcast |
| **ScaleAdd** | `scale_add(x, a, b)` | Fused MAC `a * x + b` | Single-cycle fused arithmetic |
| **CumSum** | `cumsum(a)` | Cumulative sum along the sequence axis | Used in Linear Attention & Mamba architectures |

---

## 2. Non-Linear & Transcendental Functions

Non-linear functions for `FloatML` use piece-wise linear (PWL) and algebraic separation for near-zero approximation error without hardware multipliers:

| Operation | Syntax | Description | Hardware Notes |
| :--- | :--- | :--- | :--- |
| **Exp** | `exp(a)` | Exponential $e^x$ | Algebraic range reduction + PWL LUT |
| **Log** | `log(a, base)` | Natural log ($\ln$) or $\log_{10}$ ($x \le 0 \to 0$) | Range-reduced logarithmic table |
| **Sqrt** | `sqrt(a)` | Square root $\sqrt{x}$ | Shift-and-subtract / PWL |
| **Rsqrt** | `rsqrt(a)` | Inverse square root $1/\sqrt{x}$ | Essential for LayerNorm and attention scaling |
| **Reciprocal** | `reciprocal(a)` | Multiplicative inverse $1/x$ | Used in Softmax and division pipelines |

---

## 3. Matrix Operations & Attention

| Operation | Syntax | Description | Hardware Notes |
| :--- | :--- | :--- | :--- |
| **MatMul** | `matmul(a, b)` | Streaming matrix multiplication ($M \times K \times N$) | BRAM storage for weights, batched streaming |
| **Dot** | `dot(a, b)` | Vector dot-product | Inner MAC accumulator |
| **ClassicalAttention** | `ClassicalAttention(embedDim, numHeads)` | Multi-Head Attention block | Batched MatMul + scaled Softmax pipeline |

---

## 4. Tensor Shape & Gearbox Manipulations

| Operation | Syntax | Description | Hardware Notes |
| :--- | :--- | :--- | :--- |
| **Reshape** | `reshape(a, newShape)` | Reinterprets tensor dimensions | **0 logic cells** (metadata only) |
| **Flatten** | `flatten(a)` | Flattens tensor into a 1D vector | **0 logic cells** (metadata only) |
| **Repack** | `repack(a, newLanes)` | Adapts physical streaming width (gearbox) | FIFO / Shift-register bit re-packing |
| **Slice** | `slice(a, start, end)` | Extracts elements along a specified axis | Stream filtering with flow control |
| **Concat** | `concatenate(a, b, axis)` | Joins two tensors along an axis | Sequential streaming multiplexer |
| **Transpose** | `transpose(a, perm)` | Permutes tensor axes | Ping-pong BRAM memory reordering |
| **Seq2Col** | `seq2col(a, kernel, stride)` | 1D sliding window extraction | Shift register LUT buffer |
| **Im2Col** | `im2col(a, kernel, stride)` | 2D sliding window extraction | Dual BRAM line buffers |

---

## 5. Neural Network Layers

| Layer | High-Level Syntax | Low-Level Syntax | Description |
| :--- | :--- | :--- | :--- |
| **Linear** | `Linear(inFeatures, outFeatures)` | `Linear(x, w, b)` | Fully connected dense layer. Supports weight-only quantization (`customWeightType = Some(I8())`) |
| **Conv1D** | `Conv1D(inChannels, outChannels, k)` | `Conv1D(x, w, b)` | 1D temporal convolution |
| **Conv2D** | `Conv2D(inChannels, outChannels, k)` | `Conv2D(x, w, b)` | 2D spatial convolution backed by BRAM line buffers |

---

## 6. Activations, Poolings & Normalizations

| Component | High-Level Syntax | Description | Hardware Implementation |
| :--- | :--- | :--- | :--- |
| **ReLU** | `ReLU()` | $\max(0, x)$ | Sign-bit masking (0 LUTs) |
| **LeakyReLU** | `LeakyReLU(alpha)` | $\max(\alpha x, x)$ | Conditional arithmetic shift |
| **Sigmoid** | `Sigmoid()` | $1 / (1 + e^{-x})$ | Pipelined Exp $\to$ Add 1 $\to$ Reciprocal |
| **Tanh** | `Tanh()` | $2\sigma(2x) - 1$ | Fused Sigmoid composition |
| **Softmax** | `Softmax1D()` | Normalizes inputs to probability distribution | Streaming Exp + accumulator + Reciprocal |
| **BatchNorm** | `BatchNorm()` | Batch normalization | Folded scale & shift for inference |
| **LayerNorm** | `LayerNorm()` | Dynamic mean & variance normalization | Online 2-pass accumulator + Rsqrt |
| **MaxPool1D / 2D** | `MaxPool1D(k, s)` / `MaxPool2D(k, s)` | Maximum value pooling | Shift register / BRAM line buffer |
| **AvgPool1D / 2D** | `AvgPool1D(k, s)` / `AvgPool2D(k, s)` | Average value pooling | Shift-based division (power-of-2 kernel) |

---

## 7. Quantization & Cast Operations

| Operation | Syntax | Description |
| :--- | :--- | :--- |
| **Requantize** | `Requantize(shift, targetType)` | Arithmetic right shift + saturation to smaller integer type (e.g. `I32` $\to$ `I8`) |
| **Cast** | `Cast(targetType)` | Converts integer representations into floating-point (`FloatML`) domain |
