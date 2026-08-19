# SpinalML API Reference

This document provides the API reference for all supported hardware operations in SpinalML. Operations are instantiated either as classes or invoked via convenient object apply methods (e.g., `spinalML.ops.add(a, b)`).

All operations interface via `Tensor[T]`, which embeds a physical hardware `Stream` featuring standard `valid` / `ready` flow control.

## 1. Basic Arithmetic

| Operation | Syntax | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Add** | `add(a, b)` | Element-wise addition. | `a, b: Tensor[T]` | `Tensor[T]` |
| **Sub** | `sub(a, b)` | Element-wise subtraction. | `a, b: Tensor[T]` | `Tensor[T]` |
| **Mul** | `mul(a, b)` | Element-wise multiplication. | `a, b: Tensor[T]` | `Tensor[T]` |
| **Div** | `div(a, b)` | Element-wise division. | `a, b: Tensor[T]` | `Tensor[T]` |
| **BiasAdd** | `bias_add(a, b)` | Broadcast add of a bias vector over the last dimension (columns). | `a, b: Tensor[T]` | `Tensor[T]` |
| **Cast** | `cast(a, dt)` | Casts tensor to a new datatype (SInt -> FloatML, only this direction). | `a: Tensor[T], dt: HardType[U]` | `Tensor[U]` |
| **Requantize** | `requantize(a, dt, shift)` | Shift + saturate to a smaller SInt type (e.g. I32 -> I8). | `a: Tensor[T], dt: HardType[U], shift: Int` | `Tensor[U]` |
| **Abs** | `abs(a)` | Absolute value. | `a: Tensor[T]` | `Tensor[T]` |
| **ScaleAdd** | `scale_add(x, a, b)` | Fused MAC `a * x + b`. | `x, a, b: Tensor[T]` | `Tensor[T]` |

## 2. Non-Linear Mathematics

| Operation | Syntax | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Exp** | `exp(a)` | Exponential (e^x). | `a: Tensor[T]` | `Tensor[T]` |
| **Reciprocal**| `reciprocal(a)` | Inverse (1 / x). | `a: Tensor[T]` | `Tensor[T]` |
| **Rsqrt** | `rsqrt(a)` | Inverse Square Root (1 / sqrt(x)). | `a: Tensor[T]` | `Tensor[T]` |
| **Sqrt** | `sqrt(a)` | Square root. | `a: Tensor[T]` | `Tensor[T]` |

## 3. Matrix Operations

| Operation | Syntax | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **MatMul** | `matmul(a, b)` | Matrix multiplication. Uses BRAM for `b` (weights). | `a, b: Tensor[T]` | `Tensor[T]` |
| **Dot** | `dot(a, b)` | Dot product of two 1D vectors (wraps `matmul` M=1, N=1). Length must be a multiple of `lanes`. | `a, b: Tensor[T]` | `Tensor[T]` |

## 4. Tensor Manipulations

| Operation | Syntax | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Reshape** | `reshape(a, newShape)` | Logical metadata resize. 0 logic cells. | `a: Tensor[T]` | `Tensor[T]` |
| **Flatten** | `flatten(a)` | Logical flatten to 1D. 0 logic cells. | `a: Tensor[T]` | `Tensor[T]` |
| **Repack** | `repack(a, newLanes)` | Physical Gearbox. Adapts stream widths. | `a: Tensor[T]` | `Tensor[T]` |
| **Slice** | `slice(a, start, end)` | Extracts elements along an axis. | `a: Tensor[T]` | `Tensor[T]` |
| **Concat** | `concatenate(a, b, axis)` | Joins two tensors. | `a, b: Tensor[T]` | `Tensor[T]` |
| **Seq2Col** | `seq2col(a, kernel, stride)`| 1D Sliding Window (Shift Register LUT). | `a: Tensor[T]` | `Tensor[T]` |
| **Im2Col** | `im2col(a, kernel, stride)` | 2D Sliding Window using BRAM Line Buffers. | `a: Tensor[T]` | `Tensor[T]` |
| **Transpose** | `transpose(a, perm)` | Permutes dimensions using BRAM. | `a: Tensor[T]` | `Tensor[T]` |

## 5. Neural Network Layers

Neural network layers maintain their own state or weights, and typically present standard Deep Learning input/output signatures.

| Layer | Syntax | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Linear** | `Linear(x, w, b)` | Fully Connected Layer. | `x, w, b: Tensor[T]` | `y: Tensor[T]` |
| **Conv1D** | `Conv1D(x, w, b)` | 1D Convolution. | `x, w, b: Tensor[T]` | `y: Tensor[T]` |
| **Conv2D** | `Conv2D(x, w, b)` | 2D Convolution. | `x, w, b: Tensor[T]` | `y: Tensor[T]` |

## 6. Activations & Normalizations

| Layer | Syntax | Description | Inputs | Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **ReLU** | `relu(a)` | Rectified Linear Unit. | `a: Tensor[T]` | `Tensor[T]` |
| **LeakyReLU** | `leaky_relu(a, alpha)` | Leaky ReLU. | `a: Tensor[T]` | `Tensor[T]` |
| **Softmax** | `Softmax1D(dt, c, L)` | Component: Probabilities over final dimension. | `x: Tensor[T]` | `y: Tensor[T]` |
| **BatchNorm** | `BatchNorm1D(x, g, b)` | Inference-only Folded Scale & Shift. | `x, gamma, beta` | `y: Tensor[T]` |
| **LayerNorm** | `LayerNorm1D(x, g, b)` | Dynamic mean and standard deviation. | `x, gamma, beta` | `y: Tensor[T]` |
| **MaxPool1D** | `maxpool1d(x, k, s)` | 1D Max Pooling (Supports multi-channel). | `x: Tensor[T]` | `y: Tensor[T]` |
| **AvgPool1D** | `avgpool1d(x, k, s)` | 1D Average Pooling (Supports multi-channel). | `x: Tensor[T]` | `y: Tensor[T]` |
