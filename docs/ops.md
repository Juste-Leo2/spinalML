# Machine Learning Operations

This document lists the machine learning operations that spinalML aims to support natively in hardware.

## Hardware Memory Management Guidelines
To ensure optimal synthesis on FPGA, operations must follow these memory guidelines:
- **Streaming & Element-wise Ops**: Do not instantiate `Mem`. Use pure combinational logic or simple `Vec(Reg(dataType))` for 1D sliding windows (e.g., `seq2col`, `MaxPool1D`). These map efficiently to Shift Register LUTs (SRLs).
- **Stateful Ops (Large Buffers & 2D Windows)**: When storing large tiles or full tensors (e.g., `matmul` weights, `transpose` buffers, `im2col` line buffers), you MUST use `Mem` with **`readSync`** (synchronous read). This introduces a 1-cycle latency that must be pipelined, but guarantees the synthesizer will infer **Block RAM (BRAM)**. Using `readAsync` forces the synthesizer to use distributed LUTRAM, which consumes massive logic resources and ruins Fmax.

## Basic Arithmetic
- [x] `Add`: Element-wise addition of two tensors.
- [x] `Sub`: Element-wise subtraction of two tensors.
- [x] `Mul`: Element-wise multiplication of two tensors (Hadamard product).
- [ ] `Div`: Element-wise division of two tensors.
- [x] `Exp`: Element-wise exponential. *(Requires BRAM for LUT)*
- [ ] `Log`: Element-wise natural logarithm.
- [x] `Abs`: Element-wise absolute value.

## Matrix and Vector Operations
- [x] `MatMul`: Matrix multiplication of 2D tensors. *(Requires BRAM)*
- [ ] `Dot`: Dot product of two 1D vectors.

## Tensor Manipulations
- [x] `Reshape`: Change the dimensions of a tensor without changing its data.
- [x] `Transpose`: Permute the dimensions of a tensor. *(Requires BRAM)*
- [x] `Concatenate`: Join a sequence of tensors along an existing axis.
- [x] `Slice`: Extract a subset of elements from a tensor.
- [x] `Flatten`: Flatten a multi-dimensional tensor into a 1D tensor.
- [x] `Seq2Col`: Convert a 1D sequence into sliding windows.
- [x] `Im2Col`: Convert a 2D image into sliding windows. *(Requires BRAM)*

## Neural Network Layers
- [x] `Linear` (Dense): Fully connected layer applying a linear transformation to the incoming data. *(Requires BRAM)*
- [x] `Conv1D`: 1D convolution over an input signal composed of several input planes.
- [x] `Conv2D`: 2D convolution over an input image composed of several input planes. *(Requires BRAM)*
- [ ] `DepthwiseConv`: Depthwise separable convolution. *(Requires BRAM)*

## Attention Mechanisms
- [ ] `Classical Attention`: Scaled dot-product attention (Q, K, V).
- [ ] `Multi-Head Attention`: Multiple parallel attention heads.
- [ ] `Mamba2`: Advanced state space models for sequence processing.

## Activation Functions
- [x] `ReLU`: Rectified Linear Unit.
- [x] `LeakyReLU`: Leaky Rectified Linear Unit.
- [ ] `Sigmoid`: Sigmoid activation function.
- [ ] `Tanh`: Hyperbolic tangent activation function.
- [ ] `Softmax`: Softmax function (usually applied over the last dimension).

## Pooling Operations
- [x] `MaxPool1D`: 1D max pooling.
- [ ] `MaxPool2D`: 2D max pooling. *(Requires BRAM)*
- [x] `AvgPool1D`: 1D average pooling.
- [ ] `AvgPool2D`: 2D average pooling. *(Requires BRAM)*

## Normalization
- [ ] `BatchNorm`: Batch Normalization. *(Requires BRAM)*
- [ ] `LayerNorm`: Layer Normalization. *(Requires BRAM)*
