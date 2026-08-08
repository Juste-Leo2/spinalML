# Machine Learning Operations

This document lists the machine learning operations that spinalML aims to support natively in hardware.

## Basic Arithmetic
- [x] `Add`: Element-wise addition of two tensors.
- [x] `Sub`: Element-wise subtraction of two tensors.
- [x] `Mul`: Element-wise multiplication of two tensors (Hadamard product).
- [ ] `Div`: Element-wise division of two tensors.
- [ ] `Exp`: Element-wise exponential.
- [ ] `Log`: Element-wise natural logarithm.
- [ ] `Abs`: Element-wise absolute value.

## Matrix and Vector Operations
- [x] `MatMul`: Matrix multiplication of 2D tensors.
- [ ] `Dot`: Dot product of two 1D vectors.

## Tensor Manipulations
- [x] `Reshape`: Change the dimensions of a tensor without changing its data.
- [x] `Transpose`: Permute the dimensions of a tensor.
- [x] `Concatenate`: Join a sequence of tensors along an existing axis.
- [x] `Slice`: Extract a subset of elements from a tensor.
- [x] `Flatten`: Flatten a multi-dimensional tensor into a 1D tensor.
- [x] `Seq2Col`: Convert a 1D sequence into sliding windows.
- [x] `Im2Col`: Convert a 2D image into sliding windows.

## Neural Network Layers
- [x] `Linear` (Dense): Fully connected layer applying a linear transformation to the incoming data.
- [x] `Conv1D`: 1D convolution over an input signal composed of several input planes.
- [x] `Conv2D`: 2D convolution over an input image composed of several input planes.
- [ ] `DepthwiseConv`: Depthwise separable convolution.

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
- [ ] `MaxPool2D`: 2D max pooling.
- [x] `AvgPool1D`: 1D average pooling.
- [ ] `AvgPool2D`: 2D average pooling.

## Normalization
- [ ] `BatchNorm`: Batch Normalization.
- [ ] `LayerNorm`: Layer Normalization.
