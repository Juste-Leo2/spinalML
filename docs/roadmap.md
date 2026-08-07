# spinalML Roadmap

This document outlines the development steps for the spinalML library. Development is organized into sequential priorities. Testing and validation are intrinsically part of every single step.

## 1. Tensor Management and Data Flows
- [x] Define foundational data types (e.g., fixed-point, floating-point representations if applicable).
- [x] Implement the base Tensor hardware representation in SpinalHDL.
- [x] Create memory management and addressing logic for Tensors.
- [x] Implement data flow interfaces (like AXI-Stream) for input and output data streaming.
- [x] Test and validate memory access patterns and stream handshaking.

## 2. Basic Operations
- [x] Implement element-wise arithmetic (Addition, Subtraction, Multiplication).
- [x] Implement scalar operations (Broadcast add/mul).
- [x] Implement basic Matrix Multiplication (MatMul). (Accumulator data type selection and Tiling/Double-Buffering system for large matrices are complete)
- [ ] Implement Dot Product for 1D Tensors.
- [ ] Test and validate all basic operations for accuracy and hardware synthesis efficiency.

## 3. Advanced Operations
- [x] Implement Dense (Linear) layers.
- [x] Implement Convolutional layers (1D and 2D) using `seq2col` and `im2col` strategies.
- [x] Implement Activation functions (ReLU, Sigmoid, Tanh).
- [x] Implement Pooling layers (MaxPool, AvgPool).
- [ ] Implement Normalization layers (BatchNorm, LayerNorm).
- [ ] Test and validate advanced operations, ensuring correct pipeline behavior and throughput.
