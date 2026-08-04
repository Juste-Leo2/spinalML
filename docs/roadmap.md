# spinalML Roadmap

This document outlines the development steps for the spinalML library. Development is organized into sequential priorities. Testing and validation are intrinsically part of every single step.

## 1. Tensor Management and Data Flows
- [ ] Define foundational data types (e.g., fixed-point, floating-point representations if applicable).
- [ ] Implement the base Tensor hardware representation in SpinalHDL.
- [ ] Create memory management and addressing logic for Tensors.
- [ ] Implement data flow interfaces (like AXI-Stream) for input and output data streaming.
- [ ] Test and validate memory access patterns and stream handshaking.

## 2. Basic Operations
- [ ] Implement element-wise arithmetic (Addition, Subtraction, Multiplication).
- [ ] Implement scalar operations (Broadcast add/mul).
- [ ] Implement basic Matrix Multiplication (MatMul).
- [ ] Implement Dot Product for 1D Tensors.
- [ ] Test and validate all basic operations for accuracy and hardware synthesis efficiency.

## 3. Advanced Operations
- [ ] Implement Dense (Linear) layers.
- [ ] Implement Convolutional layers (1D and 2D).
- [ ] Implement Activation functions (ReLU, Sigmoid, Tanh).
- [ ] Implement Pooling layers (MaxPool, AvgPool).
- [ ] Implement Normalization layers (BatchNorm, LayerNorm).
- [ ] Test and validate advanced operations, ensuring correct pipeline behavior and throughput.
