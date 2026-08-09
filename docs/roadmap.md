# spinalML Roadmap

This document outlines the development steps for the spinalML library. Development is organized into sequential priorities. Testing and validation are intrinsically part of every single step.

## 1. Tensor Management and Data Flows
- [x] Define foundational data types (e.g., fixed-point (I8), floating-point (BF16, FP8)).
- [x] Implement the base Tensor hardware representation in SpinalHDL.
- [x] Create memory management and addressing logic for Tensors (Enforcing BRAM inference via `readSync` for large buffers).
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

## 4. System Integration & Advanced Improvements (Future Work)
- [ ] **AXI-DMA Integration**: Develop or integrate a top-level controller to feed AXI4-Stream interfaces directly from external DDR memory.
- [ ] **Advanced Tiling (Matrix A)**: Implement a Write-Back module and advanced tiling logic for massive matrices where partial sums cannot fit entirely in the on-chip accumulators.
- [ ] **Dynamic Padding**: Add hardware or software-side logic to support tensor dimensions that are not perfect multiples of the `lanes` or `tileSize` parameters.
- [ ] **Hardware Adder Tree (Timing Optimization)**: Replace the linear accumulation loop in `MatmulOp` with a logarithmic pipelined Adder Tree to resolve severe combinatorial timing delays and preserve high $F_{max}$.
- [ ] **Floating-Point Pipelining (Retiming)**: Introduce internal pipeline registers inside `FloatML` arithmetic operations (`Add`, `Mul`) to prevent synthesis timing violations.
- [ ] **True Matrix-Matrix Multiplication (GEMM)**: Upgrade the `MatmulOp` from Matrix-Vector (currently restricted by `shapeB(1) == 1`) to full Matrix-Matrix support for batch processing and attention mechanisms.
