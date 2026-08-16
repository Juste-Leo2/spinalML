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
- [x] Implement Convolutional layers (1D and 2D) using `seq2col` and `im2col` strategies..
- [x] Implement Activation functions (ReLU, Sigmoid, Tanh).
- [x] Implement Pooling layers (MaxPool, AvgPool).
- [ ] **Multi-Channel Convolutions**: Upgrade `Conv1DLayer` and `Conv2DLayer` hardware implementations to support `inChannels > 1` and `outChannels > 1` (requires cross-channel accumulation).
- [ ] **Multi-Feature Linear Layers**: Upgrade `LinearLayer` to output multiple features instead of hardcoding `outFeatures = 1` (leverage the GEMM matmul).
- [x] Implement Normalization layers (BatchNorm, LayerNorm).
- [x] Test and validate advanced operations, ensuring correct pipeline behavior and throughput.

## 4. System Integration & Advanced Improvements (Future Work)

- [ ] **Advanced Tiling (Matrix A)**: Implement a Write-Back module and advanced tiling logic for massive matrices where partial sums cannot fit entirely in the on-chip accumulators.
- [x] **Dynamic Padding**: Add hardware or software-side logic to support tensor dimensions that are not perfect multiples of the `lanes` or `tileSize` parameters.
- [x] **Hardware Adder Tree (Timing Optimization)**: Replace the linear accumulation loop in `MatmulOp` with a logarithmic pipelined Adder Tree to resolve severe combinatorial timing delays and preserve high $F_{max}$.
- [ ] **Floating-Point Pipelining (Retiming)**: Introduce internal pipeline registers inside `FloatML` arithmetic operations (`Add`, `Mul`) to prevent synthesis timing violations.
- [x] **True Matrix-Matrix Multiplication (GEMM)**: Upgrade the `MatmulOp` from Matrix-Vector (currently restricted by `shapeB(1) == 1`) to full Matrix-Matrix support for batch processing and attention mechanisms.

## 5. Advanced Memory Architecture
- [x] **AXI4 Memory Mapped Master (DDR4)**: Implement an internal DMA controller capable of random addressing to autonomously fetch data and weights from external DDR4 memory.
- [x] **Hardware Tiling & Caching**: Implement automated 2D tiling to split large tensors (e.g. images) that cannot fit in FPGA BRAM, swapping them dynamically with DDR4 (Implemented via `DMAReader2D`).
- [ ] **Weight Manager (Pre-fetching)**: Implement asynchronous double-buffering for network weights, fetching layer $N+1$ from DDR4 while layer $N$ is currently computing.

## 6. High-Level AI Abstraction
- [x] **Automatic Dimension & Bus Management**: Develop a smart compilation pass that automatically deduces output shapes and dynamically inserts `repack` (Gearbox) or `StreamFork` modules to avoid manual hardware wiring.
- [x] **Sequential Model Builder**: Create a PyTorch-like `Sequential` API that hides the underlying AXI4-Stream handshakes and automatically manages weight/bias tensor instantiations.
- [ ] **SoC Integration Testing (Cocotb)**: Implement full system-level testing of the `HighLevelTemplate` (including AXI-Lite and AXI4 memory) using Python, Cocotb, and `cocotbext-axi`, replacing manual Scala simulation.
- [x] **Support of hardware utilities in LayerSpec**: Expose utility layers in the high-level `LayerSpec` API (e.g. `Repack`, `Concat`) to give users manual control over bus widths and complex topologies.
- [x] **Mixed Precision (Dynamic Quantization)**: Implement specific conversion layers (e.g. `Requantize`) in `LayerSpec` to dynamically alter the hardware datapath precision in the middle of a `Sequential` model.
- [ ] **ONNX One-Liner Importer**: Develop a parser that reads an ONNX model file and automatically generates a fully functional SpinalML hardware accelerator in a single line of Scala code.

## 7. Formal Verification (Yosys + SymbiYosys)

Exhaustive verification of hardware blocks via SpinalHDL Formal (`assert`/`assume`/`cover` properties proven for all input combinations by SAT/SMT solvers). Complements the sampled-vector coverage of the Python/Cocotb co-simulation.

- [ ] **Environment**: Install `yosys` + `symbiyosys` (local WSL + CI runner) and validate the SpinalHDL 1.14.2 formal API.
- [ ] **Onboarding Spike**: Write `MulFP8Formal.scala` — prove bit-exact equivalence vs the golden model for all inputs, and inject a deliberate bug to validate that the workflow catches counterexamples. Document the pattern in `docs/formal.md`.
- [ ] **Combinational Ops**: Roll out formal proofs to all primitive ops (`ops/`), dtype by dtype (8/16-bit first), reusing the translated golden models from `tests/python/golden_models/`.
- [ ] **Memory & Flow Invariants**: Prove stream invariants on `StreamDoubleBuffer` (no data loss under any `valid`/`ready` pattern, no deadlock) and reachability checks (`cover`/`bmc`) on `DMAReader`/`DMAReader2D`.
- [ ] **CI Integration & Verification Map**: Add a formal-verification workflow to CI and track module-by-module status (formally proven / simulated / uncovered) in the verification map.
