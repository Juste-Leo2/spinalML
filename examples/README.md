# SpinalML Application Examples

Welcome to the **SpinalML Examples** directory!

This folder hosts complete, end-to-end Machine Learning hardware projects. Unlike standalone RTL tests, each project in this directory bridges software training with physical silicon deployment.

---

## Standard Example Architecture

Each application subproject follows a self-contained 4-stage pipeline:

```text
examples/<project_name>/
├── train.py            # Python / PyTorch training script
├── export_weights.py   # Quantization (e.g. INT8, wXaY) and weight/bias binary export
├── Model.scala         # Hardware architecture definition in SpinalML (High-Level API)
└── infer_hardware.py   # Host client streaming inputs & reading predictions over UART
```

1. **Software Training (`train.py`)**: Standard deep learning training in PyTorch with quantization-aware training (QAT) or post-training quantization (PTQ).
2. **Weight Serialization (`export_weights.py`)**: Formats weights and biases into binary payloads matching the FPGA memory layout.
3. **Hardware Accelerators (`Model.scala`)**: Synthesizable SpinalML model using the PyTorch-like `Sequential` or `Accelerator` API.
4. **Physical Deployment & Verification (`infer_hardware.py`)**: Uses the UART CSR host bridge to trigger inference on real FPGA silicon and verify bit-exact outputs.

---

## Planned & Upcoming Examples

| Project | Description | Target Hardware | Status |
| :--- | :--- | :--- | :--- |
| **`mnist/`** | Pre-trained handwritten digit recognition (w4a8 / I8 quantized CNN) | Sipeed Tang Primer 20K | *Upcoming* |
| **`cifar10/`** | 2D Convolutional network with BRAM line buffers | Sipeed Tang Primer 20K | *Planned* |
| **`audio_keyword/`** | 1D temporal convolution for keyword spotting | FPGA / Simulation | *Planned* |
| **`tiny_transformer/`** | Multi-head attention sequence classifier | FPGA / Simulation | *Planned* |

---

## Quick Run Guide

Once an example project is selected (e.g. `mnist/`):

```bash
# 1. Synthesize and build bitstream
python cli/main.py build examples/mnist/Model.scala --board tang-primer-20k --no-dsp

# 2. Flash to FPGA SRAM
python cli/main.py flash --board tang-primer-20k

# 3. Run hardware inference
python examples/mnist/infer_hardware.py --port COM8
```
