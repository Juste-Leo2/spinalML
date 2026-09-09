<div align="center">

# SpinalML

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()
[![CI - Verification](https://github.com/Juste-Leo2/spinalML/actions/workflows/ci-simulations.yml/badge.svg?branch=main)](https://github.com/Juste-Leo2/spinalML/actions/workflows/ci-simulations.yml)
[![CI - Sentrux](https://github.com/Juste-Leo2/spinalML/actions/workflows/ci-sentrux.yml/badge.svg?branch=main)](https://github.com/Juste-Leo2/spinalML/actions/workflows/ci-sentrux.yml)
<!-- Logo placeholder: <img src="docs/assets/logo.png" width="180" alt="SpinalML Logo" /> -->

**High-Level Machine Learning Hardware Accelerators in Scala (SpinalHDL)**

*PyTorch-like developer ergonomics • Zero-to-Silicon automated flow • Bit-exact physical FPGA inference*

</div>

---

SpinalML is a hardware Machine Learning acceleration library designed for FPGA synthesis and simulation, written in Scala using [SpinalHDL](https://spinalhdl.github.io/SpinalDoc-RTD/). It bridges the gap between high-level deep learning model design and silicon RTL, enabling developers and researchers to describe neural networks with a PyTorch-like API while generating production-grade, pipelined, cycle-accurate hardware.

> [!NOTE]
> **In Development**: SpinalML is currently under active development and research. Hardware primitives, quantization pipelines, and board targets are continuously evolving.

---

## Quick Start (with `uv`)

We recommend using [**uv**](https://docs.astral.sh/uv/) for ultra-fast, modern Python environment setup.

### 1. Install `uv`
- **Linux / macOS**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. Setup Environment & Toolchain
```bash
# Clone the repository
git clone https://github.com/Juste-Leo2/spinalML.git
cd spinalML

# Setup Python virtual environment
uv venv -p 3.12.1
# Linux/macOS: source .venv/bin/activate | Windows: .\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# Provision hardware toolchain (Mill, Verilator, Yosys, nextpnr, SymbiYosys, openFPGALoader)
python cli/main.py setup
```

Once provisioned, explore ready-to-use application examples in the [`examples/`](examples/) directory, including the complete [MNIST hardware demo](examples/Mnist/README.md).

### 3. Compile from Scala to Silicon in One Command
SpinalML provides a turnkey flow that translates your Scala model, synthesizes RTL, places-and-routes, and generates a bitstream:

```bash
# Build for Sipeed Tang Primer 20K (Gowin GW2A-18)
python cli/main.py build tests/universal/Universal1DDemo.scala --board tang-primer-20k --no-dsp
```

> [!IMPORTANT]
> **Why `--no-dsp` is required on Gowin Tang Primer 20K:**
> The open-source Gowin synthesis flow (Yosys/Apycula) infers unpipelined hard DSP blocks in combinational mode by default, which can cause timing and signedness issues. Specifying `--no-dsp` forces arithmetic operations into FPGA LUTs, ensuring **100% bit-exact hardware results** (see [Gowin DSP Bug Report](docs/bugs/2026-09-gowin-dsp-combinational-signedness.md)). Universal pipelined DSP mapping is on the project roadmap.

### 4. Flash and Run Physical Hardware Inference
```bash
# Flash bitstream to SRAM
python cli/main.py flash hw_build/top.fs --board tang-primer-20k

# Run real-time physical inference over UART
python scripts/test_hardware_1d.py --port COM8
```

---

## Real-World Application Examples

To see what you can build and run on physical silicon with SpinalML, check out the [`examples/`](examples/) directory:

- [**MNIST Hardware Accelerator on Tang Primer 20K**](examples/Mnist/README.md): A complete end-to-end demonstration featuring an ultra-compact mixed-precision (W4A8) CNN accelerator running in real time on the Gowin GW2A-18, paired with an interactive Gradio web drawing canvas and live UART telemetry.

---

## Beginner's Corner

SpinalML abstracts away tedious hardware handshakes. You can construct neural networks with high-level declarative components:

- **`Tensor[T]`**: Multi-dimensional tensor abstraction embedding physical hardware `Stream` interfaces (`valid` / `ready` handshakes).
- **Automated Pipelining**: Layers handle FIFO buffering, dimensional broadcasting, and backpressure automatically.
- **Sequential API**: Define your model cleanly using a PyTorch-like layer list.

### Minimal 1D Neural Network Example

```scala
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

case class TinyMLP(
  override val axiConfig: Axi4Config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
) extends Accelerator(
  dataType = I8(),             // 8-bit integer quantization
  inputShape = Seq(16, 1),     // 1D input vector of length 16
  modelSpec = Seq(
    Linear(inFeatures = 16, outFeatures = 32),
    ReLU(),
    Linear(inFeatures = 32, outFeatures = 4)
  ),
  axiConfig = axiConfig
)
```

No boilerplate `App` object or manual Verilog runner needed: the SpinalML CLI automatically elaborates your class, wraps it in the SoC bus, and compiles it directly:
```bash
python cli/main.py build TinyMLP.scala --board tang-primer-20k --no-dsp
```

---

## Advanced & Data Scientist Corner

For complex workloads and production silicon, SpinalML provides state-of-the-art accelerators and quantization techniques:

- **Transformers & Attention Mechanisms**:
  - `ClassicalAttention(embedDim, numHeads)`: Multi-Head Attention blocks with batched streaming matrix multiplications (`matmul`) and hardware-accelerated Softmax.
- **Mixed Precision & Weight-Only Quantization (wXaY)**:
  - Run activations in floating-point (`FP8`, `BF16`) with compact integer weights (`I4`, `I8`) using `customWeightType` and compile-time `weightScales`.
  - On-the-fly precision adaptation via `Requantize(shift, targetType)` and `Cast(targetType)`.
- **2D Computer Vision**:
  - `Conv2D`, `MaxPool2D`, and `AvgPool2D` backed by BRAM-based line buffers (`Mem` + `readSync`) with zero CPU overhead.
  - Normalization layers: `LayerNorm` and inference-folded `BatchNorm`.
- **SoC & Memory Infrastructure**:
  - Autonomous AXI4 DMA memory engines (`DMAReader`, `DMAReader2D`) with `StreamDoubleBuffer` to overlap memory transfers with compute.
  - Hardware CSR bridge with UART packet framing for direct host-to-FPGA streaming inference.
- **Hardware LUT & Area Optimization Knobs**:
  - **`lanes` & `Repack(newLanes)`**: Bus streaming width (`lanes`) is automatically inferred per layer, but can be manually adapted via `Repack` to scale parallelism up or down to save routing LUTs.
  - **`temporal` Accumulator Streaming**: Matrix-multiplication layers accumulate full $M \times N$ output tables in registers by default (`temporal = 0`). Setting `temporal > 0` in `Accelerator(..., temporal = N)` drains completed rows on-the-fly, reclaiming thousands of LUTs and flip-flops on area-constrained FPGAs.

---

## Documentation Index

Explore detailed guides in the [`docs/`](docs/) directory:

- [**Getting Started Guide**](docs/getting_started.md): Core concepts, tensors, streams, and your first hardware layers.
- [**High-Level Tutorial**](docs/HighLevelTutorial.md): Complete guide to the PyTorch-like API, quantization, and DAG topologies.
- [**Operations API Reference**](docs/opsDocs.md): Detailed specifications of all supported hardware operations and layers.
- [**CLI Reference**](docs/cli.md): Commands for compilation, simulation, synthesis, formal verification, and flashing.
- [**Project Structure**](docs/project_structure.md): Repository layout, conventions, and architectural roadmap.
- [**UART Bridge & Protocol**](docs/uart_bridge.md): Specifications for physical UART host communication and CSR bridges.
- [**Application Examples**](examples/): Silicon-validated hardware projects, including the [Tang Primer 20K MNIST Accelerator](examples/Mnist/README.md).

---

## Contributing

Contributions are welcome! Please read [**CONTRIBUTING.md**](CONTRIBUTING.md) before submitting a pull request.

**Generative AI Policy**: The use of generative AI tools (LLMs, code assistants, autonomous agents) is fully permitted and encouraged. However, every Pull Request must be initiated and described by a human who understands the changes, and code must be manually verified with attached test results.

---

## Acknowledgements

- **[SpinalHDL](https://github.com/SpinalHDL/SpinalHDL)**: For providing an unmatched, expressive hardware description language.
- **The [Yosys](https://github.com/YosysHQ/yosys) & Open-Source FPGA Community**: For building phenomenal synthesis, PnR, and bitstream tools (`yosys`, `nextpnr`, `openFPGALoader`).
- **AI Collaborators**:
  - **Gemini Flash & Pro** (via Antigravity)
  - **DeepSeek Flash & Pro** (via OpenCode)
  - **GLM 5.3 Flash** (via OpenCode)

---

## Citation

If you use SpinalML in your research or hardware projects, please cite:

```bibtex
@software{adamo2026spinalml,
  author       = {Adamo, Léonard},
  title        = {{spinalML: Hardware Machine Learning Accelerators with SpinalHDL}},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Juste-Leo2/spinalML}},
  note         = {Student project, Université de Montpellier, France}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 Léonard Adamo.
