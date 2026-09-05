# spinalML

spinalML is a Machine Learning library designed for hardware synthesis and simulation, written in Scala using [SpinalHDL](https://spinalhdl.github.io/SpinalDoc-RTD/). This library aims to provide a robust foundation for implementing neural networks and other machine learning algorithms directly in hardware, leveraging the power and expressiveness of SpinalHDL.

## Overview

The core objective of spinalML is to provide a complete and tested set of tools to describe ML hardware accelerators. The library is structured around several key concepts:

*   **Tensors and Data Flows:** Fundamental representations of multi-dimensional data and the streams used to move this data through the hardware architecture.
*   **Basic Operations:** Essential mathematical and matrix operations required for any machine learning algorithm.
*   **Advanced Operations:** Complex layers such as convolutions, activation functions, and pooling mechanisms.

## Quick Start (with `uv`)

We recommend using [**uv**](https://docs.astral.sh/uv/) for ultra-fast, modern Python environment setup:

```bash
# 1. Install uv
# Linux / macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Setup Python environment
uv venv -p 3.11
# Linux/macOS: source .venv/bin/activate | Windows: .\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# 3. Setup hardware toolchain (Mill, Verilator, SymbiYosys, Yosys)
python cli/main.py setup

# 4. Verify & test hardware circuits
python cli/main.py test spinalML/src/spinalML/examples/Mnistw4a8.scala
python cli/main.py test-all          # 75 ScalaTest dynamic simulations
python cli/main.py test-all-formal   # 56 SymbiYosys formal proofs
python cli/main.py test-all-python   # Python/Cocotb co-simulations
```

For detailed usage and advanced options, see the [CLI Reference](docs/cli.md).

## Project Documentation

To understand the direction and the capabilities of this project, please refer to the following documents:

*   [CLI Reference](docs/cli.md): Complete guide to the SpinalML command-line interface, compilation, simulation, and formal verification on Linux & Windows.
*   [Getting Started](docs/getting_started.md): A tutorial to understand core concepts, tensors, streams, and how to build your first ML hardware layers.
*   [High-Level Tutorial](docs/HighLevelTutorial.md): The complete guide to the PyTorch-like API: layer catalog, wXaY quantization, memory layout and SoC simulation.
*   [Operations Support](docs/opsSupport.md): A comprehensive matrix of all machine learning operations and their hardware validation status.
*   [API Reference](docs/opsDocs.md): The detailed technical documentation and signatures for all supported hardware operations.
*   [Roadmap](docs/roadmap.md): The step-by-step checklist of the project's development phases.
*   [Python Co-Simulation](docs/pythonTest.md): Documentation on the Python/Cocotb/Verilator testing framework used to validate mathematical hardware models.
*   [Formal Verification](docs/symbolicTest.md): Documentation on the symbolic tests and formal verification infrastructure using SymbiYosys.

## Acknowledgements

Special thanks to the SpinalHDL project for providing the incredible framework that makes this library possible. You can find their repository here: [SpinalHDL GitHub Repository](https://github.com/SpinalHDL/SpinalHDL).

## AI Usage Policy

I heavily use AI tools for programming, and using AI to contribute to this project is completely fine by me. The primary focus for spinalML is the quality, correctness, and seamless integration of the code, regardless of whether it was authored by a human or an AI. 

However, there is one strict rule: **Pull Requests must be initiated by a human.** 
The PR description must be written by you (a human), and you must demonstrate a minimum understanding of the objective and the technical elements that made the implementation successful.

## Citation

If you use spinalML in your research, university courses, or hardware projects, please consider citing it:

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

## Author

* **Léonard Adamo** ([@Juste-Leo2](https://github.com/Juste-Leo2))
  * Student at Université de Montpellier, France
  * Creator of spinalML

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 Léonard Adamo (Juste-Leo2).
