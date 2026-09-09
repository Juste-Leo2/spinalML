# Contributing to SpinalML

Thank you for your interest in contributing to **SpinalML**! We welcome contributions from hardware designers, software engineers, machine learning practitioners, and open-source enthusiasts.

---

## Generative AI & Contribution Philosophy

We embrace modern development workflows: **the use of Generative AI tools (e.g. Gemini, DeepSeek, Claude, ChatGPT, Copilot, Antigravity, OpenCode, or autonomous coding agents) is completely allowed and encouraged.**

Building complex machine learning hardware architectures in SpinalHDL requires multidisciplinary knowledge across RTL, formal verification, and deep learning pipelines. AI assistants can significantly accelerate implementation and debugging.

However, to maintain the integrity, stability, and correctness of SpinalML, all contributions must strictly adhere to the following three rules:

### 1. Pull Requests Must Be Initiated and Described by a Human
- The Pull Request must be submitted by a human contributor.
- The PR description must be written by you, clearly summarizing:
  - The problem or feature addressed.
  - The architectural approach and design decisions.
  - The hardware implications (e.g. resource utilization, pipeline stages, timing impact).
- **Blind automated dumps or unreviewed AI outputs without human understanding will be closed immediately.** You must understand the code you submit and be able to discuss technical details in review.

### 2. Mandatory Manual Testing with Proof of Results
- Hardware code cannot be merged without verification. Every PR modifying RTL, CLI logic, or Python co-simulations must include verifiable test evidence in the PR description:
  - Output logs from ScalaTest simulation (`python cli/main.py test <test_target>`).
  - Output logs from formal verification (`python cli/main.py test-all-formal`) if applicable.
  - Output logs from Python/Cocotb golden model tests (`python cli/main.py test-all-python`).
  - Synthesis / PnR logs (`python cli/main.py build ...`) or physical FPGA test logs if hardware features are added.

### 3. Architectural Consistency & Code Quality
- Follow SpinalHDL idioms and existing codebase patterns:
  - Strong typing with `HardType` and parameterization over data types.
  - Proper streaming contracts: always respect `valid` / `ready` flow control semantics.
  - Self-documenting signal names and modular component structuring.
  - When introducing a new operation or layer, document its API in [`docs/opsDocs.md`](docs/opsDocs.md).

---

## How to Contribute Step-by-Step

### 1. Set Up Your Environment
Follow the instructions in the [README](README.md#quick-start-with-uv) to install `uv` and provision the hardware toolchain:

```bash
# Clone your fork
git clone https://github.com/<your-username>/spinalML.git
cd spinalML

# Setup virtualenv and dependencies
uv venv -p 3.11
# Windows: .\.venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
uv pip install -r requirements.txt

# Provision hardware toolchain (Mill, Verilator, Yosys, nextpnr)
python cli/main.py setup
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Run Existing Verification Suites
Ensure everything passes cleanly before introducing changes:
```bash
python cli/main.py test-all          # Runs ScalaTest test suite
python cli/main.py test-all-python   # Runs Cocotb / Verilator golden model co-simulations
```

### 4. Implement and Test Your Changes
- Write unit tests for your new component or layer (in `spinalML/test/src/spinalML/`).
- Validate that the simulation matches expectations.
- If relevant, test bitstream generation:
  ```bash
  python cli/main.py build <path_to_example.scala> --board tang-primer-20k --no-dsp
  ```

### 5. Submit Your Pull Request
Open a PR against the `main` branch with:
- A clear, descriptive title.
- A concise explanation of the change and rationale.
- Terminal outputs or screenshots proving that tests passed successfully.

---

Thank you for helping make open-source hardware machine learning accessible and performant!
