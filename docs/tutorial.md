# spinalML Tutorial

Welcome to the **spinalML** setup tutorial. This guide will help you compile and run your first machine learning hardware component using [SpinalHDL](https://spinalhdl.github.io/SpinalDoc-RTD/) and the **Mill** build tool.

## Prerequisites

Before starting, ensure you have the following installed on your system (this guide assumes a Linux environment like Ubuntu or WSL):

1.  **Java (JDK 8 or later):** Required to run Scala and Mill.
    ```bash
    sudo apt update
    sudo apt install default-jdk
    ```
2.  **Verilator:** A fast, open-source Verilog to C++ compiler, required for simulating SpinalHDL components.
    ```bash
    sudo apt install verilator
    ```
3.  **Mill:** The build tool we use for spinalML. If you don't have it installed globally, you can download the script directly into the project directory:
    ```bash
    curl -L https://github.com/com-lihaoyi/mill/releases/download/0.11.7/0.11.7-assembly -o mill && chmod +x mill
    ```

## Project Structure

The project follows the standard Mill module convention:
*   `build.sc`: The Mill configuration file, managing dependencies (like SpinalHDL core and lib).
*   `spinalML/src/`: Contains the actual hardware descriptions (your Scala code).
*   `spinalML/test/`: Contains the Verilator simulation code (using ScalaTest).

## Running Your First Simulation

To verify that your toolchain works perfectly, you can run the foundational `Tensor` tests we have included (e.g., verifying `Int4` and `Int8` representations).

### 1. Run the Tests
In your terminal, at the root of the project, run:

```bash
./mill spinalML.test
```
*(If you installed mill globally, simply use `mill spinalML.test`)*

Mill will automatically:
1. Download the SpinalHDL dependencies.
2. Compile your Scala hardware code.
3. Generate the Verilog for the hardware components.
4. Launch Verilator to compile the Verilog into C++.
5. Execute the test and verify the tensor shape logic and values.

If the test passes successfully, your environment is ready to start building the next generation of Machine Learning hardware!

If you just want to generate the Verilog file without running the test simulation, you can create an `App` object (e.g., `TensorVerilog`) with `SpinalVerilog(...)` and run:

```bash
./mill spinalML.runMain spinalML.tensors.TensorVerilog
```
This will produce a `.v` file in your directory.
