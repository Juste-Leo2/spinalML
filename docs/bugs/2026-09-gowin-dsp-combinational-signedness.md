# Gowin Hardware DSP (MULT18X18) Combinational Signedness Discrepancy — *WORKAROUND VIA `--no-dsp`*

## 1. Summary

During in-circuit neural inference verification on the **Sipeed Tang Primer 20K** (Gowin GW2A-LV18PG256C8/I7), the deployed accelerator (`Universal1DDemo`) produced a negative saturation output `[-128, -128]` (`0x80, 0x80`) instead of the expected golden oracle output `[127, 127]` (`0x7F, 0x7F`), despite passing Verilator simulation with zero mathematical deviation (`0.000`).

The root cause was isolated to Gowin physical silicon behavior under open-source toolchains (Yosys `synth_gowin` + nextpnr-himbaechel + Apycula `gowin_pack`) when DSP multiplier blocks (`MULT18X18`) are synthesized in **purely combinational mode** (`CLK=0`, `AREG=0`, `BREG=0`, `PREG=0`). In this mode, physical fuse mapping for signed multiplication (`ASIGN=1, BSIGN=1`) and register bypass (`IRBY`) in open-source bitstream packers misconfigures the sign extension or internal pipeline muxing, corrupting signed arithmetic on physical silicon.

Bypassing hardware DSP blocks using the `--no-dsp` synthesis flag forces Yosys to map all arithmetic into standard Gowin LUT4 and carry chains (`ALU`), immediately achieving **100% bit-exact hardware inference** (`[127, 127]`) with an Fmax of **101.33 MHz** on the 27 MHz board clock.

---

## 2. Symptoms & Hardware Observations

### Physical Execution on Tang Primer 20K (Default Synthesis with DSPs)
```text
Hardware vs Golden Oracle Inference Comparison
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Logit Index ┃ Hardware Output (Hex) ┃ Hardware Output (Signed I8) ┃ Golden Oracle (Signed I8) ┃ Bit-Exact Match ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│  Logit [0]  │         0x80          │            -128             │            127            │    MISMATCH     │
│  Logit [1]  │         0x80          │            -128             │            127            │    MISMATCH     │
└─────────────┴───────────────────────┴─────────────────────────────┴───────────────────────────┴─────────────────┘
✗ VERIFICATION FAILED: Hardware produced [-128, -128], expected [127, 127].
```

### Mathematical Significance of the Discrepancy
The model architecture ends with:
$$\text{Linear}(6 \to 2) \longrightarrow \text{Requantize}(\text{shift} = 1, \text{targetType} = \text{I8})$$

The `Requantize` layer clips intermediate accumulators to the signed 8-bit integer range $[-128, +127]$:
* When $X > 127$, it clamps to the **positive ceiling** $+127$ (`0x7F`).
* When $X < -128$, it clamps to the **negative floor** $-128$ (`0x80`).

In Verilator simulation:
* `Linear` outputs $+412$ and $+334$.
* Shifted by 1 bit: $+206$ and $+167$.
* Clamped to $+127$ (positive saturation).

On physical hardware with Gowin `MULT18X18`:
* `Linear` outputs values $\le -256$.
* Clamped to $-128$ (negative saturation).

---

## 3. Root Cause Analysis

### 3.1 Synthesis Mapping to `MULT18X18`
In `Linear(inFeatures = 6, outFeatures = 2, customType = Some(I16()))`, the 16-bit signed multiplications are combinational:
```verilog
assign _zz__zz_accumulators_0_1 = ($signed(_zz__zz_accumulators_0_1_1) * $signed(streamDoubleBuffer_7_io_readData_0));
```

Yosys Gowin synthesis (`synth_gowin`) targets Gowin DSP macros (`MULT18X18`):
```json
"soc_acc.model.linearLayer_1.matmulOp_2._zz__zz_accumulators_0_1_MULT18X18_DOUT": {
  "type": "MULT18X18",
  "port_directions": {
    "A": "input", "ASIGN": "input", "B": "input", "BSIGN": "input",
    "CLK": "input", "CE": "input", "RESET": "input", "DOUT": "output"
  },
  "connections": {
    "ASIGN": [ 15 ],  // 1'b1 (signed)
    "BSIGN": [ 15 ],  // 1'b1 (signed)
    "CLK": [ 14 ],    // 1'b0 (no clock)
    "CE": [ 14 ],     // 1'b0 (no clock enable)
    "RESET": [ 14 ]   // 1'b0
  }
}
```

### 3.2 Upstream Bitstream Reverse-Engineering Defect (Apycula)
1. **Recent DSP Support**: Support for Gowin GW2A DSP primitives (`MULT18X18`, `MULT9X9`) was integrated into Yosys and nextpnr in late 2025.
2. **Standard vs. Combinational Use Cases**: Vendor IP cores and traditional HDL designs configure Gowin DSP blocks with pipeline registers enabled (`AREG=1`, `BREG=1`, `PREG=1`) to achieve high clock speeds. The register bypass fuses (`IRBY`, `IRNS`) were imperfectly reverse-engineered in open-source bitstream databases.
3. **Sign Configuration Corrupted**: In physical silicon, unclocked input and output multiplexers inside the DSP block fail to propagate the sign extension bits (`ASIGN`, `BSIGN`) correctly, either treating the MSBs as unsigned or corrupting the two's complement inversion. Consequently, large positive values wrap around to negative numbers.

---

## 4. Verification & Solution: The `--no-dsp` Switch

### 4.1 CLI Implementation
Added the `--no-dsp` command-line option to `spinalml build`:
* In `cli/spinalml_cli/build_runner.py`: appends `-nodsp` to the Yosys synthesis invocation:
  ```python
  if no_dsp:
      synth_cmd_name = f"{synth_cmd_name} -nodsp"
  ```
* In `cli/spinalml_cli/cli.py`: exposes `--no-dsp` in the CLI.

### 4.2 Hardware Test Execution
```bash
# 1. Synthesize using LUT4 logic (zero DSP blocks)
python cli/main.py build rtl --no-dsp

# 2. Flash to volatile SRAM
python cli/main.py flash

# 3. Verify in-circuit over serial link
python scripts/test_hardware_1d.py --port COM8
```

### 4.3 Results with `--no-dsp`
```text
                                  Hardware vs Golden Oracle Inference Comparison
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Logit Index ┃ Hardware Output (Hex) ┃ Hardware Output (Signed I8) ┃ Golden Oracle (Signed I8) ┃ Bit-Exact Match ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│  Logit [0]  │         0x7F          │             127             │            127            │   EXACT MATCH   │
│  Logit [1]  │         0x7F          │             127             │            127            │   EXACT MATCH   │
└─────────────┴───────────────────────┴─────────────────────────────┴───────────────────────────┴─────────────────┘
✓ 100% BIT-EXACT VERIFICATION SUCCESSFUL!
```

* **Resource Usage**: 9,315 LUT4 (44.9% of 20,736 capacity), 0 DSP blocks used.
* **Fmax Achieved**: **101.33 MHz** (significantly exceeding the 27 MHz board constraint).

---

## 5. Architectural Discussion: Future DSP Support & Universality

### Is Pipelining the Multipliers Universal?
Yes, but with design trade-offs:
1. **Universality across FPGA Vendors**:
   * All modern FPGA families (Xilinx 7-Series/UltraScale, Intel Cyclone/Stratix, Lattice ECP5, Gowin GW2A/GW1N, Efinix Titanium) have hard DSP blocks with optional internal registers (`AREG`, `BREG`, `MREG`, `PREG`).
   * When hardware description uses registered multiplication (`val p = RegNext(a * b)`), vendor synthesis tools (Vivado, Quartus, Gowin EDA) automatically absorb the registers into the hard DSP slices without consuming external logic slice flip-flops.
   * On Gowin, this activates standard clocked modes where Apycula fuse packing is proven and robust.

2. **Latency Considerations in SpinalML**:
   * Adding an internal pipeline register to the multiplier increases compute latency by 1 clock cycle.
   * In a pure streaming architecture, introducing this latency requires aligning the valid/ready arbitration and control signals (e.g. via SpinalHDL's `StreamStage` or `m2sPipe`).
   * **Recommendation for SpinalML**: Keep combinational multipliers as the default for small layers, and provide an opt-in parameter (e.g., `pipelinedMultiplier = true` or automatic DSP mapping for large layers) to preserve 100% architectural universality across simulation, soft FPGAs, and ASIC targets.
