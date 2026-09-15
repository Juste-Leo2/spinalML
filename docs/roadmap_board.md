# FPGA & Hardware Board Roadmap

This document details the hardware compatibility matrix, tested platforms, and exploratory board targets for **SpinalML**.

---

## 1. Hardware Status Overview

SpinalML aims to remain vendor-agnostic by compiling through open-source toolchains (**Yosys**, **nextpnr**, and **openFPGALoader**) or vendor-native suites.

> [!IMPORTANT]
> **Currently Tested in Physical Silicon: Sipeed Tang Primer 20K**  
> In-circuit synthesis, physical place-and-route, SRAM bitstream flashing, and real-time UART inference are continuously validated on the **Sipeed Tang Primer 20K** (Gowin GW2A-18).  
> The remaining boards in the matrix represent targeted architectures with varying levels of profile support. Contributions and test reports from hardware owners are warmly welcomed!

---

## 2. Hardware Compatibility Matrix

| Vendor | Board | FPGA Device | Toolchain / Programmer | Target Slug | Hardware Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Gowin** | **Sipeed Tang Primer 20K** | GW2A-LV18PG256C8/I7 | Yosys + nextpnr-himbaechel / openFPGALoader | `tang-primer-20k` | ✅ **Validated in Silicon** |
| Gowin | Sipeed Tang Nano 20K | GW2A-LV18QN88 | Yosys + nextpnr-himbaechel / openFPGALoader | `tang-nano-20k` | 🟡 Target (Compatible GW2A) |
| Gowin | Sipeed Tang Nano 9K | GW1NR-LV9QN88PC6/I5 | Yosys + nextpnr-himbaechel / openFPGALoader | `tang-nano-9k` | 🟡 Target (Compact models) |
| Gowin | Sipeed Tang Mega 138K | GW5AST-LV138FPG676A | Yosys / Gowin EDA / openFPGALoader | `tang-mega-138k` | ⚪ Target (Large models) |
| **AMD / Xilinx** | Digilent Arty A7-35T / 100T | Artix-7 (XC7A35T / XC7A100T) | Yosys + nextpnr-xilinx / Vivado | `arty-a7` | 🟡 Target (Priority candidate) |
| AMD / Xilinx | Digilent Basys 3 | Artix-7 (XC7A35T) | Vivado / openFPGALoader | `basys3` | ⚪ Target |
| AMD / Xilinx | Digilent PYNQ-Z2 / Cora Z7 | Zynq-7000 (XC7Z020 / XC7Z010) | Vivado / PYNQ Linux (AXI PS-PL) | `pynq-z2` | ⚪ Target |
| AMD / Xilinx | AMD Kria KV260 / KR260 | Zynq UltraScale+ MPSoC | Vivado / Vitis AI | `kria-kv260` | ⚪ Target |
| **Lattice** | Lattice iCE40 UltraPlus (iCEBreaker) | iCE40UP5K-SG48 | Yosys + nextpnr-ice40 / iceprog | `ice40-up5k` | ⚪ Target (Micro models) |
| Lattice | Lattice ECP5 (Colorlight 5A-75B) | LFE5U-25F / 45F / 85F | Yosys + nextpnr-ecp5 / openFPGALoader | `ecp5` | 🟡 Target (Affordable edge) |
| **Intel / Altera** | Terasic DE10-Lite | MAX 10 (10M50DAF484C7G) | Quartus Prime / openFPGALoader | `de10-lite` | ⚪ Target |
| Intel / Altera | Terasic DE10-Nano | Cyclone V SE (5CSEBA6U23I7) | Quartus Prime / openFPGALoader | `de10-nano` | ⚪ Target |

---

## 3. Contributing a Board Target

Adding support for a new board involves two configuration files in the [`boards/`](../boards/) directory:

1. **Board Profile (`boards/<slug>.json`)**:
   Defines the vendor, FPGA family, device package, toolchain command templates, and default clock frequency.
   ```json
   {
     "name": "My FPGA Board",
     "vendor": "gowin",
     "family": "GW2A-18",
     "device": "GW2A-LV18PG256C8/I7",
     "clock_frequency_hz": 27000000,
     "toolchain": "oss-cad-suite"
   }
   ```

2. **Pin Constraints (`boards/constraints/<slug>.<ext>`)**:
   Physical pin mappings for:
   - System clock input
   - Hardware reset pin (active low or high)
   - UART TX / RX pins (for host communication and inference streaming)
   - Status LEDs (optional, for idle/busy/done visual indicators)

If you have verified SpinalML on a new board, please open a pull request with your board profile and reproduction steps!
