# UART Bridge — L2 Protocol & SoC Architecture

Reference specification for the spinalML accelerator UART communication link: L2 data link protocol, memory mapping, timing, and integration with the host driver (`spinalml_cli/uart_host.py`).

## 1. Overview

The SoC embeds the neural accelerator (AXI master read + CSR AXI-Lite control bus + FP8/INT8 output stream) interfaced to a physical UART communication link:

```
                ┌─────────────────────────────────────────────────────┐
 UART RX ──► [UartRx] ──► [Bridge L2] ──┬─► CSR AXI-lite (write) ──► Accelerator
                │                       ├─► BRAM Memory 4096×64 (write)
                │                       └─► (read logits via outStream)
 UART TX ◄── [UartTx] ◄─────────────────┘
                                  Accelerator ──► AXI Master read ──► [AxiReadMem]
```

- All memory (input activations, network weights) is preloaded over UART into an internal BRAM of **4096 words of 64 bits (32 KiB)**.
- The accelerator reads this memory via its AXI4 master interface; `AxiReadMem` serves these read bursts. Memory is read-only for the accelerator datapath, while writes come exclusively from the UART host.

## 2. Parameters & Timing

| Parameter | Default Value | Configurable |
|---|---|---|
| `CLK_FREQ` | 27,000,000 Hz | Yes (`clkFreq`) |
| `BAUD_RATE` | 115,200 baud | Yes (`baudRate`) |
| `CLK_PER_BIT` = `CLK_FREQ / BAUD_RATE` | 234 | Derived automatically |
| Internal Reset | Active High (external `reset_n` active low) | Yes |
| Byte Order | **LSB-first** (Little-Endian) | Protocol standard |
| Output Format | FP8: `{sign(1), exponent(4), mantissa(3)}` or INT8 | Resized to 8 bits |

**Physical RX Sampling**: The start bit is validated at mid-bit period (`CLK_PER_BIT / 2`), data bits are sampled near the end of the period (`CLK_PER_BIT - 1`), and the stop bit is confirmed before framing completion.

## 3. Protocol Specification

The bridge state machine awaits a 1-byte command opcode in `IDLE`, followed by the corresponding command payload:

| Command | Opcode | Payload | Response | Description |
|---|---|---|---|---|
| `C` CSR write | `0x43` | 4 bytes addr + 4 bytes val | Handshake B | Writes a 32-bit CSR register (`addr[7:0]`) over AXI-Lite |
| `W` Memory write | `0x57` | 4 bytes addr + 4 bytes len + N bytes data | — | Writes `len` bytes into BRAM starting at the virtual address |
| `R` Read logits | `0x52` | — | `outCount` bytes | Drains accelerator output stream to UART TX with backpressure |
| `S` Status | `0x53` | — | 1 byte | Returns live SoC execution and bus activity flags |
| `V` Version | `0x56` | — | 1 byte (`0x01`) | Returns the bridge protocol version |

### 3.1 CSR Write (`C`)

1. 4 bytes of address (LSB-first, lower 8 bits `addr[7:0]` decoded).
2. 4 bytes of value (LSB-first).
3. The bridge issues an AXI-Lite write transaction `(aw_addr, w_data, strb=1111)` and waits for the `B` channel handshake.
4. Core CSR registers:
   - `0x00`: Start pulse (`cmd = 0x01`)
   - `0x08`: Input image base address (`0x10000`)
   - `0x0C`: Weight base address (`0x20000`)

### 3.2 Memory Write (`W`)

1. 4 bytes of start address (LSB-first, virtual address).
2. 4 bytes of payload length (LSB-first): total number of data bytes to write.
3. $N$ data bytes, packed sequentially into 64-bit words (LSB-first per word).
4. Written to BRAM at `map(addr)`; the internal pointer `addrReg` increments by 8 bytes (`wordBytes`) on each completed word.

### 3.3 Read Logits (`R`)

- The bridge enters `R_WAIT`, asserts `outStream.ready := (state === R_WAIT) && tx.ready`, and waits for `outStream.valid`.
- Each received output byte is transferred to `UartTx`, repeating for exactly `outCount` bytes (e.g. 10 for MNIST, 32 for Attention/Transformer).
- Once all `outCount` bytes are emitted, the bridge returns to `IDLE`.

### 3.4 Status Query (`S`)

Returns a 1-byte status bitmask `[7:0]` (LSB = bit 0):

| Bit | Field | Description |
|---|---|---|
| 7 | `statusArValid` | AXI master read address request in flight |
| 6 | `statusRValid` | AXI master read data active |
| 5 | `csrAwValid` | CSR AXI-Lite write address pending |
| 4 | `csrWValid` | CSR AXI-Lite write data pending |
| 3 | `outValid` | Accelerator output stream valid (result ready) |
| 2 | `accDone` | Neural accelerator inference completed (`done` pulse) |
| 1 | `accBusy` | Neural accelerator actively running (`busy`) |
| 0 | `ready` | Protocol probe constant (`1`) |

### 3.5 Version Query (`V`)

Returns protocol version `0x01`. Allows the host driver to verify hardware compatibility before launching inferences.

## 4. Virtual Memory Mapping

| Virtual Region | Virtual Base Address | Physical BRAM Index (64-bit word) |
|---|---|---|
| Input Image / Activations | `0x10000 ..` | `(addr - 0x10000) >> 3` |
| Weights | `0x20000 ..` | `((addr - 0x20000) >> 3) + 2048` |

- BRAM: 4096 words of 64 bits (32 KiB), split into two equal regions of 16 KiB (2048 words each).
- Clamping: Virtual addresses exceeding the memory bounds are clamped defensively to index `memoryWords - 1` to prevent out-of-bounds corruption.

## 5. Typical Inference Session (e.g. MNIST)

```
Host ──► W 0x10000 784        (Preload input image data, 784 bytes)
Host ──► W 0x20000 weightBytes (Preload network weights)
Host ──► C 0x08 0x00010000    (Configure imgBase CSR)
Host ──► C 0x0C 0x00020000    (Configure weightBase CSR)
Host ──► C 0x00 1             (Assert Start CSR)
Host ──► S                    (Poll until accDone = 1 or accBusy = 0)
Host ──► R                    (Read outCount result bytes: logit0 .. logit9)
```

## 6. Python Host Driver

The Python host driver is implemented in [uart_host.py](file:///wsl.localhost/Ubuntu/home/leo/spinalML/cli/spinalml_cli/uart_host.py):
- Handles serial communication with automatic chunking (256 bytes per `W` packet).
- Provides high-level methods: `ping()`, `get_status()`, `write_csr()`, `write_bram()`, `read_logits()`, and `run_inference()`.
- Validated with unit tests in `tests/python/test_uart_host.py` and co-simulation in `tests/python/test_uart_bridge.py`.

## 7. Formal Verification

The bridge FSM and memory burst controller are certified by formal proofs in `spinalML/test/src/spinalML/symbolicTest/io/`:
- **`UartBridgeFormal.scala`**: Proves AXI-Lite master handshake rules, L2 protocol parsing, stream backpressure data conservation (zero dropped bytes), and FSM deadlock freedom using SymbiYosys and CVC4.
- **`AxiReadMemFormal.scala`**: Proves AXI4 burst length compliance (`ARLEN + 1`), `RLAST` assertion timing, read channel stability under master stalls, and memory clamping bounds.
