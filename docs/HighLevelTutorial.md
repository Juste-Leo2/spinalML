# SpinalML High-Level Tutorial

This guide covers the **High-Level API** of spinalML: building a complete neural network
accelerator (AXI4 DMA, double buffering, stream wiring and arbitration included) from a
declarative list of layers. For the framework overview see [Getting Started](getting_started.md);
for per-operation hardware details see the [API Reference](opsDocs.md).

---

## 1. The software stack

Three abstractions, from top to bottom:

```
Accelerator   (SoC wrapper: AXI4 master + AXI4-Lite control bus + start/status registers)
   └── Sequential   (model builder: shapes, DMAs, double buffers, arbiter, stream wiring)
         └── LayerSpec   (declarative description of ONE layer)
```

You write `modelSpec = Seq(LayerSpec, LayerSpec, ...)`, wrap it in an `Accelerator`,
and generate Verilog. Everything else is generated:

```scala
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

case class MyAccelerator(override val axiConfig: Axi4Config) extends Accelerator(
  dataType    = I8(),          // global activation dtype
  inputShape  = Seq(8, 8, 1),  // logical input tensor [H, W, C]
  modelSpec   = Seq(
    Conv2D(inChannels = 1, outChannels = 4, kernelSize = 3, customType = Some(I32())),
    Requantize(shift = 4, targetType = I8()),
    ReLU(),
    MaxPool2D(poolSize = 2, stride = 2),
    Flatten(),
    Linear(inFeatures = 3 * 3 * 4, outFeatures = 10, customType = Some(I32())), // [3,3,4] flattened
    Requantize(shift = 4, targetType = I8())
  ),
  axiConfig   = axiConfig
)

object MyVerilog extends App {
  SpinalVerilog(MyAccelerator(Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)))
}
```

Ready-to-compile templates live in [`spinalML/src/spinalML/examples/`](../spinalML/src/spinalML/examples/):
[`HighLevelTemplate`](../spinalML/src/spinalML/examples/HighLevelTemplate.scala) (1D CNN),
[`HighLevel2DTemplate`](../spinalML/src/spinalML/examples/HighLevel2DTemplate.scala) (2D CNN),
[`HighLevelAttentionTemplate`](../spinalML/src/spinalML/examples/HighLevelAttentionTemplate.scala) (quantized transformer block).

---

## 2. Supported layers

| Category | LayerSpec | Notes & constraints |
| :--- | :--- | :--- |
| Convolution | `Conv1D(inChannels, outChannels, kernelSize)` | Valid-padding (`L_out = L − K + 1`). Compute in wide dtype (`customType = Some(I32())`) to avoid overflow. |
| | `Conv2D(inChannels, outChannels, kernelSize)` | `[H, W]` or `[H, W, C]` input, valid padding. Multi-channel supported. |
| Dense | `Linear(inFeatures, outFeatures)` | Treats its input as **a single row** of `inFeatures`: insert `Flatten()` first whenever `seqLen > 1`. Supports wXaY (see §3). |
| Activations | `ReLU()`, `LeakyReLU(shift)`, `Sigmoid()`, `Tanh()` | Shape-preserving, no weights. |
| | `Softmax()` | Over the last dimension of a `[L, C]` tensor. Float domain strongly recommended. |
| Normalization | `BatchNorm1D(features)`, `LayerNorm1D(features)` | Inference-only scale & shift. |
| Pooling | `MaxPool1D(poolSize, stride)`, `AvgPool1D(poolSize, stride)` | Multi-channel; `AvgPool1D` requires `isPow2(poolSize)` (shift-based division). |
| | `MaxPool2D(poolSize, stride)`, `AvgPool2D(poolSize, stride)` | `[H, W(, C)]`; `AvgPool2D` requires `isPow2(poolSize²)`. The C-lane output is repacked back to `lanes = 1` automatically. BRAM line buffers inside. |
| Attention | `ClassicalAttention(embedDim, numHeads)` | Scaled dot-product attention + output projection. `numHeads = 1` gives classical attention; any power-of-2 `numHeads` with `embedDim % numHeads == 0` gives multi-head attention. Float activations required. Supports wXaY (see §3). |
| Utilities | `Flatten()`, `Repack(newLanes)` | Metadata / gearbox only. |
| Precision | `Requantize(shift, targetType)` | SInt -> smaller SInt (shift + saturate). |
| | `Cast(targetType)` | Mid-network dtype change, e.g. SInt -> BF16 before a float head. |

> **Precision policy**: non-linear ops (`Softmax`, attention internals, `Sigmoid`, `Tanh`)
> are only meaningful in the float family (`BF16`, `FP8`). Integer chains should stay on
> `Conv/Linear/Pooling` and cross to float via `Cast` when reaching such blocks.

---

## 3. Weight-only quantization through the API (wXaY)

`spinalML` implements the industry-standard **wXaY** scheme: X = weight bits, Y = activation bits.
Activations stay in the float family; weights are stored as compact `I4`/`I8` integers plus
compile-time scales, and dequantized *inside* the layer before the float matmul:

```scala
ClassicalAttention(
  embedDim = 8,
  numHeads = 4,
  customWeightType = Some(I8()),                       // weights stored as I8
  weightScales = Seq(0.5, 1.0, 1.5, 2.0, 0.25, 0.75, 1.25, 1.75) // per-channel (length = embedDim)
),

Linear(
  inFeatures = 32,
  outFeatures = 4,
  customWeightType = Some(I8()),
  weightScales = Seq(0.2)                              // per-tensor (length 1)
)
```

Rules:
* Omitting `customWeightType` keeps weights in the global activation dtype.
* `weightScales` length 1 = **per-tensor**, length `embedDim` (attention) or `outFeatures`
  (linear, one scale per weight column) = **per-channel**.
* Dequantization (`W_float = FloatML(W_int) × scale`) happens at the layer IO boundary;
  the matmuls themselves remain float.
* Limitation: the SoC DMA path is byte-addressed, so **activations below 8 bits (FP4) are not
  DMA-compatible** yet. All six schemes remain available at RTL level.

See [`HighLevelAttentionTemplate.scala`](../spinalML/src/spinalML/examples/HighLevelAttentionTemplate.scala)
for a full working transformer block.

---

## 4. Memory layout and control interface

The generated `Accelerator` exposes:

* **AXI4 read-only master** — autonomously fetches image, weights and biases.
* **AXI4-Lite slave** — the CPU-side control bus:

| Offset | Register | Usage |
| :--- | :--- | :--- |
| `0x00` | Start | Write `1` to trigger one inference. |
| `0x04` | Status | Read: bit 0 = output stream valid (done). |
| `0x08` | Image base address | DDR address of the input tensor. |
| `0x0C` | Weights base address | DDR address of the weight/bias region. |

Memory contents expected by the generated hardware:

* **Image**: logical tensor flattened row-major (`[H][W][C]`, channel fastest), packed into
  64-bit AXI words little-endian.
* **Weights region**: layers in declaration order; for each layer its weights then its bias.
  Matrices are stored row-major, one matrix row per stream beat. The attention layer expects
  the four projection matrices **stacked**: `Wq | Wk | Wv | Wo`, i.e. a single
  `[4 × embedDim, embedDim]` block (Wq rows first).

---

## 5. Worked example: shapes through a 2D CNN

From [`HighLevel2DTemplate.scala`](../spinalML/src/spinalML/examples/HighLevel2DTemplate.scala):

| Stage | Layer | Output shape |
| :--- | :--- | :--- |
| Input | — | `[8, 8, 1]` (I8) |
| 1 | `Conv2D(1 -> 4, K3)` computed in I32 | `[6, 6, 4]` |
| 2 | `Requantize(shift = 4, I8)` | `[6, 6, 4]` (I8) |
| 3 | `ReLU()` | `[6, 6, 4]` |
| 4 | `MaxPool2D(K2, s2)` | `[3, 3, 4]` |
| 5 | `AvgPool2D(K2, s2)` | `[1, 1, 4]` |
| 6 | `Flatten()` | `[4, 1]` |
| 7 | `Linear(4 -> 10)` computed in I32 | `[10, 1]` |
| 8 | `Requantize(shift = 4, I8)` | `[10, 1]` |

The builder deduces every intermediate shape and final output shape automatically
(`getOutShape` chaining); you never declare them by hand.

---

## 6. Simulating your accelerator

The compiled component talks real AXI4/AXI4-Lite, so it can be driven in simulation exactly
like on an FPGA host: map a memory model on the AXI4 master, program the base addresses and
start bit over AXI4-Lite, then collect the output stream.

Reference implementations (copy-paste friendly):
* [`SequentialCNNTest.scala`](../spinalML/test/src/spinalML/test/SequentialCNNTest.scala) — 2D CNN through the SoC flow.
* [`HighLevelAttentionTest.scala`](../spinalML/test/src/spinalML/test/HighLevelAttentionTest.scala) — quantized multi-head attention end to end.

Minimal skeleton:

```scala
SimConfig.withVerilator.compile(MyAccelerator(axiConfig)).doSim { dut =>
  dut.clockDomain.forkStimulus(10)
  val mem = AxiMemorySim(dut.io.axiMaster, dut.clockDomain, AxiMemorySimConfig(maxOutstandingReads = 8))
  mem.start()

  // Fill DDR: image at 0x1000, weights at 0x2000 (see §4 layout)
  mem.memory.writeBigInt(0x1000, BigInt("3C003C003C003C00", 16), 8)

  // Program control registers, then start
  writeAxiLite(0x08, 0x1000)
  writeAxiLite(0x0C, 0x2000)
  writeAxiLite(0x00, 1)

  // Collect output beats from dut.io.outStream.stream ...
}
```

Run the suites with:

```bash
./mill spinalML.test.testOnly spinalML.test.SequentialCNNTest
./mill spinalML.test.testOnly spinalML.test.HighLevelAttentionTest
```

---

## 7. Current limitations

* **Linear graph only**: `Sequential` executes layers strictly one after another. Skip
  connections / multi-branch topologies (ResNet-like DAGs) are future work — see the roadmap.
* `Linear` consumes its whole input as one row; use `Flatten()` for sequence inputs (roadmap).
* Sub-byte activation dtypes (FP4) cannot be fetched by the DMA (byte-addressed path).
* Weight double-buffers hold each parameter tensor entirely on-chip; very large models need
  the tiling roadmap items.

Track progress on all of these in the [roadmap](roadmap.md).
