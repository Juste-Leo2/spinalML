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
| Dense | `Linear(inFeatures, outFeatures)` | Features-last convention: consumes `[..., inFeatures]` and produces `[..., outFeatures]`. Multi-row native — a `[M, K]` input (e.g. attention output) flows through without `Flatten`. Supports wXaY (see §3). |
| Activations | `ReLU()`, `LeakyReLU(shift)`, `Sigmoid()`, `Tanh()` | Shape-preserving, no weights. |
| | `Softmax()` | Over the last dimension of a `[L, C]` tensor. Float domain strongly recommended. |
| Normalization | `BatchNorm1D(features)`, `LayerNorm1D(features)` | Inference-only scale & shift. |
| Pooling | `MaxPool1D(poolSize, stride)`, `AvgPool1D(poolSize, stride)` | Multi-channel; `AvgPool1D` requires `isPow2(poolSize)` (shift-based division). |
| | `MaxPool2D(poolSize, stride)`, `AvgPool2D(poolSize, stride)` | `[H, W(, C)]`; `AvgPool2D` requires `isPow2(poolSize²)`. The C-lane output is repacked back to `lanes = 1` automatically. BRAM line buffers inside. |
| Attention | `ClassicalAttention(embedDim, numHeads)` | Scaled dot-product attention + output projection. `numHeads = 1` gives classical attention; any power-of-2 `numHeads` with `embedDim % numHeads == 0` gives multi-head attention. Float activations required. Supports wXaY (see §3). |
| Utilities | `Flatten()` (produces `[1, totalElements]`), `Repack(newLanes)` | Metadata / gearbox only. |
| DAG merges | `Add(a, b)`, `Concat(a, b, axis = 0)` | Merge two earlier graph nodes by index (see §7). Identical shapes/dtypes required on both branches (`Cast`/`Repack` to align). |
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
  64-bit AXI words little-endian (element 0 = lowest bits of the word).
* **Weights region**: layers in declaration order; for each layer its weights then its bias,
  packed into 64-bit AXI words little-endian as well. **Each weight/bias section starts on a
  64-bit beat boundary** (4 BF16 / 8 I8 elements): pad every section up to a multiple of the
  beat capacity, exactly like the builder does internally — an unaligned region start would
  be silently served from the tail of the previous word by any real DDR controller.
  * `Linear`: the stored matrix is `[outFeatures, inFeatures]` (torch-style `W^T`), one weight
    row per stream beat.
  * Convolutions / attention: row-major `[K·K·C, N]` blocks. The attention layer expects
    the four projection matrices **stacked**: `Wq | Wk | Wv | Wo`, i.e. a single
    `[4 × embedDim, embedDim]` block (Wq rows first).
    One stream beat always carries one output neuron's K-vector.
  * Biases: one element per byte-addressable slot, streamed sequentially (`lanes = 1`).

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
| 6 | `Flatten()` | `[1, 4]` |
| 7 | `Linear(4 -> 10)` computed in I32 | `[1, 10]` |
| 8 | `Requantize(shift = 4, I8)` | `[1, 10]` |

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
* [`MnistTest.scala`](../spinalML/test/src/spinalML/examples/MnistTest.scala) — a real trained MNIST CNN
  (2 942 parameters, Conv→ReLU→MaxPool→Flatten→Linear in BF16) validated black-box: five digits,
  argmax against the true labels, 5/5. See [`Mnist.scala`](../spinalML/src/spinalML/examples/Mnist.scala)
  for the model and [the session notes](bugs/2026-08-mnist-session.md) for the full story
  (burst DMA, region alignment, flatten-order remap).
* [`Mnistw4a8Test.scala`](../spinalML/test/src/spinalML/examples/Mnistw4a8Test.scala) — the same network in
  **W4A8 mixed precision**: true INT4 convolution (nibble-packed weights, integer activations and
  accumulator, quantized bias) followed by `Cast(FP8_E4M3, scales)` and an FP8 Linear, 5/5.
  See [the W4A8 session notes](bugs/2026-08-w4a8-session.md).

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

## 7. Beyond linear chains: DAG topologies

`Sequential` generalizes from a linear chain to a **directed acyclic graph (DAG)**: any
layer output can feed several consumers, and `Add`/`Concat` merge nodes can combine
several branches — enabling ResNet-style skip connections and multi-branch heads.

Nodes are identified by position in the graph: **node 0 is the network input**, node *k*
is the output of the *k*-th spec entry. Standard layers implicitly consume the previous
node; merge layers take two explicit references that must point **backwards**, which makes
cycles impossible by construction:

```scala
Accelerator(dataType = BF16(), inputShape = Seq(2, 4),
  modelSpec = Seq(
    Linear(4, 4),          // node 1 <- node 0 (implicit)
    ReLU(),                // node 2 <- node 1
    Linear(4, 4),          // node 3 <- node 2      <- main branch
                           //                        (node 0 waits in its FIFO)
    Add(a = 0, b = 3),     // node 4 = node 0 + node 3   <- skip connection
    ReLU()                 // node 5 <- node 4  = accelerator output
  ))
```

A complete working example lives in
[`ResidualMLPTemplate.scala`](../spinalML/src/spinalML/examples/ResidualMLPTemplate.scala),
validated bit-exactly through the SoC flow by
[`DagTopologyTest.scala`](../spinalML/test/src/spinalML/nn/DagTopologyTest.scala).

Rules enforced at elaboration:
* References must satisfy `ref <= current index` (acyclic by construction) — forward
  references are rejected.
* Both merge inputs must share identical shapes and dtypes; insert `Cast`/`Repack`
  layers on a branch to align them. Mismatches produce explicit error messages.
* Every node must be consumed; the last node drives the accelerator output stream.

Hardware cost model: when a node feeds more than one consumer, the builder inserts a
stream fork; each deferred branch owns an exact-capacity FIFO (`TapBuffer`) holding one
full tensor — safe for one-shot inference since the deferred consumer always drains it.
Execution order still follows the spec list (topology support, not parallel scheduling);
the shared AXI arbiter serializes weight fetches as before.

---

## 8. Current limitations

* **One-shot inference contract**: every buffer holds one full tensor and each `start`
  runs a whole inference — models are implicitly capped at what fits on-chip. Concretely,
  back-to-back `start` pulses on the same live datapath are not re-armed yet (buffers/FSMs
  keep residual state): give each inference a fresh elaboration/simulation, as
  [`MnistTest`](../spinalML/test/src/spinalML/examples/MnistTest.scala) does. The multi-tile
  continuous execution model (weight residency, activation tiling, DAG tap
  contract for streaming) is detailed in the [roadmap](roadmap.md), section 6.
* Sub-byte activation dtypes (FP4) cannot be fetched by the DMA (byte-addressed path).
* Weight double-buffers hold each parameter tensor entirely on-chip; very large models need
  the tiling roadmap items.
* DAG execution follows spec order (no inter-branch parallel scheduling yet); `Concat`
  supports axis 0 only for now.

Track progress on all of these in the [roadmap](roadmap.md).
