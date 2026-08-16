# Getting Started with SpinalML

Welcome to **SpinalML**! SpinalML is an open-source framework written in Scala (on top of SpinalHDL) that allows you to easily design and deploy custom Neural Network hardware accelerators on FPGA.

## 1. The High-Level Sequential API (PyTorch-like)

The recommended way to use SpinalML is via the **High-Level API**. If you are familiar with PyTorch or Keras, you will feel right at home! 

SpinalML allows you to define your neural network architecture declaratively using a `Sequential` model builder. The framework will automatically handle:
- **Dimensions & Shapes**: Deduce the tensor dimensions throughout the network.
- **AXI4 Bus Arbitration**: Instantiate the AXI memory-mapped DMAs and arbiters to autonomously fetch your inputs, weights, and biases from the external DDR4 memory.
- **Hardware Pipelining**: Insert FIFOs and connect the AXI4-Stream handshakes between layers.

### The High-Level Template

Here is a complete, ready-to-use template (`HighLevelTemplate.scala`) that defines an entire CNN and generates the Verilog hardware:

```scala
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._

case class HighLevelTemplate(override val axiConfig: Axi4Config) extends Accelerator(
  dataType = I8(),            // Global quantization format for the network
  inputShape = Seq(28, 1),    // The expected shape of the input tensor (e.g. 1D signal of length 28)
  
  // ==========================================
  // DEFINE YOUR NEURAL NETWORK TOPOLOGY HERE
  // ==========================================
  modelSpec = Seq(
    // 1. Compute Conv1D in I32 to prevent overflow
    Conv1D(inChannels = 1, outChannels = 4, kernelSize = 3, customType = Some(I32())),
    
    // 2. Requantize the I32 output back to I8 for the rest of the network
    Requantize(shift = 4, targetType = I8()),
    
    ReLU(),
    MaxPool1D(poolSize = 2, stride = 2),
    Flatten(),
    
    // 3. Do the same for the final dense layer
    Linear(inFeatures = 52, outFeatures = 10, customType = Some(I32())),
    Requantize(shift = 4, targetType = I8())
  ),
  
  axiConfig = axiConfig
)

// Generate the Verilog for the FPGA
object HighLevelTemplateVerilog extends App {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  SpinalVerilog(HighLevelTemplate(axiConfig))
}
```

> [!TIP]
> **Advanced Topology Control:** The High-Level API supports dynamic modifications of the hardware datapath directly within the `modelSpec`:
> - **Dynamic Mixed Precision**: Use `Requantize(shift, targetType)` to adjust quantization and change precision on the fly.
> - **Manual Repacking**: Use `Repack(newLanes)` to dynamically change the physical bus width between layers to save FPGA resources.

---

## 2. Low-Level Hardware API

If you need total control over your architecture, you can wire up your datapath manually using the low-level hardware modules.

### Tensors and Streams
In SpinalML, data is passed between layers using the `Tensor[T]` interface. A tensor is essentially a multi-dimensional array of data, but in hardware, it is transmitted piece by piece over time.

SpinalML uses an **AMBA AXI4-Stream**-like protocol to pass data. Every `Tensor[T]` exposes a `stream` with:
- `payload`: The actual data bits.
- `valid`: A signal from the sender indicating the `payload` is ready.
- `ready`: A signal from the receiver indicating it can accept data.

You **never** need to manage `valid` and `ready` signals manually when using built-in operations. SpinalML automatically pipelines and synchronizes the flow.

### Shape vs Lanes
A tensor has two distinct dimension concepts:
1. **`shape: Seq[Int]`**: The logical dimensions of the Deep Learning tensor.
2. **`lanes: Int`**: The physical bus width / parallelism. If `lanes = 4`, the hardware will process 4 elements per clock cycle.

### Repack (Gearbox)
What happens if your external memory provides 64 elements per clock cycle, but your Neural Network layer only processes 8 elements per cycle? You use `repack`!
```scala
val memStream = Tensor(FP8, Seq(64, 32, 32), lanes = 64)
val nnStream = spinalML.ops.repack(memStream, newLanes = 8)
```
`repack` safely buffers and slices the physical bus without altering the ML shape.

### The Low-Level Minimal Template

You can use the provided **[Template.scala](https://github.com/Juste-Leo2/spinalML/blob/main/spinalML/src/spinalML/examples/Template.scala)** as your boilerplate for writing manual logic:

```scala
import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops._
import spinalML.activations._

case class Template[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(dataType, shape, lanes))
    val y = master(Tensor(dataType, shape, lanes))
  }
  
  // Y = relu(abs(X))
  io.y <> relu(abs(io.x))
}
```

## 3. Data Types

SpinalML supports standard `SInt` and `UInt` for integer quantization, but excels with its custom **`FloatML`** data type for floating-point operations.

- `I8`: `SInt(8 bits)`
- `FP8` (E4M3): `FloatML(expBits = 4, mantBits = 3)`
- `BF16`: `FloatML(expBits = 8, mantBits = 7)`

Operations like `Exp`, `Softmax`, and `Rsqrt` are implemented using a novel **Algebraic Separation** technique for `FloatML` types > 8 bits, guaranteeing near-perfect accuracy without DSP multipliers!

## 4. Explore the Examples

To see these concepts in action, check out the provided examples directly in the repository:

- **High-Level API:**
  - **`HighLevelTemplate.scala`**: The base PyTorch-like boilerplate.
  - **`SequentialCNN.scala`**: A slightly more complex CNN using the High-Level API.
  - **`Comprehensive1DCNN.scala`**: A full real-world CNN with Max Pooling and BN.
- **Low-Level Hardware API:**
  - **`Template.scala`**: The manual routing boilerplate.
  - **`SimplePipeline.scala`**: Matrix multiplication + element-wise addition.
  - **`SimpleCNN.scala`**: A fully hand-wired CNN with manual repacks and memory mapping.
