# Getting Started with SpinalML

Welcome to **SpinalML**! SpinalML is an open-source framework written in Scala (on top of SpinalHDL) that allows you to easily design and deploy custom Neural Network hardware accelerators on FPGA.

## 1. Core Concepts

> [!TIP]
> **API Reference:** For a full list of all supported hardware operations, their inputs, and outputs, please consult the **[Operations Documentation (opsDocs.md)](./opsDocs.md)**.


### Tensors and Streams
In SpinalML, data is passed between layers using the `Tensor[T]` interface. A tensor is essentially a multi-dimensional array of data, but in hardware, it is transmitted piece by piece over time.

SpinalML uses an **AMBA AXI4-Stream**-like protocol to pass data. Every `Tensor[T]` exposes a `stream` with:
- `payload`: The actual data bits.
- `valid`: A signal from the sender indicating the `payload` is ready.
- `ready`: A signal from the receiver indicating it can accept data.

You **never** need to manage `valid` and `ready` signals manually when using built-in operations. SpinalML automatically pipelines and synchronizes the flow.

### Shape vs Lanes
A tensor has two distinct dimension concepts:
1. **`shape: Seq[Int]`**: The logical dimensions of the Deep Learning tensor (e.g., `Seq(64, 32, 32)` for an image with 64 channels and 32x32 pixels).
2. **`lanes: Int`**: The physical bus width / parallelism. If `lanes = 4`, the hardware will process 4 elements per clock cycle. The total time to process the tensor will be `shape.product / lanes` clock cycles.

### Repack (Gearbox)
What happens if your external memory provides 64 elements per clock cycle, but your Neural Network layer only processes 8 elements per cycle?
You use `repack`!
```scala
val memStream = Tensor(FP8, Seq(64, 32, 32), lanes = 64)
val nnStream = spinalML.ops.repack(memStream, newLanes = 8)
```
`repack` safely buffers and slices the physical bus without altering the ML shape.

## 2. The Minimal Template

To start building a custom hardware operation, you can use the provided **[Template.scala](https://github.com/Juste-Leo2/spinalML/blob/main/spinalML/src/spinalML/examples/Template.scala)** as your boilerplate. It features a simple "fill-in-the-blanks" structure:

```scala
import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops._
import spinalML.activations._

case class Template[T <: Data](dataType: HardType[T], shape: Seq[Int], lanes: Int) extends Component {
  
  // ==========================================
  // 1. DEFINE YOUR IO (Inputs / Outputs)
  // ==========================================
  val io = new Bundle {
    val x = slave(Tensor(dataType, shape, lanes))
    val y = master(Tensor(dataType, shape, lanes))
  }
  
  // ==========================================
  // 2. WRITE YOUR ML DATAFLOW
  // ==========================================
  // Example: Y = relu(abs(X))
  
  val absX = abs(io.x)
  val reluX = relu(absX)
  
  // ==========================================
  // 3. CONNECT TO OUTPUT
  // ==========================================
  io.y <> reluX
  
}
```

You can simulate this exact template using its companion testbench, **[TemplateTest.scala](https://github.com/Juste-Leo2/spinalML/blob/main/spinalML/test/src/spinalML/examples/TemplateTest.scala)**.

## 3. Data Types

SpinalML supports standard `SInt` and `UInt` for integer quantization, but excels with its custom **`FloatML`** data type for floating-point operations.

- `I8`: `SInt(8 bits)`
- `FP8` (E4M3): `FloatML(expBits = 4, mantBits = 3)`
- `BF16`: `FloatML(expBits = 8, mantBits = 7)`

Operations like `Exp`, `Softmax`, and `Rsqrt` are implemented using a novel **Algebraic Separation** technique for `FloatML` types > 8 bits, guaranteeing near-perfect accuracy without DSP multipliers!

## 4. Explore the Examples

To see these concepts in action, check out the provided examples directly in the repository:

- **[SimplePipeline.scala](https://github.com/Juste-Leo2/spinalML/blob/main/spinalML/src/spinalML/examples/SimplePipeline.scala)**: A short demonstration of matrix multiplication combined with element-wise addition.
- **[SimpleCNN.scala](https://github.com/Juste-Leo2/spinalML/blob/main/spinalML/src/spinalML/examples/SimpleCNN.scala)**: A complete 1D Convolutional Neural Network featuring Conv1D, BatchNorm, MaxPool, Gearbox (`repack`), Linear, and Softmax layers!
