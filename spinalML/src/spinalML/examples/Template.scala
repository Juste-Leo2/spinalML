package spinalML.examples

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.FloatML
import spinalML.ops._
import spinalML.activations._

/**
 * SpinalML Template
 * A minimalist boilerplate to start writing your own AI hardware operations.
 */
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
