package spinalML.ops

import spinal.core._
import spinal.lib._
import spinalML.tensors.Tensor

// Hardware Gearbox component to convert stream widths
case class RepackOp[T <: Data](dataType: HardType[T], shape: Seq[Int], oldLanes: Int, newLanes: Int) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, shape, oldLanes))
    val c = master(Tensor(dataType, shape, newLanes))
  }
  
  // Convert payload to Bits to use SpinalHDL's native StreamWidthAdapter
  val bitStreamIn = io.a.stream.translateWith(io.a.stream.payload.asBits)
  
  // Adapt the width (Gearbox). The new width is newLanes * widthOf(dataType)
  val bitStreamOut = Stream(Bits(newLanes * widthOf(dataType) bits))
  StreamWidthAdapter(bitStreamIn, bitStreamOut)
  
  // Re-connect the adapted stream to the output tensor
  io.c.stream.arbitrationFrom(bitStreamOut)
  
  // Cast the Bits back into a Vec of the original dataType
  io.c.stream.payload.assignFromBits(bitStreamOut.payload)
}

object repack {
  /**
   * Modifies the physical lane width of a Tensor stream (Gearbox).
   * For instance, converts a 64-lane tensor stream into a 32-lane stream.
   * This does not modify the logical ML shape of the tensor.
   */
  def apply[T <: Data](a: Tensor[T], newLanes: Int): Tensor[T] = {
    // If the lanes are already correct, return the tensor directly
    if (a.lanes == newLanes) return a
    
    // If widths are not multiples of each other, chain through 1 lane
    if (a.lanes % newLanes != 0 && newLanes % a.lanes != 0) {
      val temp = repack(a, 1)
      return repack(temp, newLanes)
    }
    
    val repackComp = RepackOp(a.dataType, a.shape, a.lanes, newLanes)
    repackComp.io.a <> a
    repackComp.io.c
  }
}
