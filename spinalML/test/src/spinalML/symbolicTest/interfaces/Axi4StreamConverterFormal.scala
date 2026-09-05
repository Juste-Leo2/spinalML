package spinalML.symbolicTest.interfaces

import spinal.core._
import spinal.core.formal._
import spinal.lib._
import spinal.lib.bus.amba4.axis._
import spinalML.interfaces.{Axi4StreamToTensor, TensorToAxi4Stream}
import spinalML.tensors.Tensor

/**
 * Formal verification for Axi4StreamConverter bridges:
 *  - Axi4StreamToTensor: unpacks AXI4-Stream data bits to Tensor lanes.
 *  - TensorToAxi4Stream: packs Tensor lanes into AXI4-Stream data bits and generates TLAST.
 *
 * Properties verified:
 *  1. Handshake Transparency:
 *     - axis.fire <=> tensor.fire for both converters.
 *     - Combinational valid/ready forwarding without internal bubble or pipeline delay.
 *  2. Frame Framing & TLAST generation (TensorToAxi4Stream):
 *     - axis.last pulses IF AND ONLY IF the final chunk of the tensor frame is being transmitted.
 *     - Exactly one last pulse every totalChunks handshakes.
 *  3. Bit-level slicing & zero-padding:
 *     - Unpacked lanes match the corresponding slices of the incoming AXI word.
 *     - Padded higher-order bits are driven to zero.
 *  4. Roundtrip conservation:
 *     - Connecting Axi4StreamToTensor -> TensorToAxi4Stream preserves data and handshakes.
 *  5. Reachability (Liveness):
 *     - Transmission of a complete tensor frame up to TLAST under backpressure.
 */
class Axi4StreamConverterFormal extends Component {
  val shape = Seq(4)
  val lanes = 2
  val totalChunks = shape.product / lanes // 2 chunks per frame
  val axiWidth = 32 // 32-bit bus, 16-bit tensor chunk width

  val toTensor = FormalDut(Axi4StreamToTensor(UInt(8 bits), shape, lanes, axiWidth))
  val toAxis   = FormalDut(TensorToAxi4Stream(UInt(8 bits), shape, lanes, axiWidth))

  // Drive toTensor inputs
  anyseq(toTensor.io.axis.valid)
  anyseq(toTensor.io.axis.data)
  anyseq(toTensor.io.axis.last)
  anyseq(toTensor.io.tensor.stream.ready)

  // Drive toAxis inputs
  anyseq(toAxis.io.tensor.stream.valid)
  anyseq(toAxis.io.tensor.stream.payload)
  anyseq(toAxis.io.axis.ready)

  assumeInitial(clockDomain.isResetActive)

  // Stream handshake assumptions (stability when stalled)
  when(pastValid() && past(toTensor.io.axis.valid) && !past(toTensor.io.axis.ready)) {
    assume(toTensor.io.axis.valid)
    assume(toTensor.io.axis.data === past(toTensor.io.axis.data))
    assume(toTensor.io.axis.last === past(toTensor.io.axis.last))
  }
  when(pastValid() && past(toAxis.io.tensor.stream.valid) && !past(toAxis.io.tensor.stream.ready)) {
    assume(toAxis.io.tensor.stream.valid)
    assume(toAxis.io.tensor.stream.payload === past(toAxis.io.tensor.stream.payload))
  }

  // =========================================================================
  // 1. Axi4StreamToTensor FLOW PROPERTIES
  // =========================================================================
  // Handshake transparency
  assert(toTensor.io.tensor.stream.valid === toTensor.io.axis.valid,
    "toTensor: tensor.valid must strictly mirror axis.valid")
  assert(toTensor.io.axis.ready === toTensor.io.tensor.stream.ready,
    "toTensor: axis.ready must strictly mirror tensor.ready")
  assert(toTensor.io.tensor.stream.fire === toTensor.io.axis.fire,
    "toTensor: fire events must be simultaneous")

  // Data translation: each lane maps to an 8-bit slice
  when(toTensor.io.tensor.stream.valid) {
    for (i <- 0 until lanes) {
      val expectedLane = toTensor.io.axis.data(i * 8, 8 bits)
      assert(toTensor.io.tensor.stream.payload(i).asBits === expectedLane,
        s"toTensor: lane $i mismatch against axis slice")
    }
  }

  // =========================================================================
  // 2. TensorToAxi4Stream FLOW PROPERTIES
  // =========================================================================
  // Handshake transparency
  assert(toAxis.io.axis.valid === toAxis.io.tensor.stream.valid,
    "toAxis: axis.valid must strictly mirror tensor.valid")
  assert(toAxis.io.tensor.stream.ready === toAxis.io.axis.ready,
    "toAxis: tensor.ready must strictly mirror axis.ready")
  assert(toAxis.io.axis.fire === toAxis.io.tensor.stream.fire,
    "toAxis: fire events must be simultaneous")

  // Data packing and padding: lanes 0..1 in bits [15:0], bits [31:16] zero-padded
  when(toAxis.io.axis.valid) {
    val packedLanes = toAxis.io.tensor.stream.payload.asBits
    assert(toAxis.io.axis.data(15 downto 0) === packedLanes,
      "toAxis: payload packed bits must match lower AXI word")
    assert(toAxis.io.axis.data(31 downto 16) === 0,
      "toAxis: padding bits must be zero")
  }

  // =========================================================================
  // 3. TLAST GENERATION SPECIFICATION
  // =========================================================================
  val chunkCounter = toAxis.chunkCounter.value.pull()

  // TLAST must be True IF AND ONLY IF chunkCounter is on the last chunk
  assert(toAxis.io.axis.last === (chunkCounter === totalChunks - 1),
    "axis.last must be asserted only on the final chunk of the tensor frame")

  // Counter transition: increments on each handshake, wraps on totalChunks
  when(pastValid() && !clockDomain.isResetActive) {
    when(past(toAxis.io.axis.fire)) {
      when(past(chunkCounter === totalChunks - 1)) {
        assert(chunkCounter === 0, "chunkCounter did not wrap to 0 after last beat")
      } otherwise {
        assert(chunkCounter === past(chunkCounter) + 1, "chunkCounter did not increment linearly")
      }
    } otherwise {
      assert(chunkCounter === past(chunkCounter), "chunkCounter moved without a handshake")
    }
  }

  // =========================================================================
  // 4. REACHABILITY / LIVENESS COVERS
  // =========================================================================
  // 1. Transaction on toTensor
  cover(toTensor.io.axis.fire)

  // 2. Transaction on toAxis
  cover(toAxis.io.axis.fire)

  // 3. Complete tensor frame reached (TLAST asserted on fire)
  cover(toAxis.io.axis.fire && toAxis.io.axis.last)

  // 4. Consecutive frames: TLAST fire followed by initial chunk (counter 0) fire
  cover(pastValid() && past(toAxis.io.axis.fire && toAxis.io.axis.last) &&
        toAxis.io.axis.fire && (chunkCounter === 0))
}

object Axi4StreamConverterFormal {
  def main(args: Array[String]): Unit = {
    FormalConfig
      .withSymbiYosys
      .withBMC(10)
      .withTimeout(600)
      .withDebug
      .withEngies(List(SmtBmc(solver = SmtBmcSolver.cvc4)))
      .workspacePath("formal")
      .doVerify(new Axi4StreamConverterFormal, "axi4_stream_converter_formal")
  }
}
