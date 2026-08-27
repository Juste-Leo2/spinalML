package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.sim._
import spinalML.tensors.Tensor
import spinalML.dtypes.BF16
import spinalML.dtypes.FloatML
import spinalML.layers.Conv2D

/** Raw probe: does a K=1 Conv2DLayer emit exactly (H-K+1)*(W-K+1) output
 *  elements (the skip-chain entry probe for the "+1 beat" of M3.2b)? */
case class K1ConvCountComp(H: Int, W: Int, K: Int) extends Component {
  val io = new Bundle {
    val x = slave(Tensor(BF16(), Seq(H, W), lanes = 1))
    val w = slave(Tensor(BF16(), Seq(K * K, 1), lanes = K * K))
    val b = slave(Tensor(BF16(), Seq(1, 1), lanes = 1))
    val y = master(Tensor(BF16(), Seq(H - K + 1, W - K + 1), lanes = 1))
  }
  io.y <> Conv2D(io.x, io.w, io.b, BF16(), parallelN = false)
}

class K1ConvCountTest extends AnyFunSuite {
  private val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384,
    defaultConfigForClockDomains = ClockDomainConfig(resetKind = BOOT))

  private def setF(p: FloatML, bits: Int): Unit = {
    p.sign #= (bits >> 15 & 1) == 1
    p.exponent #= (bits >> 7) & 0xFF
    p.mantissa #= bits & 0x7F
  }

  private def bf16Bits(f: Float): Int = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF

  test("K1Conv: exactly (H-K+1)*(W-K+1) y fires for one clean frame (K=1 then K=3)") {
    val H = 16
    val W = 16
    val K = if (sys.env.get("PROBE_K").map(_.toInt).getOrElse(0) == 3) 3 else 1
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig)
      .compile(K1ConvCountComp(H, W, K))
    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      dut.io.x.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.y.stream.ready #= true
      dut.clockDomain.waitSampling(3)

      // Weights (K*K = 1) then bias, accepted-only handshake.
      def sendOne(stream: Stream[Vec[FloatML]], bits: Int): Unit = {
        var done = false
        stream.valid #= true
        setF(stream.payload(0), bits)
        while (!done) {
          dut.clockDomain.waitSampling()
          if (stream.ready.toBoolean) done = true
        }
        stream.valid #= false
      }
      sendOne(dut.io.w.stream, bf16Bits(0.5f))
      sendOne(dut.io.b.stream, bf16Bits(0.0f))

      // Pixels 0..255 as unique small non-zero values.
      var xIdx = 0
      val total = H * W
      while (xIdx < total) {
        dut.io.x.stream.valid #= true
        setF(dut.io.x.stream.payload(0), bf16Bits((xIdx + 1).toFloat))
        dut.clockDomain.waitSampling()
        if (dut.io.x.stream.ready.toBoolean) xIdx += 1
      }
      dut.io.x.stream.valid #= false

      // Drain y, count fires.
      var fires = 0
      var timeout = 0
      val expected = (H - K + 1) * (W - K + 1)
      while (fires < expected && timeout < 20000) {
        dut.clockDomain.waitSampling()
        if (dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean) fires += 1
        timeout += 1
      }
      dut.clockDomain.waitSampling(10)
      println(s"K1Conv probe: $H x $W, K=$K -> $fires y fires (expected $expected)")
      assert(fires == expected, s"K1 conv emitted $fires vs expected $expected")
    }
  }
}
