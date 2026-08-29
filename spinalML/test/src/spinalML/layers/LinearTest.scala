package spinalML.layers

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{FloatML, I4, I8, I16, I32, FP4_E2M1, FP8_E4M3, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component
case class LinearTestComp[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(1, 2), lanes = 2)) // 1 row, 2 cols (M=1, K=2)
    val w = slave(Tensor(dataType, Seq(2, 1), lanes = 2)) // 2 weights (K=2, 1)
    val b = slave(Tensor(accType, Seq(1, 1), lanes = 1)) // 1 bias
    val y = master(Tensor(accType, Seq(1, 1), lanes = 1))
  }
  io.y <> Linear(io.a, io.w, io.b, accType, tileSize = 2, parallelN = false)
}

case class LinearTestCompMulti[T <: Data, TAcc <: Data](dataType: HardType[T], accType: HardType[TAcc]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 3), lanes = 3)) // M=2, K=3
    val w = slave(Tensor(dataType, Seq(3, 4), lanes = 3)) // K=3, N=4
    val b = slave(Tensor(accType, Seq(1, 4), lanes = 1)) // 1, 4 bias
    val y = master(Tensor(accType, Seq(2, 4), lanes = 1)) // M=2, N=4
  }
  io.y <> Linear(io.a, io.w, io.b, accType, tileSize = 2, parallelN = false)
}

// Weight-only quantization (wXaY): SInt weights + compile-time scale(s),
// float activations. accType = activation dtype.
case class LinearQuantTestComp[T <: Data](dataType: HardType[T], weightDt: HardType[SInt], scales: Seq[Double] = Seq(1.0)) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(1, 2), lanes = 2)) // M=1, K=2
    val w = slave(Tensor(weightDt, Seq(2, 1), lanes = 2)) // K=2, N=1
    val b = slave(Tensor(dataType, Seq(1, 1), lanes = 1))
    val y = master(Tensor(dataType, Seq(1, 1), lanes = 1))
  }
  io.y <> Linear(io.a, io.w, io.b, dataType, scales, false, 1024, None, None)
}

case class LinearQuantTestCompMulti[T <: Data](dataType: HardType[T], weightDt: HardType[SInt], scales: Seq[Double] = Seq(1.0)) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(2, 3), lanes = 3)) // M=2, K=3
    val w = slave(Tensor(weightDt, Seq(3, 4), lanes = 3)) // K=3, N=4
    val b = slave(Tensor(dataType, Seq(1, 4), lanes = 1))
    val y = master(Tensor(dataType, Seq(2, 4), lanes = 1))
  }
  io.y <> Linear(io.a, io.w, io.b, dataType, scales, false, 1024, None, None)
}

class LinearTest extends AnyFunSuite {
  test("Test Linear Layer: Y = A * W + b on I4 tensors") {
    SimConfig.withWave.compile(LinearTestComp(I4(), I4())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Step 1: Load Weights W into the Matmul Double-Buffer
      // W = [2, -1]T
      dut.io.w.stream.valid #= true
      dut.io.w.stream.payload(0) #= 2
      dut.io.w.stream.payload(1) #= -1
      dut.clockDomain.waitSamplingWhere(dut.io.w.stream.ready.toBoolean)
      dut.io.w.stream.valid #= false
      
      // Step 2: Send the bias b
      // b = 3
      dut.io.b.stream.valid #= true
      dut.io.b.stream.payload(0) #= 3
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.valid #= false
      
      // Step 3: Stream Activations A (1 row)
      // Row 0: [-2, 3] -> Y0 = (-2*2 + 3*(-1)) + 3 = -4 - 3 + 3 = -4
      
      val a_data = Seq(Seq(-2, 3))
      val expected_y = Seq(-4)
      
      // Start streaming A
      dut.io.a.stream.valid #= true
      dut.io.a.stream.payload(0) #= a_data(0)(0)
      dut.io.a.stream.payload(1) #= a_data(0)(1)
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      
      // Step 4: Verify output Y (1 row)
      for (i <- 0 until 1) {
        dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean)
        val result = dut.io.y.stream.payload(0).toInt
        assert(result == expected_y(i), s"Expected ${expected_y(i)} for row $i, got $result")
        // Since y streams at 1 element per cycle, wait 1 cycle to consume
        dut.clockDomain.waitSampling(1)
      }
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test Linear Quant Layer: Y = A * dequant(W) + b (wXaY)") {
    SimConfig.withWave.compile(LinearQuantTestComp(BF16(), I8())).doSim { dut =>
      
      def setFloat(p: FloatML, f: Float): Unit = {
        val bits = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF
        p.sign #= ((bits >> 15) & 1) == 1
        p.exponent #= (bits >> 7) & 0xFF
        p.mantissa #= bits & 0x7F
      }
      def getFloat(p: FloatML): Float = {
        val bits = ((if (p.sign.toBoolean) 1 else 0) << 15) | ((p.exponent.toInt & 0xFF) << 7) | (p.mantissa.toInt & 0x7F)
        java.lang.Float.intBitsToFloat(bits << 16)
      }
      
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.y.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      // Step 1: Load Weights W (I8, scale = 1.0)
      // W = [2, -1]^T streamed column-major as one beat of 2 lanes
      dut.io.w.stream.valid #= true
      dut.io.w.stream.payload(0).asInstanceOf[SInt] #= 2
      dut.io.w.stream.payload(1).asInstanceOf[SInt] #= -1
      dut.clockDomain.waitSamplingWhere(dut.io.w.stream.ready.toBoolean)
      dut.io.w.stream.valid #= false
      
      // Step 2: Send the bias b (BF16)
      // b = 3
      dut.io.b.stream.valid #= true
      setFloat(dut.io.b.stream.payload(0).asInstanceOf[FloatML], 3.0f)
      dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      dut.io.b.stream.valid #= false
      
      // Step 3: Stream Activations A (BF16)
      // Row 0: [-2, 3] -> Y0 = (-2*dequant(2) + 3*dequant(-1)) + 3 = -4
      
      val expected_y = Seq(-4.0f)
      
      // Start streaming A
      dut.io.a.stream.valid #= true
      setFloat(dut.io.a.stream.payload(0).asInstanceOf[FloatML], -2.0f)
      setFloat(dut.io.a.stream.payload(1).asInstanceOf[FloatML], 3.0f)
      dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      
      dut.io.a.stream.valid #= false
      
      // Step 4: Verify output Y (1 row)
      for (i <- 0 until 1) {
        dut.clockDomain.waitSamplingWhere(dut.io.y.stream.valid.toBoolean)
        val result = getFloat(dut.io.y.stream.payload(0).asInstanceOf[FloatML])
        assert(result == expected_y(i), s"Expected ${expected_y(i)} for row $i, got $result")
        // Since y streams at 1 element per cycle, wait 1 cycle to consume
        dut.clockDomain.waitSampling(1)
      }
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test Linear Layer multi-row: Y = A * W + b with M=2 on I8 tensors") {
    val A = Seq(Seq(1, -2, 3), Seq(0, 1, -4))
    val Wflat = Seq(1, 2, 0, 0, 1, -1, -1, 0, 1, 2, -1, 1)
    val bias = Seq(1, 2, 3, 4)
    val expected = Seq(-2, -3, 5, 11, 3, 7, -1, -1)

    SimConfig.withWave.compile(LinearTestCompMulti(I8(), I32())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.a.stream.valid #= false
      dut.io.w.stream.valid #= false
      dut.io.b.stream.valid #= false
      dut.io.y.stream.ready #= true

      dut.clockDomain.waitSampling()

      dut.io.w.stream.valid #= true
      for (beat <- 0 until 4) {
        for (i <- 0 until 3) {
          dut.io.w.stream.payload(i) #= Wflat(beat * 3 + i)
        }
        dut.clockDomain.waitSamplingWhere(dut.io.w.stream.ready.toBoolean)
      }
      dut.io.w.stream.valid #= false

      dut.io.b.stream.valid #= true
      for (n <- 0 until 4) {
        dut.io.b.stream.payload(0) #= bias(n)
        dut.clockDomain.waitSamplingWhere(dut.io.b.stream.ready.toBoolean)
      }
      dut.io.b.stream.valid #= false

      dut.io.a.stream.valid #= true
      for (m <- 0 until 2) {
        for (i <- 0 until 3) {
          dut.io.a.stream.payload(i) #= A(m)(i)
        }
        dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
      }
      dut.io.a.stream.valid #= false

      val collected = scala.collection.mutable.ArrayBuffer[Int]()
      var timeout = 0
      while (collected.length < expected.length && timeout < 10000) {
        if (dut.io.y.stream.valid.toBoolean && dut.io.y.stream.ready.toBoolean) {
          collected += dut.io.y.stream.payload(0).toInt
        }
        dut.clockDomain.waitSampling()
        timeout += 1
      }

      assert(timeout < 10000, s"Timeout: collected ${collected.length}/${expected.length}")
      assert(collected.toSeq == expected,
        s"Expected $expected, got $collected")
      dut.clockDomain.waitSampling(5)
    }
  }

  val compileTypes = Seq(
    ("I8", () => I8()),
    ("FP8", () => FP8_E4M3()),
    ("I16", () => I16()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    val accDt = if (name == "I8" || name == "I16") () => spinalML.dtypes.I32() else dt
    test(s"Test Linear compilation on $name") {
      SpinalConfig().generateVerilog(LinearTestComp(dt(), accDt()))
      SpinalConfig().generateVerilog(LinearTestCompMulti(dt(), accDt()))
    }
  }

  val quantCombos = Seq(
    ("w8a16", () => I8(), () => BF16()),
    ("w4a16", () => I4(), () => BF16()),
    ("w8a8", () => I8(), () => FP8_E4M3()),
    ("w4a8", () => I4(), () => FP8_E4M3()),
    ("w8a4", () => I8(), () => FP4_E2M1()),
    ("w4a4", () => I4(), () => FP4_E2M1())
  )

  for ((combo, wd, ad) <- quantCombos) {
    test(s"Test Linear Quant compilation on $combo") {
      SpinalConfig().generateVerilog(LinearQuantTestComp(ad(), wd()))
      SpinalConfig().generateVerilog(LinearQuantTestCompMulti(ad(), wd()))
    }
  }

  test("Test LinearQuant PerChannel compilation multi") {
    SpinalConfig().generateVerilog(LinearQuantTestCompMulti(BF16(), I8(), Seq(0.5, -0.25, 1.5, 2.0)))
  }
}

