// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import spinal.core._
import spinal.core.sim._
import spinal.lib.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}
import org.scalatest.funsuite.AnyFunSuite

// Wrapper component for Seq of 3 with K=2
case class Seq2ColTestComp_3_K2[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(3, 1), lanes = 1)) // Sequence of 3
    val c = master(Tensor(dataType, Seq(2, 2), lanes = 2)) // 2 windows of size 2
  }
  io.c <> seq2col(io.a, kernelSize = 2, outLanes = 2)
}

// Wrapper component for Seq of 5 with K=3
case class Seq2ColTestComp_5_K3[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(5, 1), lanes = 1)) // Sequence of 5
    val c = master(Tensor(dataType, Seq(3, 3), lanes = 3)) // 3 windows of size 3
  }
  io.c <> seq2col(io.a, kernelSize = 3, outLanes = 3)
}

// Wrapper component for Seq of 3 with K=2 and C=2
case class Seq2ColTestComp_3_K2_C2[T <: Data](dataType: HardType[T]) extends Component {
  val io = new Bundle {
    val a = slave(Tensor(dataType, Seq(3, 2), lanes = 1)) // Sequence of 3, C=2, lanes=1
    val c = master(Tensor(dataType, Seq(2, 4), lanes = 4)) // 2 windows of size 4
  }
  io.c <> seq2col(io.a, kernelSize = 2, outLanes = 4)
}

class Seq2ColTest extends AnyFunSuite {
  test("Test Seq2Col sliding window logic") {
    SimConfig.withWave.compile(Seq2ColTestComp_3_K2(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      
      dut.clockDomain.waitSampling()
      
      val inputs = Seq(1, 2, 3)
      val expectedOutputs = Seq(
        Seq(1, 2),
        Seq(2, 3)
      )
      
      // Thread to feed inputs
      fork {
        var i = 0
        dut.io.a.stream.valid #= true
        while (i < 3) {
          dut.io.a.stream.payload(0) #= inputs(i)
          dut.clockDomain.waitSampling()
          if (dut.io.a.stream.ready.toBoolean) {
            i += 1
          }
        }
        dut.io.a.stream.valid #= false
      }
      
      // Thread to check outputs
      var i = 0
      while (i < 2) {
        dut.clockDomain.waitSampling()
        if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
          val w0 = dut.io.c.stream.payload(0).toInt
          val w1 = dut.io.c.stream.payload(1).toInt
          
          assert(w0 == expectedOutputs(i)(0), s"Window $i element 0: expected ${expectedOutputs(i)(0)}, got $w0")
          assert(w1 == expectedOutputs(i)(1), s"Window $i element 1: expected ${expectedOutputs(i)(1)}, got $w1")
          i += 1
        }
      }
      
      dut.clockDomain.waitSampling(5)
    }
  }

  test("Test Seq2Col sliding window logic on Seq of 3 with K=2 and C=2") {
    SimConfig.withWave.compile(Seq2ColTestComp_3_K2_C2(I8())).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.a.stream.valid #= false
      dut.io.c.stream.ready #= true
      dut.clockDomain.waitSampling()
      
      val inputs = Seq(
        Seq(1, 2),
        Seq(3, 4),
        Seq(5, 6)
      )
      
      val expectedOutputs = Seq(
        Seq(1, 2, 3, 4), // Window 0
        Seq(3, 4, 5, 6)  // Window 1
      )
      
      fork {
        var i = 0
        dut.io.a.stream.valid #= true
        while (i < 3) {
          dut.io.a.stream.payload(0) #= inputs(i)(0)
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
          
          dut.io.a.stream.payload(0) #= inputs(i)(1)
          dut.clockDomain.waitSamplingWhere(dut.io.a.stream.ready.toBoolean)
          
          i += 1
        }
        dut.io.a.stream.valid #= false
      }
      
      var i = 0
      while (i < 2) {
        dut.clockDomain.waitSampling()
        if (dut.io.c.stream.valid.toBoolean && dut.io.c.stream.ready.toBoolean) {
          for (ch <- 0 until 4) {
            val w = dut.io.c.stream.payload(ch).toInt
            assert(w == expectedOutputs(i)(ch), s"Window $i element $ch: expected ${expectedOutputs(i)(ch)}, got $w")
          }
          i += 1
        }
      }
      
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
    test(s"Test Seq2Col compilation on $name") {
      SpinalConfig().generateVerilog(Seq2ColTestComp_3_K2(dt()))
      SpinalConfig().generateVerilog(Seq2ColTestComp_5_K3(dt()))
    }
  }
}
