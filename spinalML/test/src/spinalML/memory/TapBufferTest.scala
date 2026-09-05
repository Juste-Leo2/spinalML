// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.I8

import scala.util.Random

case class TapBufferTestComp[T <: Data](dataType: HardType[T], nElems: Int, lanes: Int) extends Component {
  val io = new Bundle {
    val in = slave(Tensor(dataType, Seq(nElems), lanes))
    val direct = master(Tensor(dataType, Seq(nElems), lanes))
    val deferred = master(Tensor(dataType, Seq(nElems), lanes))
    val fireCount = out UInt(16 bits)
  }
  val forks = TapBuffer.fork(io.in, 2)

  // Registered outputs: testbenches observe only stable, edge-synchronized
  // valid/payload pairs regardless of coroutine scheduling order.
  val directPipe = forks(0).stream.m2sPipe()
  val deferredPipe = forks(1).stream.m2sPipe()
  val directT = Tensor(dataType, Seq(nElems), lanes)
  val deferredT = Tensor(dataType, Seq(nElems), lanes)
  directT.stream << directPipe
  deferredT.stream << deferredPipe
  io.direct <> directT
  io.deferred <> deferredT

  // DUT-side transaction counter: lets testbenches count input handshakes
  // without racing on combinational ready signals.
  val fireReg = Reg(UInt(16 bits)) init (0)
  fireReg := fireReg + io.in.stream.fire.asUInt(16 bits)
  io.fireCount := fireReg
}

class TapBufferTest extends AnyFunSuite {
  test("TapBuffer fork: deferred consumer receives the full stream after the direct one") {
    val lanes = 4
    val beats = 6
    val elements = beats * lanes

    SimConfig.withVerilator.compile(TapBufferTestComp(I8(), elements, lanes)).doSim { dut =>
      dut.clockDomain.forkStimulus(10)

      val expected = Seq.tabulate(elements)(i => ((i * 7 + 3 + 128) & 0xFF) - 128)
      val direct = scala.collection.mutable.ArrayBuffer[Int]()
      val deferred = scala.collection.mutable.ArrayBuffer[Int]()
      val rnd = new Random(42)

      dut.io.in.stream.valid #= false
      dut.io.direct.stream.ready #= true
      dut.io.deferred.stream.ready #= false

      var sent = 0
      dut.clockDomain.onSamplings {
        if (dut.io.in.stream.valid.toBoolean && dut.io.in.stream.ready.toBoolean) sent += 1
        if (direct.length < elements &&
            dut.io.direct.stream.valid.toBoolean && dut.io.direct.stream.ready.toBoolean) {
          direct ++= Seq.tabulate(lanes)(l => dut.io.direct.stream.payload(l).toInt)
        }
        if (deferred.length < elements &&
            dut.io.deferred.stream.valid.toBoolean && dut.io.deferred.stream.ready.toBoolean) {
          deferred ++= Seq.tabulate(lanes)(l => dut.io.deferred.stream.payload(l).toInt)
        }
      }

      var timeout = 0
      var deferDrainStarted = false
      while ((direct.length < elements || deferred.length < elements) && timeout < 50000) {
        // Producer: stream everything as fast as the fork accepts it
        dut.io.in.stream.valid #= sent < beats
        if (sent < beats) {
          for (l <- 0 until lanes) dut.io.in.stream.payload(l) #= expected(sent * lanes + l)
        }

        // Start draining the deferred branch once the producer is done,
        // with random backpressure to exercise the FIFO.
        if (!deferDrainStarted && sent == beats) {
          deferDrainStarted = true
        }
        if (deferDrainStarted) {
          dut.io.deferred.stream.ready #= rnd.nextBoolean()
        }

        dut.clockDomain.waitSampling()
        timeout += 1
      }

      assert(timeout < 50000, s"Timeout: direct=${direct.length}/${elements} deferred=${deferred.length}/${elements}")
      assert(direct.length == elements, s"Direct got ${direct.length}/${elements}")
      assert(deferred.length == elements, s"Deferred got ${deferred.length}/${elements}")
      assert(direct.toSeq == expected, "Direct stream corrupted")
      assert(deferred.toSeq == expected, "Deferred stream corrupted")
    }
  }

  test("TapBuffer compilation") {
    SpinalConfig().generateVerilog(TapBufferTestComp(I8(), 32, 2))
    SpinalConfig().generateVerilog(TapBuffer(I8(), 64, 4))
  }

  test("TapBuffer cocotb verilog generation") {
    SpinalConfig().generateVerilog(TapBufferTestComp(I8(), 32, 4))
  }
}
