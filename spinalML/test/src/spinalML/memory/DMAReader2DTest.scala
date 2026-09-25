// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axi.sim._
import spinalML.dtypes.I8 
import spinalML.tensors.Tensor
import org.scalatest.funsuite.AnyFunSuite

// Wrapper to expose full Axi4 for the simulator and Python Cocotb
case class DMAReader2DTestWrapper() extends Component {
  val axiDataWidth = 64
  val outLanes = 1
  val axiConfig = Axi4Config(
    addressWidth = 32,
    dataWidth = axiDataWidth,
    idWidth = 4
  )
  
  val dataType = SInt(16 bits)
  val io = new Bundle {
    val cmd = slave(Stream(FetchRequest2D(32)))
    val axiMaster = master(Axi4(axiConfig))
    val outStream = master(Tensor(dataType, Seq(4, 4), outLanes))
  }
  val reader = DMAReader2D(dataType, Seq(4, 4), outLanes, axiConfig)
  reader.io.cmd << io.cmd
  io.outStream <> reader.io.outStream
  
  io.axiMaster.ar << reader.io.axiMaster.ar
  reader.io.axiMaster.r << io.axiMaster.r
  
  io.axiMaster.aw.valid := False
  io.axiMaster.aw.payload.assignDontCare()
  io.axiMaster.w.valid := False
  io.axiMaster.w.payload.assignDontCare()
  io.axiMaster.b.ready := False
}

// Unaligned multi-lane wrapper: I8 elements, outLanes = 4, 64-bit AXI.
// With base = 0x1004 and stride = 12, even rows start 4 bytes into an AXI
// beat (headSkipElems = 4, group-aligned with outLanes = 4).
case class DMAReader2DUnalignedTestWrapper() extends Component {
  val axiDataWidth = 64
  val outLanes = 4
  val axiConfig = Axi4Config(
    addressWidth = 32,
    dataWidth = axiDataWidth,
    idWidth = 4
  )

  val dataType = I8()
  val io = new Bundle {
    val cmd = slave(Stream(FetchRequest2D(32)))
    val axiMaster = master(Axi4(axiConfig))
    val outStream = master(Tensor(dataType, Seq(4, 8), outLanes))
  }
  val reader = DMAReader2D(dataType, Seq(4, 8), outLanes, axiConfig)
  reader.io.cmd << io.cmd
  io.outStream <> reader.io.outStream

  io.axiMaster.ar << reader.io.axiMaster.ar
  reader.io.axiMaster.r << io.axiMaster.r

  io.axiMaster.aw.valid := False
  io.axiMaster.aw.payload.assignDontCare()
  io.axiMaster.w.valid := False
  io.axiMaster.w.payload.assignDontCare()
  io.axiMaster.b.ready := False
}

class DMAReader2DTest extends AnyFunSuite {

  test("DMAReader2D Hardware Sim") {
    SimConfig.withVerilator.withWave.workspacePath("sim_build").compile {
      val dut = DMAReader2DTestWrapper()
      dut.setDefinitionName("DMAReader2DTestComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      dut.io.cmd.valid #= false
      dut.io.cmd.baseAddress #= 0
      dut.io.cmd.stride #= 0
      dut.io.cmd.patchWidth #= 0
      dut.io.cmd.patchHeight #= 0
      dut.io.outStream.stream.ready #= true
      
      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4)
      )
      
      val baseAddr = 0x1000
      memorySim.memory.writeBigInt(baseAddr + 0, BigInt("0004000300020001", 16), 8)
      memorySim.memory.writeBigInt(baseAddr + 8, BigInt("0008000700060005", 16), 8)
      memorySim.memory.writeBigInt(baseAddr + 16, BigInt("000C000B000A0009", 16), 8)
      memorySim.memory.writeBigInt(baseAddr + 24, BigInt("0010000F000E000D", 16), 8)
      
      memorySim.start()
      dut.clockDomain.waitSampling(5)
      
      dut.io.cmd.valid #= true
      dut.io.cmd.baseAddress #= baseAddr
      dut.io.cmd.stride #= 8 
      dut.io.cmd.patchWidth #= 0 
      dut.io.cmd.patchHeight #= 4 
      
      fork {
        dut.clockDomain.waitSamplingWhere(dut.io.cmd.ready.toBoolean)
        dut.io.cmd.valid #= false
      }
      
      var validCount = 0
      var timeout = 0
      while (validCount < 16 && timeout < 500) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          validCount += 1
          val p0 = dut.io.outStream.stream.payload(0).asInstanceOf[SInt].toBigInt
        }
        dut.clockDomain.waitSampling()
        timeout += 1
      }
      
      assert(validCount == 16, s"Expected 16 valid outputs, got $validCount")
    }
  }

  test("DMAReader2D outLanes=4 with unaligned row starts") {
    SimConfig.withVerilator.withWave.workspacePath("sim_build").compile {
      val dut = DMAReader2DUnalignedTestWrapper()
      dut.setDefinitionName("DMAReader2DUnalignedTestComp")
      dut
    }.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)

      dut.io.cmd.valid #= false
      dut.io.cmd.baseAddress #= 0
      dut.io.cmd.stride #= 0
      dut.io.cmd.patchWidth #= 0
      dut.io.cmd.patchHeight #= 0
      dut.io.outStream.stream.ready #= false

      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 4)
      )

      // Row r lives at base + r*stride, 8 bytes wide. base % 8 = 4 and
      // stride % 8 = 4, so rows alternate between 4-byte-misaligned and
      // aligned starts; every start is group-aligned (4-byte) for outLanes=4.
      val base = 0x1004
      val stride = 12
      val expected = Seq.tabulate(4, 8)((r, c) => r * 8 + c)

      val bytes = Array.fill[Byte](0x40)(0)
      def putByte(addr: Int, value: Int): Unit = bytes(addr - 0x1000) = value.toByte
      for (r <- 0 until 4; c <- 0 until 8) putByte(base + r * stride + c, expected(r)(c))
      for (w <- 0 until 0x40 / 8) {
        var word = BigInt(0)
        for (k <- 0 until 8) word |= BigInt(bytes(w * 8 + k) & 0xFF) << (8 * k)
        memorySim.memory.writeBigInt(0x1000 + w * 8, word, 8)
      }

      memorySim.start()
      dut.clockDomain.waitSampling(5)

      dut.io.cmd.valid #= true
      dut.io.cmd.baseAddress #= base
      dut.io.cmd.stride #= stride
      dut.io.cmd.patchWidth #= 8
      dut.io.cmd.patchHeight #= 4
      dut.io.outStream.stream.ready #= true

      fork {
        dut.clockDomain.waitSamplingWhere(dut.io.cmd.ready.toBoolean)
        dut.io.cmd.valid #= false
      }

      val got = scala.collection.mutable.ArrayBuffer[Int]()
      var timeout = 0
      while (got.length < 32 && timeout < 2000) {
        if (dut.io.outStream.stream.valid.toBoolean && dut.io.outStream.stream.ready.toBoolean) {
          for (l <- 0 until 4) {
            got += dut.io.outStream.stream.payload(l).asInstanceOf[SInt].toInt
          }
        }
        dut.clockDomain.waitSampling()
        timeout += 1
      }

      val flatExpected = expected.flatten
      assert(timeout < 2000, s"Timeout waiting for 32 elements: got ${got.length} (${got.mkString(",")})")
      assert(got.toSeq == flatExpected, s"Expected $flatExpected, got ${got.toSeq}")
    }
  }

  test("Generate Verilog for Python Cocotb") {
    SpinalConfig().generateVerilog {
      val dut = DMAReader2DTestWrapper()
      dut.setDefinitionName("DMAReader2DTestComp")
      dut
    }
  }
}
