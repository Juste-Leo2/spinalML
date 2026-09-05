// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.ops

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinalML.tensors.Tensor
import spinalML.dtypes.{I8, FP8_E4M3, I16, BF16}

class FlattenTest extends AnyFunSuite {
  test("Flatten tensor dimensions") {
    val compiled = SimConfig.withWave.compile {
      new Component {
        val io = new Bundle {
          val a = slave(Tensor(I8(), Seq(2, 3, 4), lanes = 2))
          val c = master(Tensor(I8(), Seq(24), lanes = 2))
        }
        io.c <> flatten(io.a)
      }
    }
    
    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      
      assert(dut.io.c.shape == Seq(24))
      
      // Just verifying connectivity
      dut.io.a.stream.valid #= true
      dut.io.c.stream.ready #= true
      for(i <- 0 until 2) dut.io.a.stream.payload(i) #= i
      dut.clockDomain.waitSampling()
      
      assert(dut.io.c.stream.valid.toBoolean == true)
      assert(dut.io.c.stream.payload(0).toInt == 0)
      assert(dut.io.c.stream.payload(1).toInt == 1)
    }
  }

  val compileTypes = Seq(
    ("I8", () => I8()),
    ("FP8", () => FP8_E4M3()),
    ("I16", () => I16()),
    ("BF16", () => BF16())
  )

  for ((name, dt) <- compileTypes) {
    test(s"Test Flatten compilation on $name") {
      SpinalConfig().generateVerilog(new Component {
        val a = slave(Tensor(dt(), Seq(2, 3, 4), lanes = 2))
        val c = master(Tensor(dt(), Seq(24), lanes = 2))
        c <> flatten(a)
      })
    }
  }
}
