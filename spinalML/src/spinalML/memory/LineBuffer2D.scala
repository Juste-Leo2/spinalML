// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory

import spinal.core._
import spinal.lib._

/**
 * BRAM delay-line: delays its input stream by exactly `depth` accepted beats.
 * Circular Mem + readSync guarantees Block RAM inference on FPGA and maps to
 * SRAM macros / clean synchronous memory on ASIC.
 * Read address = ptr + 1 so that the popped value is aligned with the current
 * input beat even under arbitrary valid/ready stalls.
 */
case class LineBuffer2D[T <: Data](dataType: HardType[T], depth: Int) extends Component {
  require(depth >= 1, s"LineBuffer2D depth must be >= 1, got $depth")

  val io = new Bundle {
    val push = slave Flow (dataType())
    val pop  = master Flow (dataType())
  }

  if (depth == 1) {
    val reg = Reg(dataType()) init (dataType().getZero)
    val regValid = RegNext(io.push.valid) init (False)
    when(io.push.valid) {
      reg := io.push.payload
    }
    io.pop.valid := regValid
    io.pop.payload := reg
  } else {
    val mem = Mem(dataType, depth)
    mem.init(Seq.fill(depth)(dataType().getZero))

    val ptr = Counter(depth)
    val rdAddr = Mux(ptr.value === depth - 1, U(0, log2Up(depth) bits), ptr.value + 1)

    mem.write(ptr.value, io.push.payload, enable = io.push.valid)
    when(io.push.valid) {
      ptr.increment()
    }

    io.pop.valid := RegNext(io.push.valid) init (False)
    io.pop.payload := mem.readSync(rdAddr, enable = io.push.valid)
  }
}
