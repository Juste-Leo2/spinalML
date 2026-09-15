// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.io

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.examples.Mnistw4a8
import spinalML.dtypes.BF16
import spinalML.nn.{Accelerator, Flatten}

import spinalML.Target

/**
 * Elaboration of the reference UART SoC top (UartSoC.v, used by
 * tests/python/test_uart_soc.py). Same configuration as UartSoCGen and the
 * reference top.v: 64-bit AXI beats, 27 MHz @ 115200, MNIST W4A8.
 */
class UartSoCTest extends AnyFunSuite {
  test("uart_soc_toplevel (FPGA default with BOOT reset)") {
    val cfg = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    val report = SpinalConfig(
      headerWithDate = true,
      rtlHeader = "/* spinalML | Copyright (c) 2026 Léonard Adamo (Juste-Leo2) | SPDX-License-Identifier: MIT */"
    ).generateVerilog(new UartSoC(() => new Mnistw4a8(axiConfig = cfg), axiConfig = cfg, target = Target.FPGA()))

    // FPGA target includes BOOT clock domain / power-on reset logic
    assert(report.toplevel.isInstanceOf[UartSoC[_]])
  }

  test("uart_soc_toplevel (ASIC target without BOOT reset, SramAsicAdapter)") {
    val cfg = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    val report = SpinalConfig(
      headerWithDate = true,
      rtlHeader = "/* spinalML | Copyright (c) 2026 Léonard Adamo (Juste-Leo2) | SPDX-License-Identifier: MIT */"
    ).generateVerilog(new UartSoC(() => new Mnistw4a8(axiConfig = cfg), axiConfig = cfg, target = Target.ASIC()))

    val soc = report.toplevel.asInstanceOf[UartSoC[_]]
    assert(soc.target.isAsic, "Target should be ASIC")
    assert(soc.soc.mem.isInstanceOf[spinalML.memory.SramAsicAdapter], "ASIC target must select SramAsicAdapter by default")
  }

  test("uart_soc_toplevel refuses non-8-bit output elements") {
    val cfg = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    val ex = intercept[Exception] {
      SpinalConfig().generateVerilog {
        new UartSoC(
          acceleratorFactory = () => new Accelerator(
            dataType = BF16(),
            inputShape = Seq(4, 4, 1),
            modelSpec = Seq(Flatten()),
            axiConfig = cfg
          ),
          axiConfig = cfg,
          target = Target.FPGA()
        )
      }
    }
    val msg = Option(ex.getMessage).getOrElse("") + Option(ex.getCause).map(_.getMessage).getOrElse("")
    assert(msg.contains("8-bit"),
      s"Elaboration must reject 16-bit logits with a clear message, got: $ex / ${ex.getCause}")
  }
}
