// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.memory.litedram

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.dtypes.I8
import spinalML.nn.Accelerator
import spinalML.io.DramSoCTop

class LiteDramBlackBoxTest extends AnyFunSuite {

  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

  test("DramSoCTop: elaborates with DdrAdapter + LiteDRAM bridge") {
    val targetDir = "out/test_dram_soc"
    SpinalConfig(targetDirectory = targetDir).generateVerilog(
      new DramSoCTop(
        acceleratorFactory = () => new Accelerator(
          dataType = I8(),
          inputShape = Seq(4, 4, 1),
          modelSpec = Seq(spinalML.nn.Flatten()),
          axiConfig = axiConfig
        ),
        axiConfig = axiConfig
      )
    )
    val content = scala.io.Source.fromFile(s"$targetDir/DramSoCTop.v").mkString
    assert(content.contains("litedram_core"), "DramSoCTop Verilog must instantiate litedram_core")
    assert(content.contains("DdrAdapter"), "DramSoCTop Verilog must contain DdrAdapter")
    assert(content.contains("ddram_a"), "DramSoCTop Verilog must expose DDR pads")
  }
}
