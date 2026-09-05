// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML

import spinal.core._

object SpinalMLConfig {
  def apply(targetDirectory: String = ".") = SpinalConfig(
    targetDirectory = targetDirectory,
    headerWithDate = true,
    rtlHeader =
      """// -----------------------------------------------------------------------------
        |// spinalML - Hardware Machine Learning Accelerator
        |// Copyright (c) 2026 Léonard Adamo (Juste-Leo2)
        |// Generated with SpinalHDL
        |// SPDX-License-Identifier: MIT
        |// -----------------------------------------------------------------------------
        |""".stripMargin
  )
}
