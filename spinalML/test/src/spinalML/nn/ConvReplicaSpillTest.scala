// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite

/**
 * P1-5 end-to-end: spilling Conv2D inside a real Sequential + Accelerator,
 * DDR-backed throughout, bit-exact against the full `ModelReplica` oracle
 * (slice-transposed layout P1-4 + pass fold). Driver shared via
 * ReplicaSpillE2E. Geometry: [6,6,2] in, K=2 -> [5,5,2] out (25 windows),
 * KFull=8, Ks=4, P=2.
 */
class ConvReplicaSpillTest extends AnyFunSuite {

  test("P1-5 e2e I8 conv spill P=2 bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 4)),
      Seq(6, 6, 2), isInt = true, label = "CONV-I8-P2")
  }

  test("P1-5 e2e BF16 conv spill P=2 bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 4)),
      Seq(6, 6, 2), isInt = false, label = "CONV-BF16-P2")
  }

  test("P1-5 e2e I8 conv spill P=1 single-pass regression vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2, spillKSlice = 8)),
      Seq(6, 6, 2), isInt = true, label = "CONV-I8-P1")
  }

  test("P1-5 e2e I8 dense conv (no spill) regression via the same driver") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Conv2D(inChannels = 2, outChannels = 2, kernelSize = 2)),
      Seq(6, 6, 2), isInt = true, label = "CONV-DENSE", temporal = 0)
  }

  test("P1-5 e2e I8 dense conv inC=1 (no spill) regression via the same driver") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Conv2D(inChannels = 1, outChannels = 2, kernelSize = 2)),
      Seq(6, 6, 1), isInt = true, label = "CONV-DENSE-C1", temporal = 0)
  }
}
