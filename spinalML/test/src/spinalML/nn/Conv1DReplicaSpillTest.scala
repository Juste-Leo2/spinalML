// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite

/**
 * P2-5 end-to-end: spilling Conv1D inside a real Sequential + Accelerator,
 * DDR-backed throughout, bit-exact against the full `ModelReplica` oracle
 * (slice-transposed layout P2-4 + pass fold). Driver shared via
 * ReplicaSpillE2E. Geometry: [6,4] in, K=2 -> [5,2] out (5 windows),
 * KFull=8, Ks=4, P=2.
 */
class Conv1DReplicaSpillTest extends AnyFunSuite {

  test("P2-5 e2e I8 conv1d spill P=2 bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Conv1D(inChannels = 4, outChannels = 2, kernelSize = 2, weightLanes = 4, spillKSlice = 4)),
      Seq(6, 4), isInt = true, label = "CONV1D-I8-P2")
  }

  test("P2-5 e2e BF16 conv1d spill P=2 bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Conv1D(inChannels = 4, outChannels = 2, kernelSize = 2, weightLanes = 4, spillKSlice = 4)),
      Seq(6, 4), isInt = false, label = "CONV1D-BF16-P2")
  }

  test("P2-5 e2e I8 conv1d spill P=1 single-pass regression vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Conv1D(inChannels = 4, outChannels = 2, kernelSize = 2, weightLanes = 4, spillKSlice = 8)),
      Seq(6, 4), isInt = true, label = "CONV1D-I8-P1")
  }

  test("P2-5 e2e I8 conv1d dense (no spill) via the same driver") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Conv1D(inChannels = 4, outChannels = 2, kernelSize = 2, weightLanes = 4)),
      Seq(6, 4), isInt = true, label = "CONV1D-DENSE")
  }
}
