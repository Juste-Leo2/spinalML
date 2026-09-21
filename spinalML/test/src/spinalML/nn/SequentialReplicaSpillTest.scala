// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.nn

import org.scalatest.funsuite.AnyFunSuite

/**
 * R3/P0b end-to-end: spilling Linear inside a real Sequential + Accelerator,
 * DDR-backed throughout, bit-exact against the full `ModelReplica` oracle.
 * Driver shared with the Conv suite via ReplicaSpillE2E (P1).
 */
class SequentialReplicaSpillTest extends AnyFunSuite {

  test("R3 e2e I8 spill P=2 bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(1, 8), isInt = true, label = "I8-K8-P2")
  }

  test("R3 e2e BF16 spill P=2 bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(1, 8), isInt = false, label = "BF16-K8-P2")
  }

  test("P0b e2e I8 spill P=4 bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Linear(inFeatures = 16, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(1, 16), isInt = true, label = "I8-K16-P4")
  }

  test("P0b e2e BF16 spill P=4 bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Linear(inFeatures = 16, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(1, 16), isInt = false, label = "BF16-K16-P4")
  }

  test("P0b e2e I8 spill P=2 with M=2 rows bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(2, 8), isInt = true, label = "I8-M2-K8-P2")
  }

  test("P0b e2e deep-node spill P=2 via StreamTap bit-exact vs ModelReplica") {
    // [dense 4->4, spill 4->4 Ks=2]: the spill sits at node 1, its A is the
    // layer-0 output replayed on-chip (tap), layers share one W region.
    ReplicaSpillE2E.runReplicaCase(
      Seq(Linear(inFeatures = 4, outFeatures = 4, weightLanes = 2),
        Linear(inFeatures = 4, outFeatures = 4, weightLanes = 2, spillKSlice = 2)),
      Seq(1, 4), isInt = true, label = "I8-TAP-P2")
  }

  test("P0b e2e I8 spill K=64 P=8 at scale bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Linear(inFeatures = 64, outFeatures = 4, weightLanes = 2, spillKSlice = 8)),
      Seq(1, 64), isInt = true, label = "I8-K64-P8")
  }

  test("P0b e2e I8 spill P=2 back-to-back rerun bit-exact vs ModelReplica") {
    ReplicaSpillE2E.runReplicaCase(
      Seq(Linear(inFeatures = 8, outFeatures = 4, weightLanes = 2, spillKSlice = 4)),
      Seq(1, 8), isInt = true, label = "I8-K8-P2-RERUN", runs = 2)
  }
}
