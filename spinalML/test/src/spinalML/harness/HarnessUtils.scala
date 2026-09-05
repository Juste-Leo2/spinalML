// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.harness

import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.sim.SparseMemory
import spinal.lib.bus.amba4.axilite.AxiLite4
import spinalML.utils.MemLayout

object MemoryHarness {

  def bf16Bits(f: Float): Int = (java.lang.Float.floatToIntBits(f) >>> 16) & 0xFFFF

  def word(elems: Seq[Int]): BigInt =
    elems.zipWithIndex.foldLeft(BigInt(0))((acc, e) => acc | (BigInt(e._1 & 0xFFFF) << (16 * e._2)))

  def packFloats(values: Seq[Float]): Seq[BigInt] =
    values.grouped(4).map(g => word(g.map(bf16Bits).padTo(4, 0))).toSeq

  def padded(elems: Seq[Float]): Seq[Float] = {
    val capacity = MemLayout.alignToBeat(MemLayout.regionBytes(elems.length, 16), 8) / 2
    elems ++ Seq.fill(capacity - elems.length)(0.0f)
  }

  def packBytes(bytes: Seq[Int]): Seq[BigInt] = {
    val capacity = MemLayout.alignToBeat(bytes.length, 8)
    val padded = bytes ++ Seq.fill(capacity - bytes.length)(0)
    padded.grouped(8).map { g =>
      g.zipWithIndex.foldLeft(BigInt(0))((acc, e) =>
        acc | (BigInt(e._1 & 0xFF) << (8 * e._2)))
    }.toSeq
  }

  def writeWords(mem: SparseMemory, base: Long, words: Seq[BigInt]): Unit =
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)
}

object StreamingHarness {

  def writeAxiLite(bus: AxiLite4, cd: ClockDomain)(addr: BigInt, data: BigInt): Unit = {
    bus.aw.valid #= true
    bus.aw.payload.addr #= addr
    bus.w.valid #= true
    bus.w.payload.data #= data
    bus.w.payload.strb #= 0xF
    bus.b.ready #= true
    cd.waitSamplingWhere(bus.aw.ready.toBoolean && bus.w.ready.toBoolean)
    bus.aw.valid #= false
    bus.w.valid #= false
    cd.waitSamplingWhere(bus.b.valid.toBoolean)
    bus.b.ready #= false
    cd.waitSampling()
  }

  def readAxiLite(bus: AxiLite4, cd: ClockDomain)(addr: BigInt): BigInt = {
    bus.ar.valid #= true
    bus.ar.payload.addr #= addr
    bus.r.ready #= true
    cd.waitSamplingWhere(bus.ar.ready.toBoolean)
    bus.ar.valid #= false
    cd.waitSamplingWhere(bus.r.valid.toBoolean)
    val data = bus.r.payload.data.toBigInt
    bus.r.ready #= false
    cd.waitSampling()
    data
  }
}
