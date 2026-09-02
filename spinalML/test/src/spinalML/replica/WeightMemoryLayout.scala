package spinalML.replica

import spinal.core.HardType
import spinal.core.Data
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.utils.MemLayout
import HWArithmetic._

/**
 * Single source of truth for generating deterministic test weights and packing them
 * into DDR memory matching Sequential.scala's exact AXI beat alignment.
 */
object WeightMemoryLayout {

  case class LayerWeightInfo(
    layerIdx: Int,
    name: String,
    weightElements: Int,
    biasElements: Int,
    weightOffset: Int,
    biasOffset: Int,
    weightValues: Seq[F],
    biasValues: Seq[F]
  )

  case class PackedWeightsResult(
    words: Seq[BigInt],
    layers: Seq[LayerWeightInfo],
    totalBytes: Int
  )

  /**
   * Builds deterministic weights for a given sequence of LayerSpecs.
   * Mirrors Sequential.scala layout:
   *   For each layer:
   *     - if weights: alignToBeat(offset), write weights, offset += regionBytes
   *     - if bias: alignToBeat(offset), write bias, offset += regionBytes
   */
  def buildDeterministicWeights(
    layers: Seq[LayerSpec],
    pipelineDataType: HardType[Data],
    axiConfig: Axi4Config,
    expBits: Int = 8,
    mantBits: Int = 7
  ): PackedWeightsResult = {
    val beatBytes = axiConfig.dataWidth / 8
    var currentOffset = 0
    val layerInfos = scala.collection.mutable.ArrayBuffer[LayerWeightInfo]()

    // Memory buffer as byte array
    val memoryBytes = scala.collection.mutable.ArrayBuffer.fill(65536)(0.toByte)

    for (i <- layers.indices) {
      val layer = layers(i)
      val wShape = layer.getWeightShape()
      val bShape = layer.getBiasShape()

      val wElems = if (wShape.head > 0) wShape.product else 0
      val bElems = if (bShape.head > 0) bShape.product else 0

      var wOffset = -1
      var bOffset = -1
      var wValues = Seq[F]()
      var bValues = Seq[F]()

      if (wElems > 0) {
        currentOffset = MemLayout.alignToBeat(currentOffset, beatBytes)
        wOffset = currentOffset

        // Generate small non-zero deterministic floats between 0.125 and 0.5
        wValues = (0 until wElems).map { idx =>
          val floatVal = (((idx % 7) + 1) * 0.0625).toFloat
          fromDouble(floatVal, expBits, mantBits)
        }

        // Pack 16-bit BF16 into bytes
        for (idx <- 0 until wElems) {
          val f = wValues(idx)
          val bits = ((if (f.s) 1 else 0) << 15) | ((f.e & 0xFF) << 7) | (f.m & 0x7F)
          val addr = wOffset + idx * 2
          while (memoryBytes.length <= addr + 2) memoryBytes += 0.toByte
          memoryBytes(addr) = (bits & 0xFF).toByte
          memoryBytes(addr + 1) = ((bits >> 8) & 0xFF).toByte
        }

        val wBytes = MemLayout.regionBytes(wElems, 16)
        currentOffset += wBytes
      }

      if (bElems > 0) {
        currentOffset = MemLayout.alignToBeat(currentOffset, beatBytes)
        bOffset = currentOffset

        bValues = (0 until bElems).map { idx =>
          val floatVal = (((idx % 5) + 1) * 0.03125).toFloat
          fromDouble(floatVal, expBits, mantBits)
        }

        for (idx <- 0 until bElems) {
          val f = bValues(idx)
          val bits = ((if (f.s) 1 else 0) << 15) | ((f.e & 0xFF) << 7) | (f.m & 0x7F)
          val addr = bOffset + idx * 2
          while (memoryBytes.length <= addr + 2) memoryBytes += 0.toByte
          memoryBytes(addr) = (bits & 0xFF).toByte
          memoryBytes(addr + 1) = ((bits >> 8) & 0xFF).toByte
        }

        val bBytes = MemLayout.regionBytes(bElems, 16)
        currentOffset += bBytes
      }

      layerInfos += LayerWeightInfo(
        layerIdx = i,
        name = layer.getClass.getSimpleName,
        weightElements = wElems,
        biasElements = bElems,
        weightOffset = wOffset,
        biasOffset = bOffset,
        weightValues = wValues,
        biasValues = bValues
      )
    }

    // Convert byte array into 64-bit AXI words
    val alignedTotalBytes = MemLayout.alignToBeat(currentOffset, beatBytes)
    val words = scala.collection.mutable.ArrayBuffer[BigInt]()
    for (i <- 0 until alignedTotalBytes by 8) {
      var word = BigInt(0)
      for (b <- 0 until 8) {
        val byteVal = if (i + b < memoryBytes.length) memoryBytes(i + b).toInt & 0xFF else 0
        word |= (BigInt(byteVal) << (8 * b))
      }
      words += word
    }

    PackedWeightsResult(
      words = words.toSeq,
      layers = layerInfos.toSeq,
      totalBytes = alignedTotalBytes
    )
  }
}
