package spinalML.replica

import spinal.core.HardType
import spinal.core.Data
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes.FloatML
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
    biasValues: Seq[F],
    weightInts: Seq[Long] = Nil,
    biasInts: Seq[Long] = Nil,
    weightDtype: HardType[Data] = null,
    biasDtype: HardType[Data] = null
  )

  case class PackedWeightsResult(
    words: Seq[BigInt],
    layers: Seq[LayerWeightInfo],
    totalBytes: Int
  )

  def packRawBits(rawBits: Seq[Long], elemBits: Int): Seq[Byte] = {
    elemBits match {
      case 4 =>
        rawBits.grouped(2).map { g =>
          val low = (g.head & 0xF).toInt
          val high = if (g.length > 1) (g(1) & 0xF).toInt else 0
          (low | (high << 4)).toByte
        }.toSeq
      case 8 =>
        rawBits.map(b => (b & 0xFF).toByte)
      case 16 =>
        rawBits.flatMap(w => Seq((w & 0xFF).toByte, ((w >> 8) & 0xFF).toByte))
      case 32 =>
        rawBits.flatMap(w => Seq(
          (w & 0xFF).toByte,
          ((w >> 8) & 0xFF).toByte,
          ((w >> 16) & 0xFF).toByte,
          ((w >> 24) & 0xFF).toByte
        ))
      case _ =>
        throw new IllegalArgumentException(s"Unsupported elemBits: $elemBits")
    }
  }

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

    val nodeTypes = scala.collection.mutable.ArrayBuffer[HardType[Data]](pipelineDataType)
    for (i <- layers.indices) {
      val l = layers(i)
      val outType = l match {
        case ad: Add => nodeTypes(ad.a)
        case cc: Concat => nodeTypes(cc.a)
        case _ => l.outType(nodeTypes(i))
      }
      nodeTypes += outType
    }

    for (i <- layers.indices) {
      val layer = layers(i)
      val layerInputType = nodeTypes(i)
      val wType = layer.weightType(layerInputType)
      val bType = layer.outType(layerInputType)

      val wShape = layer.getWeightShape()
      val bShape = layer.getBiasShape()

      val wElems = if (wShape.head > 0) wShape.product else 0
      val bElems = if (bShape.head > 0) bShape.product else 0

      var wOffset = -1
      var bOffset = -1
      var wValues = Seq[F]()
      var bValues = Seq[F]()
      var wInts = Seq[Long]()
      var bInts = Seq[Long]()

      if (wElems > 0) {
        currentOffset = MemLayout.alignToBeat(currentOffset, beatBytes)
        wOffset = currentOffset

        val wData = wType()
        val wElemBits = wData.getBitsWidth
        val isFloat = wData.isInstanceOf[FloatML]

        val rawBits: Seq[Long] = if (isFloat) {
          val fType = wData.asInstanceOf[FloatML]
          val eW = fType.expBits
          val mW = fType.mantBits
          wValues = (0 until wElems).map { idx =>
            val floatVal = (((idx % 7) + 1) * 0.0625).toFloat
            fromDouble(floatVal, eW, mW)
          }
          wValues.map { f =>
            val sign = if (f.s) 1L else 0L
            (sign << (eW + mW)) | ((f.e.toLong & ((1L << eW) - 1)) << mW) | (f.m.toLong & ((1L << mW) - 1))
          }
        } else {
          // Integer domain (e.g. I4, I8, I16)
          wInts = (0 until wElems).map { idx =>
            ((idx % 7) + 1).toLong
          }
          val mask = if (wElemBits >= 64) -1L else (1L << wElemBits) - 1
          wValues = wInts.map(v => fromSInt(v, wElemBits, expBits, mantBits))
          wInts.map(v => v & mask)
        }

        val packedBytes = packRawBits(rawBits, wElemBits)
        while (memoryBytes.length < wOffset + packedBytes.length) memoryBytes += 0.toByte
        for (idx <- packedBytes.indices) {
          memoryBytes(wOffset + idx) = packedBytes(idx)
        }

        val wBytes = MemLayout.regionBytes(wElems, wElemBits)
        currentOffset = MemLayout.alignToBeat(wOffset + wBytes, beatBytes)
      }

      if (bElems > 0) {
        currentOffset = MemLayout.alignToBeat(currentOffset, beatBytes)
        bOffset = currentOffset

        val bData = bType()
        val bElemBits = bData.getBitsWidth
        val isFloat = bData.isInstanceOf[FloatML]

        val rawBits: Seq[Long] = if (isFloat) {
          val fType = bData.asInstanceOf[FloatML]
          val eW = fType.expBits
          val mW = fType.mantBits
          bValues = (0 until bElems).map { idx =>
            val floatVal = (((idx % 5) + 1) * 0.03125).toFloat
            fromDouble(floatVal, eW, mW)
          }
          bValues.map { f =>
            val sign = if (f.s) 1L else 0L
            (sign << (eW + mW)) | ((f.e.toLong & ((1L << eW) - 1)) << mW) | (f.m.toLong & ((1L << mW) - 1))
          }
        } else {
          bInts = (0 until bElems).map { idx =>
            ((idx % 5) + 1).toLong
          }
          val mask = if (bElemBits >= 64) -1L else (1L << bElemBits) - 1
          bValues = bInts.map(v => fromSInt(v, bElemBits, expBits, mantBits))
          bInts.map(v => v & mask)
        }

        val packedBytes = packRawBits(rawBits, bElemBits)
        while (memoryBytes.length < bOffset + packedBytes.length) memoryBytes += 0.toByte
        for (idx <- packedBytes.indices) {
          memoryBytes(bOffset + idx) = packedBytes(idx)
        }

        val bBytes = MemLayout.regionBytes(bElems, bElemBits)
        currentOffset = MemLayout.alignToBeat(bOffset + bBytes, beatBytes)
      }

      layerInfos += LayerWeightInfo(
        layerIdx = i,
        name = layer.getClass.getSimpleName,
        weightElements = wElems,
        biasElements = bElems,
        weightOffset = wOffset,
        biasOffset = bOffset,
        weightValues = wValues,
        biasValues = bValues,
        weightInts = wInts,
        biasInts = bInts,
        weightDtype = wType,
        biasDtype = bType
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
