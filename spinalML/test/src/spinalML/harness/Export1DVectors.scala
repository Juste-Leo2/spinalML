// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.harness

import java.nio.file.{Files, Paths}
import spinal.core._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.nn._
import spinalML.dtypes._
import spinalML.replica.{HWArithmetic, ModelReplica, WeightMemoryLayout}

object Export1DVectors extends App {
  println("[Export1DVectors] Computing deterministic test vectors for Universal1DDemo...")

  var weightBytes: Array[Byte] = null
  var imageBytes: Array[Byte] = null
  var inInts: Seq[Long] = null
  var logits: Seq[Double] = null

  SpinalConfig(targetDirectory = "hw_build/tang-primer-20k/tmp").generateVerilog(new Component {
    setDefinitionName("DummyExportContext")
    val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)

    // Define the exact Universal1DDemo accelerator specification
    val modelSpec = Seq(
      Conv1D(inChannels = 2, outChannels = 2, kernelSize = 3, customType = Some(I16())),
      AvgPool1D(poolSize = 2, stride = 2),
      ReLU(),
      Flatten(),
      Linear(inFeatures = 6, outFeatures = 2, customType = Some(I16())),
      Requantize(shift = 1, targetType = I8())
    )
    val inputShape = Seq(8, 2)
    val pipelineDtype = I8()

    // 1. Pack deterministic weights
    val packed = WeightMemoryLayout.buildDeterministicWeights(modelSpec, pipelineDtype, axiConfig)
    val weightWords = packed.words

    // 2. Pack deterministic input
    val inElems = inputShape.product
    inInts = (0 until inElems).map { idx =>
      (((idx * 7 + 3) % 15) - 7).toLong
    }
    val imgWords = MemoryHarness.packBytes(inInts.map(_.toInt))
    val inputTensor = ModelReplica.IntTensor(inputShape, inInts, 8)

    // 3. Oracle forward
    val replicaResult = ModelReplica.forwardWithTrace(modelSpec, inputShape, inputTensor, packed)
    logits = replicaResult.logits

    // Convert BigInt 64-bit words to byte arrays (LSB-first per word, matching AxiReadMem & UartBridge)
    def wordsToBytes(words: Seq[BigInt]): Array[Byte] = {
      val buf = new Array[Byte](words.length * 8)
      for (i <- words.indices) {
        val w = words(i)
        for (b <- 0 until 8) {
          buf(i * 8 + b) = ((w >> (8 * b)) & 0xFF).toByte
        }
      }
      buf
    }

    weightBytes = wordsToBytes(weightWords)
    imageBytes = wordsToBytes(imgWords)
  })

  val outDir = Paths.get("hw_build/tang-primer-20k")
  Files.createDirectories(outDir)

  // Write JSON
  val jsonFile = outDir.resolve("test_vectors.json")
  val jsonText = s"""{
  "model": "Universal1DDemo",
  "out_count": ${logits.length},
  "img_base": 65536,
  "weight_base": 131072,
  "input_ints": [${inInts.mkString(", ")}],
  "image_bytes_hex": "${imageBytes.map(b => "%02x".format(b & 0xFF)).mkString}",
  "weight_bytes_hex": "${weightBytes.map(b => "%02x".format(b & 0xFF)).mkString}",
  "expected_logits": [${logits.map(_.toInt).mkString(", ")}],
  "expected_bytes_hex": "${logits.map(l => "%02x".format(l.toInt & 0xFF)).mkString}"
}"""

  Files.write(jsonFile, jsonText.getBytes("UTF-8"))
  println(s"[Export1DVectors] Exported test vectors to ${jsonFile.toAbsolutePath}")
  println(s"  Image bytes  : ${imageBytes.length} bytes (hex: ${imageBytes.map(b => "%02x".format(b & 0xFF)).mkString})")
  println(s"  Weight bytes : ${weightBytes.length} bytes")
  println(s"  Expected out : ${logits.map(_.toInt).mkString(", ")} (hex: ${logits.map(l => "%02x".format(l.toInt & 0xFF)).mkString})")
}
