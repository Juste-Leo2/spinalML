// Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

package spinalML.interfaces

import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axis._
import spinalML.tensors.Tensor

// Bridge from standard AXI4-Stream to our custom Tensor Stream
case class Axi4StreamToTensor[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  lanes: Int,
  axiDataWidth: Int // Configurable AXI width in bits (e.g., 32, 64)
) extends Component {
  
  val tensorChunkWidth = dataType.getBitsWidth * lanes
  require(axiDataWidth >= tensorChunkWidth, s"AXI bus width ($axiDataWidth) must be at least as wide as the tensor chunk width ($tensorChunkWidth)")
  
  val io = new Bundle {
    val axis = slave(Axi4Stream(
      Axi4StreamConfig(
        dataWidth = axiDataWidth / 8, // AXI dataWidth is configured in bytes
        useLast = true
      )
    ))
    val tensor = master(Tensor(dataType, shape, lanes))
  }
  
  // Handshake connection
  io.tensor.stream.valid := io.axis.valid
  io.axis.ready := io.tensor.stream.ready
  
  // Data translation (cast AXI Bits to Tensor data)
  val slicedData = io.axis.data(tensorChunkWidth - 1 downto 0)
  
  for(i <- 0 until lanes) {
    val laneBits = slicedData(i * dataType.getBitsWidth, dataType.getBitsWidth bits)
    io.tensor.stream.payload(i).assignFromBits(laneBits)
  }
}

// Bridge from our custom Tensor Stream back to standard AXI4-Stream
case class TensorToAxi4Stream[T <: Data](
  dataType: HardType[T],
  shape: Seq[Int],
  lanes: Int,
  axiDataWidth: Int // Configurable AXI width in bits (e.g., 32, 64)
) extends Component {
  
  val tensorChunkWidth = dataType.getBitsWidth * lanes
  require(axiDataWidth >= tensorChunkWidth, s"AXI bus width ($axiDataWidth) must be at least as wide as the tensor chunk width ($tensorChunkWidth)")
  
  val io = new Bundle {
    val tensor = slave(Tensor(dataType, shape, lanes))
    val axis = master(Axi4Stream(
      Axi4StreamConfig(
        dataWidth = axiDataWidth / 8, // in bytes
        useLast = true
      )
    ))
  }
  
  io.axis.valid := io.tensor.stream.valid
  io.tensor.stream.ready := io.axis.ready
  
  // Pack tensor lanes into Bits
  val packedBits = io.tensor.stream.payload.asBits
  
  // Pad if AXI bus is wider
  val paddedData = if (axiDataWidth > tensorChunkWidth) {
    packedBits.resize(axiDataWidth bits)
  } else {
    packedBits
  }
  
  io.axis.data := paddedData
  
  // Generate 'last' signal based on tensor shape
  // For a tensor stream, 'last' should pulse when the final chunk of the tensor is transmitted.
  val totalElements = shape.product
  val totalChunks = totalElements / lanes
  
  val chunkCounter = Counter(totalChunks)
  when(io.tensor.stream.valid && io.axis.ready) {
    chunkCounter.increment()
  }
  
  io.axis.last := chunkCounter.willOverflowIfInc
}
