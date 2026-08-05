package spinalML

import spinal.core._

// A simple hardware module that adds two 8-bit unsigned integers.
// This serves as our "Hello World" to verify the compilation and simulation toolchain.
case class HelloWorld() extends Component {
  val io = new Bundle {
    val a = in UInt(8 bits)
    val b = in UInt(8 bits)
    val result = out UInt(8 bits)
  }

  io.result := io.a + io.b
}

object HelloWorldVerilog {
  def main(args: Array[String]) {
    SpinalVerilog(HelloWorld())
  }
}
