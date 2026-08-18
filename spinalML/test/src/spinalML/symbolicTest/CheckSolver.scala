package spinalML.symbolicTest

object CheckSolver {
  def main(args: Array[String]): Unit = {
    println(spinal.core.formal.SmtBmcSolver.getClass.getMethods.map(_.getName).filterNot(_.contains("$")).mkString(", "))
  }
}
