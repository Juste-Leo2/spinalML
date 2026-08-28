package spinalML.utils

/**
 * Structured simulation logging shared by the source tree and the test
 * benches: "[LEVEL] [TAG] message" lines on stdout with a 4-level threshold.
 *
 * Levels (all constant formats):
 *   [ERROR] [TAG] ...  always emitted
 *   [WARN]  [TAG] ...  always emitted (normal-but-notable situations, e.g.
 *                       knob-driven exclusions or large lane footprints)
 *   [INFO]  [TAG] ...  always emitted (verdicts, model summaries, probe
 *                       products — the evidence journal)
 *   [DEBUG] [TAG] ...  emitted when SML_DEBUG=1 (or any non-zero value)
 *   [TRACE] [TAG] ...  emitted when SML_DEBUG=2 (per-beat / per-cycle detail)
 *
 * Tags are chosen by the caller and capitalized automatically; the standard
 * set is [DAG] [TAP] [SIM] [MODEL] [WIDE] [MNIST] [FORMAL] plus probe tags
 * such as [FORK] [K1] [IM2COL].
 *
 * A single tag can be focused with SML_DEBUG_TAG=TAP: when set, DEBUG/TRACE
 * lines with any other tag are suppressed (information levels always pass).
 */
object SimLog {
  sealed trait Level { def tag: String; def rank: Int }
  case object ERROR extends Level { val tag = "ERROR"; val rank = 0 }
  case object WARN extends Level { val tag = "WARN"; val rank = 1 }
  case object INFO extends Level { val tag = "INFO"; val rank = 2 }
  case object DEBUG extends Level { val tag = "DEBUG"; val rank = 3 }
  case object TRACE extends Level { val tag = "TRACE"; val rank = 4 }

  /** 0 = up to INFO, 1 = +DEBUG, 2 = +TRACE (SML_DEBUG env). */
  private val debugLevel: Int = sys.env.get("SML_DEBUG") match {
    case Some(s) if s.trim.nonEmpty =>
      try math.max(0, s.trim.toInt) catch { case _: NumberFormatException => 3 }
    case _ => 0
  }

  private val tagFilter: Option[String] = sys.env.get("SML_DEBUG_TAG") match {
    case Some(s) if s.trim.nonEmpty => Some(s.trim.toUpperCase)
    case _ => None
  }

  def isDebug: Boolean = debugLevel >= 1
  def isTrace: Boolean = debugLevel >= 2
  def threshold: Int = debugLevel

  def error(tag: String)(msg: => String): Unit = emit(ERROR, tag, msg)
  def warn(tag: String)(msg: => String): Unit = emit(WARN, tag, msg)
  def info(tag: String)(msg: => String): Unit = emit(INFO, tag, msg)
  def debug(tag: String)(msg: => String): Unit = emit(DEBUG, tag, msg)
  def trace(tag: String)(msg: => String): Unit = emit(TRACE, tag, msg)

  private def emit(level: Level, tag: String, msg: => String): Unit = {
    if (level.rank <= INFO.rank + debugLevel &&
      (tagFilter.isEmpty || level.rank <= INFO.rank || tagFilter.get == tag.toUpperCase)) {
      System.out.println(s"[${level.tag}] [${tag.toUpperCase}] $msg")
    }
  }

  /** Bench wrapper: headers, wall-clock, PASS/FAIL verdict (rethrows on failure). */
  def bench(name: String, tag: String = "SIM")(body: => Unit): Unit = {
    val t0 = System.nanoTime()
    info(tag)(s"bench=$name start")
    try {
      body
      info(tag)(f"bench=$name PASS in ${(System.nanoTime() - t0) / 1e6}%.0f ms")
    } catch {
      case e: Throwable =>
        error(tag)(f"bench=$name FAILED after ${(System.nanoTime() - t0) / 1e6}%.0f ms: ${e.getMessage}")
        throw e
    }
  }

  /** CRC-32 fingerprint of an int sequence (LUt per raw 32-bit word pattern). */
  def crc32(values: Iterable[Int]): Long = {
    val crc = new java.util.zip.CRC32
    values.foreach { v =>
      crc.update(v & 0xFF)
      crc.update((v >> 8) & 0xFF)
      crc.update((v >> 16) & 0xFF)
      crc.update((v >> 24) & 0xFF)
    }
    crc.getValue
  }
}
