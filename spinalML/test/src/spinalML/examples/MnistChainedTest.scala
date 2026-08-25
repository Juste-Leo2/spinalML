package spinalML.examples

import org.scalatest.funsuite.AnyFunSuite
import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.Axi4Config
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig, SparseMemory}

/**
 * Inter-start state re-arming regression (roadmap: Multi-Tile Continuous
 * Inference / inter-start re-arming).
 *
 * Runs SEVERAL full inferences back-to-back inside a SINGLE simulation
 * session (one doSim): image k is written, start pulsed, logits collected
 * and compared bit-exactly against the software replica — then image k+1
 * follows without any reset. Each step must reproduce the fresh-session
 * result exactly.
 *
 * Status since Aug 2026 re-arm work: GREEN — see docs/bugs/2026-08-rearm-session.md
 * (buffer reArm pulses + weight/bias exact-element trim + flushable gearboxes).
 */
class MnistChainedTest extends AnyFunSuite {
  val axiConfig = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
  val imgBase = 0x10000
  val weightBase = 0x20000

  /** Number of chained inferences per session (override via env). */
  val chainLength = sys.env.get("MNIST_CHAIN_N").map(_.toInt).getOrElse(3)

  private def writeWords(mem: SparseMemory, base: Int, words: Seq[BigInt]): Unit =
    for ((w, i) <- words.zipWithIndex) mem.writeBigInt(base + i * 8, w, 8)

  test("Mnistw4a8: chained inferences in one session match the replica") {
    val bench = new Mnistw4a8Test
    val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig)
      .withWave
      .compile(Mnistw4a8(axiConfig))

    // Same digit repeated first (deterministic staleness fingerprint)
    val cases =
      if (sys.env.contains("MNIST_CHAIN_SAME")) Seq.fill(chainLength)(0)
      else Seq.tabulate(chainLength)(k => k % MnistData.images.length)

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memorySim.start()
      writeWords(memorySim.memory, weightBase, bench.weightWords())

      def writeAxiLite(addr: BigInt, data: BigInt) = {
        dut.io.ctrlBus.aw.valid #= true
        dut.io.ctrlBus.aw.payload.addr #= addr
        dut.io.ctrlBus.w.valid #= true
        dut.io.ctrlBus.w.payload.data #= data
        dut.io.ctrlBus.w.payload.strb #= 0xF
        dut.io.ctrlBus.b.ready #= true

        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.aw.ready.toBoolean && dut.io.ctrlBus.w.ready.toBoolean)
        dut.io.ctrlBus.aw.valid #= false
        dut.io.ctrlBus.w.valid #= false
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.b.valid.toBoolean)
        dut.io.ctrlBus.b.ready #= false
        dut.clockDomain.waitSampling()
      }

      for ((idx, step) <- cases.zipWithIndex) {
        val logits = bench.runInference(dut, memorySim.memory,
          MnistData.images(idx), writeAxiLite)

        val expected = Mnistw4a8Replica.logits(MnistData.images(idx))
        val dev = logits.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
        println(f"[step $step] image#$idx predicted ${logits.indexOf(logits.max)}" +
          f" (label ${MnistData.labels(idx)})  max|hw-sw|=$dev%.3f")
        assert(dev == 0.0,
          s"[step $step] image#$idx corrupted after chaining: hw ${logits.map(_.toFloat)} vs sw ${expected.map(f => f.toFloat)}")

      }
      println(s"Mnistw4a8 chained: $chainLength consecutive inferences OK in one session")
    }
  }

  test("Mnist BF16: chained inferences in one session match the replica") {
    assume(!sys.env.contains("W4A8_ONLY"), "W4A8_ONLY set - skipping BF16 chained run")
    val bench = new MnistTest
    val spinalConfig = SpinalConfig(bitVectorWidthMax = 16384)
    val compiled = SimConfig.withVerilator.withConfig(spinalConfig).compile(Mnist(axiConfig))

    val cases = Seq.tabulate(chainLength)(k => k % MnistData.images.length)

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(10)
      val memorySim = AxiMemorySim(
        axi = dut.io.axiMaster,
        clockDomain = dut.clockDomain,
        config = AxiMemorySimConfig(maxOutstandingReads = 8))
      memorySim.start()
      writeWords(memorySim.memory, weightBase, bench.weightWords())

      def writeAxiLite(addr: BigInt, data: BigInt) = {
        dut.io.ctrlBus.aw.valid #= true
        dut.io.ctrlBus.aw.payload.addr #= addr
        dut.io.ctrlBus.w.valid #= true
        dut.io.ctrlBus.w.payload.data #= data
        dut.io.ctrlBus.w.payload.strb #= 0xF
        dut.io.ctrlBus.b.ready #= true

        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.aw.ready.toBoolean && dut.io.ctrlBus.w.ready.toBoolean)
        dut.io.ctrlBus.aw.valid #= false
        dut.io.ctrlBus.w.valid #= false
        dut.clockDomain.waitSamplingWhere(dut.io.ctrlBus.b.valid.toBoolean)
        dut.io.ctrlBus.b.ready #= false
        dut.clockDomain.waitSampling()
      }

      for ((idx, step) <- cases.zipWithIndex) {
        val logits = bench.runInference(dut, memorySim.memory,
          MnistData.images(idx), writeAxiLite)

        val expected = MnistReplica.logits(MnistData.images(idx))
        val dev = logits.zip(expected).map { case (h, s) => math.abs(h.toDouble - s) }.max
        println(f"[step $step] image#$idx predicted ${logits.indexOf(logits.max)}" +
          f" (label ${MnistData.labels(idx)})  max|hw-sw|=$dev%.3f")
        assert(dev == 0.0,
          s"[step $step] image#$idx corrupted after chaining: hw ${logits.map(_.toFloat)} vs sw ${expected.map(f => f.toFloat)}")
      }
      println(s"Mnist BF16 chained: $chainLength consecutive inferences OK in one session")
    }
  }
}
