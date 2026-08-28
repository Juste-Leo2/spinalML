package spinalML.examples

import spinal.core.SInt
import spinal.lib.bus.amba4.axi.Axi4Config
import spinalML.dtypes.I8
import spinalML.nn.{Accelerator, Linear => LinearSpec}

/** Env-driven W4A8 model construction shared by the SoC/chain suites:
 *  - MNIST_WLANES overrides the Linear K-chunk width (default: spec, 4 lanes);
 *  - MNIST_TEMPORAL sets the windowed accumulator (default 0 = legacy).
 *  `make` must be called INSIDE the (by-name) compile generator (there is no
 *  elaboration context outside it). */
object W4A8Knob {
  def spec(): Seq[spinalML.nn.LayerSpec] = sys.env.get("MNIST_WLANES") match {
    case Some(s) => Mnistw4a8.defaultModelSpec.map {
      case l: LinearSpec => l.copy(weightLanes = s.toInt)
      case o => o
    }
    case None => Mnistw4a8.defaultModelSpec
  }

  def lanes(): Int = spec().collectFirst { case l: LinearSpec => l.effLanes }.getOrElse(288)

  def temporal(): Int = sys.env.get("MNIST_TEMPORAL").map(_.toInt).getOrElse(0)

  def make(axiConfig: Axi4Config): Accelerator[SInt] =
    new Accelerator(dataType = I8(), inputShape = Seq(28, 28, 1),
      modelSpec = spec(), axiConfig = axiConfig, temporal = temporal())
}
