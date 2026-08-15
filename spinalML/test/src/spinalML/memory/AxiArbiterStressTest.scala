package spinalML.memory

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import org.scalatest.funsuite.AnyFunSuite
import scala.util.Random

class AxiArbiterStressTest extends AnyFunSuite {
  case class AxiArbiterTestComp(numInputs: Int, config: Axi4Config) extends Component {
    val inputConfig = config.copy(idWidth = config.idWidth - log2Up(Math.max(2, numInputs)))
    val io = new Bundle {
      val inputs = Vec(slave(Axi4ReadOnly(inputConfig)), numInputs)
      val output = master(Axi4ReadOnly(config))
    }
    val arbiter = Axi4ReadOnlyArbiter(config, numInputs)
    (arbiter.io.inputs, io.inputs).zipped.foreach(_ <> _)
    io.output <> arbiter.io.output
  }

  test("Axi4ReadOnlyArbiter Stress Test under severe backpressure") {
    val numInputs = 4
    val config = Axi4Config(addressWidth = 32, dataWidth = 64, idWidth = 4)
    
    SimConfig.withWave.compile(AxiArbiterTestComp(numInputs, config)).doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      
      // Initialize inputs
      for (i <- 0 until numInputs) {
        dut.io.inputs(i).ar.valid #= false
        dut.io.inputs(i).r.ready #= false
      }
      dut.io.output.ar.ready #= false
      dut.io.output.r.valid #= false
      
      dut.clockDomain.waitSampling(5)
      
      var pendingRequests = 0
      var completedRequests = 0
      val totalRequestsPerMaster = 50
      
      // Fork master stimulus (AR channels)
      for (i <- 0 until numInputs) {
        fork {
          for (req <- 0 until totalRequestsPerMaster) {
            dut.io.inputs(i).ar.valid #= true
            dut.io.inputs(i).ar.addr #= req * 0x100 + (i * 0x10000)
            dut.io.inputs(i).ar.id #= i // ID used to route back
            dut.io.inputs(i).ar.len #= 3 // 4 beats
            
            dut.clockDomain.waitSamplingWhere(dut.io.inputs(i).ar.ready.toBoolean)
            dut.io.inputs(i).ar.valid #= false
            
            // Random delay between requests to test starvation/fairness
            dut.clockDomain.waitSampling(Random.nextInt(10))
          }
        }
      }
      
      // Fork slave memory (AR channel receiver & R channel sender)
      fork {
        while (completedRequests < numInputs * totalRequestsPerMaster) {
          // Slave AR channel with backpressure
          dut.io.output.ar.ready #= (Random.nextInt(100) < 30) // 70% backpressure
          
          if (dut.io.output.ar.valid.toBoolean && dut.io.output.ar.ready.toBoolean) {
            pendingRequests += 1
            val id = dut.io.output.ar.id.toLong
            val len = dut.io.output.ar.len.toLong
            
            // Send R response in a new thread
            fork {
              dut.clockDomain.waitSampling(Random.nextInt(15) + 5) // Memory latency
              for (beat <- 0 to len.toInt) {
                dut.io.output.r.valid #= true
                dut.io.output.r.id #= id
                dut.io.output.r.data #= beat + (id * 1000)
                dut.io.output.r.last #= (beat == len.toInt)
                
                dut.clockDomain.waitSamplingWhere(dut.io.output.r.ready.toBoolean)
                dut.io.output.r.valid #= false
              }
              completedRequests += 1
              pendingRequests -= 1
            }
          }
          dut.clockDomain.waitSampling()
        }
      }
      
      // Fork Master R channels (receivers with backpressure)
      for (i <- 0 until numInputs) {
        fork {
          while (completedRequests < numInputs * totalRequestsPerMaster) {
            dut.io.inputs(i).r.ready #= (Random.nextInt(100) < 40) // 60% backpressure
            dut.clockDomain.waitSampling()
          }
        }
      }
      
      // Wait for all requests to finish
      dut.clockDomain.waitSamplingWhere(completedRequests == numInputs * totalRequestsPerMaster)
      dut.clockDomain.waitSampling(10)
      
      assert(completedRequests == numInputs * totalRequestsPerMaster, "Not all requests completed")
    }
  }
}
