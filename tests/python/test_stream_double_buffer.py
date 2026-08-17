import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from utils.tb_utils import run_mill

@cocotb.test()
async def run_double_buffer_sim(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.reset.value = 1
    dut.io_streamIn_valid.value = 0
    dut.io_nextTile.value = 0
    dut.io_readAddr.value = 0
    
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # The depth is 8, lanes is 2. So 4 addresses per bank.
    
    # 1. Fill Ping Bank
    for i in range(4):
        dut.io_streamIn_valid.value = 1
        dut.io_streamIn_payload_0.value = i * 2
        dut.io_streamIn_payload_1.value = i * 2 + 1
        await RisingEdge(dut.clk)
        
    dut.io_streamIn_valid.value = 0
    
    # Wait for internal registers to flip (pingFull)
    await RisingEdge(dut.clk)
    
    assert dut.io_tileReady.value == 1, "Ping bank should be ready!"
    
    # 2. Read from Ping Bank
    for i in range(4):
        dut.io_readAddr.value = i
        await RisingEdge(dut.clk) # address registers in Mem
        await RisingEdge(dut.clk) # memory readSync latency (1 cycle)
        
        v0 = dut.io_readData_0.value.signed_integer
        v1 = dut.io_readData_1.value.signed_integer
        
        assert v0 == i * 2, f"Expected {i * 2}, got {v0}"
        assert v1 == i * 2 + 1, f"Expected {i * 2 + 1}, got {v1}"
        
    # 3. Fill Pong Bank while computing Ping
    for i in range(4):
        dut.io_streamIn_valid.value = 1
        dut.io_streamIn_payload_0.value = 100 + i * 2
        dut.io_streamIn_payload_1.value = 100 + i * 2 + 1
        await RisingEdge(dut.clk)
        
    dut.io_streamIn_valid.value = 0
    
    # 4. Finish computing Ping, switch to Pong
    dut.io_nextTile.value = 1
    await RisingEdge(dut.clk)
    dut.io_nextTile.value = 0
    
    # Wait for computeBank to update
    await RisingEdge(dut.clk)
    
    # 5. Read from Pong Bank
    dut.io_readAddr.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    v0 = dut.io_readData_0.value.signed_integer
    assert v0 == 100, f"Expected 100, got {v0}"
    
    print("StreamDoubleBuffer Ping-Pong logic validated successfully in Python/Cocotb!")

def test_stream_double_buffer():
    v_file = run_mill("spinalML.memory.StreamDoubleBufferTest", "", "StreamDoubleBufferTestComp")
    
    from cocotb_test.simulator import run
    run(
        simulator="verilator",
        verilog_sources=[v_file],
        toplevel="StreamDoubleBufferTestComp",
        module="test_stream_double_buffer",
        timescale="1ns/1ps",
        testcase="run_double_buffer_sim"
    )
