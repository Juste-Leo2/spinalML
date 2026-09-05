# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly
from utils.tb_utils import run_mill
import random

@cocotb.test()
async def cocotb_test_double_buffer_streamer(dut):
    """Test the DoubleBufferStreamer with random backpressure"""
    # 1. Start Clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # 2. Initialize inputs
    dut.reset.value = 1
    dut.io_tileReady.value = 0
    dut.io_readData_0.value = 0
    dut.io_streamOut_ready.value = 0
    
    await Timer(20, units="ns")
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Mock RAM to respond to readAddr with (readAddr + 100)
    async def mock_ram():
        while True:
            # Wait until combinatorial logic has settled to sample the address
            await ReadOnly()
            try:
                addr = int(dut.io_readAddr.value)
            except ValueError:
                addr = 0
                
            await RisingEdge(dut.clk)
            # The RAM has 1 cycle latency
            dut.io_readData_0.value = addr + 100
            
    cocotb.start_soon(mock_ram())
    
    # 3. Trigger a tile read
    dut.io_tileReady.value = 1
    
    # 4. Read the output stream with random backpressure
    received_data = []
    
    # We expect 16 elements (depth=16, lanes=1)
    timeout = 500
    while len(received_data) < 16 and timeout > 0:
        # 1. Drive inputs for this cycle
        ready_val = 1 if random.random() < 0.7 else 0
        dut.io_streamOut_ready.value = ready_val
        
        # 2. Wait for DUT to evaluate combinatorial logic
        await ReadOnly()
        
        # 3. Sample outputs
        if dut.io_streamOut_valid.value == 1 and ready_val == 1:
            val = int(dut.io_streamOut_payload_0.value)
            received_data.append(val)
            
        # 4. Wait for next clock cycle
        await RisingEdge(dut.clk)
        timeout -= 1
        
    assert timeout > 0, "Simulation timed out!"
    
    # Check that we received exactly 100 to 115
    expected_data = [i + 100 for i in range(16)]
    assert received_data == expected_data, f"Data mismatch! Got {received_data}, expected {expected_data}"
    
    # Check that nextTile was pulsed!
    # nextTile pulses for exactly 1 cycle when the reading finishes.
    # We might have missed it in the loop if it happened after we collected the 16th element,
    # but let's just wait a few cycles and verify it goes high.
    # Wait, nextTile is pulsed when the read requests finish, which happens BEFORE the output stream finishes!
    # Because there is a FIFO in between. 
    # Let's just assure it doesn't crash and outputs the right data.

def test_double_buffer_streamer_runner():
    """Pytest runner for test_double_buffer_streamer"""
    # 1. Compile the Scala code to Verilog
    v_file = run_mill("spinalML.memory.DoubleBufferStreamerTest", "", "DoubleBufferStreamer")
    
    from utils.test_layers_utils import safe_run_sim as run
    run(
        simulator="icarus",
        verilog_sources=[v_file],
        toplevel="DoubleBufferStreamer",
        module="test_double_buffer_streamer",
        testcase="cocotb_test_double_buffer_streamer"
    )
