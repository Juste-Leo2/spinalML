import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ReadOnly
import struct
from utils.tb_utils import run_mill

async def mock_axi_ram(dut, memory_dict):
    try:
        dut.io_axiMaster_ar_ready.value = 0
        dut.io_axiMaster_r_valid.value = 0
        dut.io_axiMaster_r_payload_last.value = 0
        dut.io_axiMaster_r_payload_data.value = 0
        
        while True:
            await ReadOnly()
            ar_valid = str(dut.io_axiMaster_ar_valid.value) == '1'
            ar_ready = str(dut.io_axiMaster_ar_ready.value) == '1'
            await RisingEdge(dut.clk)
            
            if ar_valid and ar_ready:
                try:
                    addr = int(dut.io_axiMaster_ar_payload_addr.value)
                    length = int(dut.io_axiMaster_ar_payload_len.value) + 1
                except ValueError:
                    continue
                dut.io_axiMaster_ar_ready.value = 0
                
                for i in range(length):
                    dut.io_axiMaster_r_valid.value = 1
                    
                    curr_addr = addr + i * 8
                    data_bytes = memory_dict.get(curr_addr, b'\x00'*8)
                    val = int.from_bytes(data_bytes, byteorder='little')
                    
                    dut.io_axiMaster_r_payload_data.value = val
                    dut.io_axiMaster_r_payload_last.value = 1 if i == length - 1 else 0
                    
                    while True:
                        await ReadOnly()
                        r_ready = str(dut.io_axiMaster_r_ready.value) == '1'
                        await RisingEdge(dut.clk)
                        if r_ready:
                            break
                            
                dut.io_axiMaster_r_valid.value = 0
                dut.io_axiMaster_r_payload_last.value = 0
            else:
                dut.io_axiMaster_ar_ready.value = 1
    except Exception as e:
        print(f"MOCK AXI RAM EXCEPTION: {e}")

@cocotb.test()
async def run_dma_reader2d_sim(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    memory = {}
    base_addr = 0x1000
    stride_bytes = 8
    
    image_values = list(range(1, 17))
    for row in range(4):
        data_bytes = bytearray()
        for col in range(4):
            val = image_values[row * 4 + col]
            data_bytes += struct.pack("<h", val)
        memory[base_addr + row * stride_bytes] = data_bytes
        
    cocotb.start_soon(mock_axi_ram(dut, memory))
    
    dut.reset.value = 1
    dut.io_cmd_valid.value = 0
    dut.io_cmd_payload_baseAddress.value = 0
    dut.io_cmd_payload_stride.value = 0
    dut.io_cmd_payload_patchWidth.value = 0
    dut.io_cmd_payload_patchHeight.value = 0
    dut.io_outStream_stream_ready.value = 0
    
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_cmd_valid.value = 1
    dut.io_cmd_payload_baseAddress.value = base_addr
    dut.io_cmd_payload_stride.value = stride_bytes
    dut.io_cmd_payload_patchWidth.value = 0
    dut.io_cmd_payload_patchHeight.value = 4
    dut.io_outStream_stream_ready.value = 1
    
    received_values = []
    cmd_accepted = False
    timeout = 1000
    
    while len(received_values) < 16 and timeout > 0:
        await RisingEdge(dut.clk)
        
        if not cmd_accepted and dut.io_cmd_ready.value == 1:
            cmd_accepted = True
            dut.io_cmd_valid.value = 0
            
        if dut.io_outStream_stream_valid.value == 1 and dut.io_outStream_stream_ready.value == 1:
            v0 = dut.io_outStream_stream_payload_0.value.signed_integer
            received_values.append(v0)
            
        timeout -= 1
        
    assert timeout > 0, "Simulation timed out waiting for 2D DMA data"
    assert received_values == image_values, f"Expected {image_values}, got {received_values}"
    print(f"DMA 2D Reader successfully fetched 2D tile: {received_values}")

def test_dma_reader2d():
    v_file = run_mill("spinalML.memory.DMAReader2DTest", "", "DMAReader2DTestComp")
    
    from utils.test_layers_utils import safe_run_sim as run
    run(
        simulator="icarus",
        verilog_sources=[v_file],
        toplevel="DMAReader2DTestComp",
        module="test_dma_reader2d",
        timescale="1ns/1ps",
        testcase="run_dma_reader2d_sim",
        sim_build="sim_build/py_dma_reader2d"
    )
