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
                except ValueError as ve:
                    print(f"mock_axi_ram ValueError: {ve}")
                    continue
                
                print(f"mock_axi_ram ACCEPTED AR: addr={addr}, len={length}")
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
                            print(f"mock_axi_ram R TRANSFER: data={val}, last={1 if i == length - 1 else 0}")
                            break
                            
                dut.io_axiMaster_r_valid.value = 0
                dut.io_axiMaster_r_payload_last.value = 0
                print("mock_axi_ram BURST COMPLETE")
            else:
                dut.io_axiMaster_ar_ready.value = 1
    except Exception as e:
        print(f"MOCK AXI RAM FATAL EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

@cocotb.test()
async def run_dma_reader_sim(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    memory = {}
    base_addr = 0x1000
    
    ml_values = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8]
    data_bytes = bytearray()
    for v in ml_values:
        data_bytes += struct.pack("<h", v)
        
    for i in range(0, len(data_bytes), 8):
        memory[base_addr + i] = data_bytes[i:i+8]
        
    cocotb.start_soon(mock_axi_ram(dut, memory))
    
    dut.reset.value = 1
    dut.io_cmd_valid.value = 0
    dut.io_cmd_payload_address.value = 0
    dut.io_cmd_payload_length.value = 0
    dut.io_outStream_stream_ready.value = 0
    
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut.io_cmd_valid.value = 1
    dut.io_cmd_payload_address.value = base_addr
    dut.io_cmd_payload_length.value = 3
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
            v1 = dut.io_outStream_stream_payload_1.value.signed_integer
            received_values.extend([v0, v1])
            
        timeout -= 1
        
    assert timeout > 0, f"Simulation timed out waiting for DMA data, received {len(received_values)}: {received_values}"
    assert received_values == ml_values, f"Expected {ml_values}, got {received_values}"
    print(f"DMA Reader successfully fetched and repacked {len(received_values)} values: {received_values}")

def test_dma_reader():
    v_file = run_mill("spinalML.memory.DMAReaderTest", "", "DMAReaderTestComp")
    
    from utils.test_layers_utils import safe_run_sim as run
    run(
        simulator="icarus",
        verilog_sources=[v_file],
        toplevel="DMAReaderTestComp",
        module="test_dma_reader",
        timescale="1ns/1ps",
        testcase="run_dma_reader_sim",
        sim_build="sim_build/py_dma_reader"
    )
