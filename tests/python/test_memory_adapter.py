# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

# Python/Cocotb co-simulation of the Layer-2 MemoryAdapter abstractions
# (Refactoring 1.3): BramAdapter (FPGA BRAM), SramAsicAdapter (OpenRAM ASIC
# macros) and DdrAdapter (off-chip DDR controller bridge).
#
# Python plays the AXI4 master on the accelerator side (AR/R read bursts) and
# the AXI4 slave on the DDR side, while driving the host write port
# (wrEnable/wrAddr/wrData). Two virtual regions share the physical memory:
#   [imgBase, weightBase)      -> physical [0, memoryWords/2)
#   [weightBase, ...)          -> physical [memoryWords/2, memoryWords)
# with defensive clamping: addresses below imgBase clamp to the first physical
# word (no unsigned underflow), addresses past the memory clamp to the last.

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge
from cocotb_test.simulator import run

from utils.tb_utils import run_mill

FILTER = "Generate Verilog for Python co-simulation"

IMG_BASE = 0x1000
WEIGHT_BASE = 0x2000
WORDS = 64
HALF_WORDS = WORDS // 2
BEAT_BYTES = 8


async def reset_adapter(dut):
    dut.io_wrEnable.value = 0
    dut.io_wrAddr.value = 0
    dut.io_wrData.value = 0
    dut.io_wrStrb.value = 0xFF
    dut.io_axi_aw_valid.value = 0
    dut.io_axi_w_valid.value = 0
    dut.io_axi_b_ready.value = 0
    dut.io_axi_ar_valid.value = 0
    dut.io_axi_ar_payload_addr.value = 0
    dut.io_axi_ar_payload_len.value = 0
    dut.io_axi_ar_payload_id.value = 0
    dut.io_axi_r_ready.value = 1
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)


async def host_write(dut, addr, data, strb=0xFF):
    await RisingEdge(dut.clk)
    dut.io_wrEnable.value = 1
    dut.io_wrAddr.value = addr
    dut.io_wrData.value = data
    dut.io_wrStrb.value = strb
    await RisingEdge(dut.clk)
    dut.io_wrEnable.value = 0
    await RisingEdge(dut.clk)


async def axi_write_burst(dut, addr, words, strbs=None, timeout=500):
    """Drive one AXI4 INCR write burst (test = AXI master), one beat per word."""
    # An idle slave must be able to accept AW at any time (read-only
    # implementations tie aw.ready low: this probe fails loudly there).
    aw_ready_seen = False
    for _ in range(100):
        await ReadOnly()
        aw_ready_seen = int(dut.io_axi_aw_ready.value) == 1
        await RisingEdge(dut.clk)
        if aw_ready_seen:
            break
    assert aw_ready_seen, "Adapter never accepts accelerator writes (aw.ready stuck low)"

    dut.io_axi_aw_valid.value = 1
    dut.io_axi_aw_payload_addr.value = addr
    dut.io_axi_aw_payload_len.value = len(words) - 1
    aw_accepted = False
    for _ in range(timeout):
        await ReadOnly()
        if int(dut.io_axi_aw_ready.value) == 1:
            aw_accepted = True
        await RisingEdge(dut.clk)
        if aw_accepted:
            break
    assert aw_accepted, f"AW never accepted for write burst at 0x{addr:X}"
    dut.io_axi_aw_valid.value = 0

    dut.io_axi_b_ready.value = 1
    for i, word in enumerate(words):
        dut.io_axi_w_valid.value = 1
        dut.io_axi_w_payload_data.value = word
        dut.io_axi_w_payload_strb.value = 0xFF if strbs is None else strbs[i]
        dut.io_axi_w_payload_last.value = 1 if i == len(words) - 1 else 0
        w_accepted = False
        for _ in range(timeout):
            await ReadOnly()
            if int(dut.io_axi_w_ready.value) == 1:
                w_accepted = True
            await RisingEdge(dut.clk)
            if w_accepted:
                break
        assert w_accepted, f"W beat {i} never accepted"
    dut.io_axi_w_valid.value = 0

    b_seen = False
    for _ in range(timeout):
        await ReadOnly()
        if int(dut.io_axi_b_valid.value) == 1:
            b_seen = True
        await RisingEdge(dut.clk)
        if b_seen:
            break
    assert b_seen, "B response never received"
    dut.io_axi_b_ready.value = 0
    await RisingEdge(dut.clk)


async def axi_read_burst(dut, addr, beats, id_val=0, stall_cycles=0, timeout=2000):
    """Drive one AR burst (test = AXI master) and collect the R beats.

    Returns (data, last_flags, ids). `stall_cycles` deasserts r_ready for that
    many cycles right after the first captured beat to exercise backpressure.
    """
    dut.io_axi_ar_valid.value = 1
    dut.io_axi_ar_payload_addr.value = addr
    dut.io_axi_ar_payload_len.value = beats - 1
    dut.io_axi_ar_payload_id.value = id_val
    while True:
        await ReadOnly()
        accepted = int(dut.io_axi_ar_ready.value) == 1
        await RisingEdge(dut.clk)
        if accepted:
            break
    dut.io_axi_ar_valid.value = 0

    dut.io_axi_r_ready.value = 1
    data_out = []
    last_flags = []
    ids = []
    stalls_left = stall_cycles

    for _ in range(timeout):
        await ReadOnly()
        valid = int(dut.io_axi_r_valid.value)
        ready = int(dut.io_axi_r_ready.value)
        if valid and ready:
            data_out.append(int(dut.io_axi_r_payload_data.value))
            last_flags.append(int(dut.io_axi_r_payload_last.value))
            ids.append(int(dut.io_axi_r_payload_id.value))

        await RisingEdge(dut.clk)
        if len(data_out) == beats:
            dut.io_axi_r_ready.value = 1
            return data_out, last_flags, ids

        if stalls_left > 0:
            dut.io_axi_r_ready.value = 0
            stalls_left -= 1
        else:
            dut.io_axi_r_ready.value = 1

    raise AssertionError(f"Timed out reading burst at 0x{addr:X}: {len(data_out)}/{beats}")


def check_single_read(data, lasts, ids, expected, id_val):
    assert data == [expected], f"Readback mismatch: 0x{data[0]:016X} != 0x{expected:016X}"
    assert lasts == [1], f"Single-beat burst must assert RLAST, got {lasts}"
    assert ids == [id_val], f"RID echo mismatch: {ids} != [{id_val}]"


async def check_write_idle(dut):
    await ReadOnly()
    assert int(dut.io_axi_aw_ready.value) == 1, "Adapter must accept AW when idle"
    assert int(dut.io_axi_w_ready.value) == 0, "W must wait for an accepted AW"
    assert int(dut.io_axi_b_valid.value) == 0, "No B response before any write"
    await RisingEdge(dut.clk)


async def bram_like_readback(dut):
    await reset_adapter(dut)
    await check_write_idle(dut)

    img_word = 0x1122334455667788
    weight_word = 0xAABBCCDDEEFF0011
    await host_write(dut, IMG_BASE, img_word)
    await host_write(dut, WEIGHT_BASE, weight_word)

    data, lasts, ids = await axi_read_burst(dut, IMG_BASE, beats=1, id_val=1)
    check_single_read(data, lasts, ids, img_word, 1)

    data, lasts, ids = await axi_read_burst(dut, WEIGHT_BASE, beats=1, id_val=2)
    check_single_read(data, lasts, ids, weight_word, 2)


@cocotb.test()
async def cocotb_bram_adapter_readback(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await bram_like_readback(dut)
    print("BramAdapter img/weight region readback bit-exact (both virtual regions)")


@cocotb.test()
async def cocotb_sram_adapter_readback(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await bram_like_readback(dut)
    print("SramAsicAdapter img/weight region readback bit-exact (ASIC macro path)")


@cocotb.test()
async def cocotb_bram_adapter_burst(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_adapter(dut)

    words = [0x0102030405060708 * (i + 1) & ((1 << 64) - 1) for i in range(4)]
    for i, word in enumerate(words):
        await host_write(dut, IMG_BASE + i * BEAT_BYTES, word)

    data, lasts, ids = await axi_read_burst(dut, IMG_BASE, beats=4, id_val=7)
    assert data == words, f"Burst data mismatch: {[hex(w) for w in data]} != {[hex(w) for w in words]}"
    assert lasts == [0, 0, 0, 1], f"RLAST must only be asserted on the final beat: {lasts}"
    assert ids == [7, 7, 7, 7], f"RID must stay stable across the burst: {ids}"
    print("BramAdapter 4-beat AXI4 burst bit-exact (RLAST timing + RID echo)")


@cocotb.test()
async def cocotb_bram_adapter_clamp(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_adapter(dut)

    # First physical word = first image-region word; last = last weight word.
    first_word = 0x0123456789ABCDEF
    last_word = 0xDEADBEEFCAFEBABE
    await host_write(dut, IMG_BASE, first_word)
    await host_write(dut, WEIGHT_BASE + (HALF_WORDS - 1) * BEAT_BYTES, last_word)

    # Addresses below imgBase must never underflow into the last word: they
    # clamp to the first physical word. Addresses past the weight region clamp
    # to the last one.
    data, lasts, ids = await axi_read_burst(dut, 0x0, beats=1, id_val=3)
    check_single_read(data, lasts, ids, first_word, 3)

    data, lasts, ids = await axi_read_burst(
        dut, WEIGHT_BASE + HALF_WORDS * BEAT_BYTES + 0x800, beats=1, id_val=3)
    check_single_read(data, lasts, ids, last_word, 3)

    print("BramAdapter out-of-range reads clamp to the first (addr < imgBase) / last physical word")


@cocotb.test()
async def cocotb_bram_adapter_partial_write(dut):
    """Host byte strobes must preserve the untouched bytes of a word."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_adapter(dut)

    sentinel = 0xAABBCCDDEEFF0011
    await host_write(dut, IMG_BASE, sentinel)
    # Overwrite the 3 low bytes only (0x332211 -> bytes 0..2).
    await host_write(dut, IMG_BASE, 0x0000000000332211, strb=0x07)

    data, lasts, ids = await axi_read_burst(dut, IMG_BASE, beats=1, id_val=1)
    expected = (sentinel & ~0xFFFFFF) | 0x332211
    check_single_read(data, lasts, ids, expected, 1)
    print("BramAdapter partial host write honours byte strobes")


@cocotb.test()
async def cocotb_bram_adapter_backpressure(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_adapter(dut)

    words = [0xAAAABBBBCCCCDDDD - i for i in range(4)]
    for i, word in enumerate(words):
        await host_write(dut, IMG_BASE + i * BEAT_BYTES, word)

    data, lasts, ids = await axi_read_burst(dut, IMG_BASE, beats=4, id_val=5, stall_cycles=6)
    assert data == words, f"Data corrupted under r_ready stalls: {[hex(w) for w in data]}"
    assert lasts == [0, 0, 0, 1], f"RLAST corrupted under stalls: {lasts}"
    assert ids == [5, 5, 5, 5], f"RID corrupted under stalls: {ids}"
    print("BramAdapter burst preserved under r_ready backpressure")


@cocotb.test()
async def cocotb_bram_adapter_write_burst(dut):
    """Accelerator-side AXI4 write burst (AW/W/B) must land in the same memory
    the accelerator reads back (DMAWriter write-back path)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_adapter(dut)

    words = [0x0102030405060708 * (i + 1) & ((1 << 64) - 1) for i in range(4)]
    await axi_write_burst(dut, IMG_BASE, words)

    data, lasts, ids = await axi_read_burst(dut, IMG_BASE, beats=4, id_val=2)
    assert data == words, f"Write-back mismatch: {[hex(w) for w in data]} != {[hex(w) for w in words]}"
    assert lasts == [0, 0, 0, 1], f"RLAST corrupted after write-back: {lasts}"
    assert ids == [2, 2, 2, 2], f"RID corrupted after write-back: {ids}"
    print("BramAdapter accelerator AXI4 write-back burst round-trips bit-exact")


@cocotb.test()
async def cocotb_bram_adapter_write_strobes(dut):
    """AXI w.strb must preserve the untouched bytes of a word."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_adapter(dut)

    sentinel = 0x1122334455667788
    await host_write(dut, IMG_BASE, sentinel)
    await axi_write_burst(dut, IMG_BASE, [0x000000000000BEEF], strbs=[0x03])

    data, _, _ = await axi_read_burst(dut, IMG_BASE, beats=1, id_val=3)
    expected = (sentinel & ~0xFFFF) | 0xBEEF
    assert data == [expected], (
        f"AXI strobe mismatch: 0x{data[0]:016X} != 0x{expected:016X}"
    )
    print("BramAdapter AXI w.strb preserves untouched bytes")


@cocotb.test()
async def cocotb_ddr_adapter_write(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.io_wrEnable.value = 0
    dut.io_wrAddr.value = 0
    dut.io_wrData.value = 0
    dut.io_wrStrb.value = 0xFF
    dut.io_axi_ar_valid.value = 0
    dut.io_axi_r_ready.value = 1
    dut.extIo_ddrMaster_aw_ready.value = 0
    dut.extIo_ddrMaster_w_ready.value = 0
    dut.extIo_ddrMaster_b_valid.value = 0
    dut.extIo_ddrMaster_r_valid.value = 0
    dut.extIo_ddrMaster_r_payload_data.value = 0
    dut.extIo_ddrMaster_r_payload_last.value = 0
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)

    ddr_addr = 0x80000000
    ddr_data = 0xDEADBEEF00112233
    dut.io_wrEnable.value = 1
    dut.io_wrAddr.value = ddr_addr
    dut.io_wrData.value = ddr_data
    await RisingEdge(dut.clk)
    dut.io_wrEnable.value = 0
    await RisingEdge(dut.clk)

    await ReadOnly()
    assert int(dut.extIo_ddrMaster_aw_valid.value) == 1, "DDR AW must be pending after host write"
    assert int(dut.extIo_ddrMaster_aw_payload_addr.value) == ddr_addr, "DDR AW address mismatch"
    assert int(dut.extIo_ddrMaster_aw_payload_len.value) == 0, "DDR AW must be a single-beat burst"
    assert int(dut.extIo_ddrMaster_w_valid.value) == 1, "DDR W must be pending after host write"
    assert int(dut.extIo_ddrMaster_w_payload_data.value) == ddr_data, "DDR W data mismatch"
    assert int(dut.extIo_ddrMaster_w_payload_last.value) == 1, "DDR W last must be asserted"

    await RisingEdge(dut.clk)
    dut.extIo_ddrMaster_aw_ready.value = 1
    dut.extIo_ddrMaster_w_ready.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.extIo_ddrMaster_aw_valid.value) == 0, "DDR AW must retire after acceptance"
    assert int(dut.extIo_ddrMaster_w_valid.value) == 0, "DDR W must retire after acceptance"
    print("DdrAdapter host write translated to a single-beat AXI4 write on the DDR master")


@cocotb.test()
async def cocotb_ddr_adapter_accel_write(dut):
    """The accelerator AXI4 write burst shares the DDR master with the host
    port: host priority, burst forwarding, and B response routing."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.io_wrEnable.value = 0
    dut.io_wrAddr.value = 0
    dut.io_wrData.value = 0
    dut.io_wrStrb.value = 0xFF
    dut.io_axi_aw_valid.value = 0
    dut.io_axi_w_valid.value = 0
    dut.io_axi_b_ready.value = 0
    dut.io_axi_ar_valid.value = 0
    dut.io_axi_r_ready.value = 1
    dut.extIo_ddrMaster_aw_ready.value = 0
    dut.extIo_ddrMaster_w_ready.value = 0
    dut.extIo_ddrMaster_b_valid.value = 0
    dut.extIo_ddrMaster_r_valid.value = 0
    dut.extIo_ddrMaster_r_payload_data.value = 0
    dut.extIo_ddrMaster_r_payload_last.value = 0
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)

    # An idle adapter must accept accelerator AW (read-only implementations
    # tie aw.ready low: probe before driving the AXI payloads).
    acc_ready_seen = False
    for _ in range(50):
        await ReadOnly()
        acc_ready_seen = int(dut.io_axi_aw_ready.value) == 1
        await RisingEdge(dut.clk)
        if acc_ready_seen:
            break
    assert acc_ready_seen, "DdrAdapter never accepts accelerator writes (aw.ready stuck low)"

    # 1. Host write takes the bus and blocks the accelerator AW
    dut.io_wrEnable.value = 1
    dut.io_wrAddr.value = 0x80000000
    dut.io_wrData.value = 0x1122334455667788
    dut.io_wrStrb.value = 0x0F
    await RisingEdge(dut.clk)
    dut.io_wrEnable.value = 0
    await RisingEdge(dut.clk)

    await ReadOnly()
    assert int(dut.extIo_ddrMaster_aw_valid.value) == 1, "Host AW must be pending"
    assert int(dut.extIo_ddrMaster_aw_payload_addr.value) == 0x80000000, "Host AW address mismatch"
    assert int(dut.extIo_ddrMaster_w_payload_strb.value) == 0x0F, "Host w.strb mismatch"
    await RisingEdge(dut.clk)

    dut.io_axi_aw_valid.value = 1
    dut.io_axi_aw_payload_addr.value = 0x9000
    dut.io_axi_aw_payload_len.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.io_axi_aw_ready.value) == 0, "Accelerator AW must wait for the host write"
    await RisingEdge(dut.clk)

    # Complete the host single-beat write, then its auto-acked B.
    dut.extIo_ddrMaster_aw_ready.value = 1
    dut.extIo_ddrMaster_w_ready.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.extIo_ddrMaster_aw_ready.value = 0
    dut.extIo_ddrMaster_w_ready.value = 0
    await RisingEdge(dut.clk)
    dut.extIo_ddrMaster_b_valid.value = 1
    await RisingEdge(dut.clk)
    dut.extIo_ddrMaster_b_valid.value = 0

    # 2. Accelerator 2-beat burst now owns the bus
    burst_addr = 0x9000
    words = [0xAAAABBBBCCCCDDDD, 0x1234567890ABCDEF]
    strbs = [0xFF, 0x0F]
    aw_done = False
    for _ in range(200):
        await ReadOnly()
        ready = int(dut.io_axi_aw_ready.value)
        await RisingEdge(dut.clk)
        if ready:
            aw_done = True
            break
    assert aw_done, "Accelerator AW never accepted after the host write"
    dut.io_axi_aw_valid.value = 0

    for _ in range(50):
        await ReadOnly()
        if int(dut.extIo_ddrMaster_aw_valid.value) == 1:
            break
        await RisingEdge(dut.clk)
    assert int(dut.extIo_ddrMaster_aw_valid.value) == 1, "Burst AW never reached the DDR master"
    assert int(dut.extIo_ddrMaster_aw_payload_addr.value) == burst_addr, "Burst AW address mismatch"
    assert int(dut.extIo_ddrMaster_aw_payload_len.value) == 1, "Burst AW len mismatch"
    await RisingEdge(dut.clk)
    dut.extIo_ddrMaster_aw_ready.value = 1
    await RisingEdge(dut.clk)
    dut.extIo_ddrMaster_aw_ready.value = 0

    dut.extIo_ddrMaster_w_ready.value = 1
    for i, word in enumerate(words):
        last = 1 if i == len(words) - 1 else 0
        dut.io_axi_w_valid.value = 1
        dut.io_axi_w_payload_data.value = word
        dut.io_axi_w_payload_strb.value = strbs[i]
        dut.io_axi_w_payload_last.value = last
        w_done = False
        for _ in range(200):
            await ReadOnly()
            ready = int(dut.io_axi_w_ready.value)
            if ready:
                assert int(dut.extIo_ddrMaster_w_valid.value) == 1, f"ext W valid dropped on beat {i}"
                assert int(dut.extIo_ddrMaster_w_payload_data.value) == word, f"beat {i} data mismatch"
                assert int(dut.extIo_ddrMaster_w_payload_strb.value) == strbs[i], f"beat {i} strb mismatch"
                assert int(dut.extIo_ddrMaster_w_payload_last.value) == last, f"beat {i} last mismatch"
            await RisingEdge(dut.clk)
            if ready:
                w_done = True
                break
        assert w_done, f"Accelerator W beat {i} never handshook"
    dut.io_axi_w_valid.value = 0

    # B is now owned by the accelerator burst and must be forwarded.
    dut.io_axi_b_ready.value = 1
    for _ in range(50):
        await ReadOnly()
        if int(dut.extIo_ddrMaster_b_ready.value) == 1:
            break
        await RisingEdge(dut.clk)
    assert int(dut.extIo_ddrMaster_b_ready.value) == 1, "Burst B must be accepted for the accelerator"
    await RisingEdge(dut.clk)

    dut.extIo_ddrMaster_b_valid.value = 1
    b_seen = False
    for _ in range(50):
        await ReadOnly()
        if int(dut.io_axi_b_valid.value) == 1:
            b_seen = True
        await RisingEdge(dut.clk)
        if b_seen:
            break
    assert b_seen, "B response was not forwarded to the accelerator"
    dut.extIo_ddrMaster_b_valid.value = 0
    dut.io_axi_b_ready.value = 0
    await RisingEdge(dut.clk)
    print("DdrAdapter host priority + accelerator burst write-back with B routing validated")


@cocotb.test()
async def cocotb_ddr_adapter_read(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.io_wrEnable.value = 0
    dut.io_wrAddr.value = 0
    dut.io_wrData.value = 0
    dut.io_wrStrb.value = 0xFF
    dut.io_axi_ar_valid.value = 0
    dut.io_axi_ar_payload_addr.value = 0
    dut.io_axi_ar_payload_len.value = 0
    dut.io_axi_ar_payload_id.value = 0
    dut.io_axi_r_ready.value = 1
    dut.extIo_ddrMaster_aw_ready.value = 0
    dut.extIo_ddrMaster_w_ready.value = 0
    dut.extIo_ddrMaster_b_valid.value = 0
    dut.extIo_ddrMaster_ar_ready.value = 0
    dut.extIo_ddrMaster_r_valid.value = 0
    dut.extIo_ddrMaster_r_payload_data.value = 0
    dut.extIo_ddrMaster_r_payload_last.value = 0
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)

    ar_addr = 0x4000
    dut.io_axi_ar_valid.value = 1
    dut.io_axi_ar_payload_addr.value = ar_addr
    dut.io_axi_ar_payload_len.value = 0

    for _ in range(100):
        await ReadOnly()
        if int(dut.extIo_ddrMaster_ar_valid.value) == 1:
            break
        await RisingEdge(dut.clk)
    else:
        raise AssertionError("Accelerator AR was never forwarded to the DDR master")

    assert int(dut.extIo_ddrMaster_ar_payload_addr.value) == ar_addr, "Forwarded AR address mismatch"
    assert int(dut.extIo_ddrMaster_ar_payload_len.value) == 0, "Forwarded AR must be single-beat"

    # Accept AR on both sides simultaneously
    await RisingEdge(dut.clk)
    dut.extIo_ddrMaster_ar_ready.value = 1
    await RisingEdge(dut.clk)
    dut.io_axi_ar_valid.value = 0
    dut.extIo_ddrMaster_ar_ready.value = 0

    # DDR controller answers with one beat
    read_data = 0xCAFEBABE12345678
    dut.extIo_ddrMaster_r_valid.value = 1
    dut.extIo_ddrMaster_r_payload_data.value = read_data
    dut.extIo_ddrMaster_r_payload_last.value = 1

    for _ in range(100):
        await ReadOnly()
        if int(dut.io_axi_r_valid.value) == 1:
            break
        await RisingEdge(dut.clk)
    else:
        raise AssertionError("DDR response was never forwarded to the accelerator master")

    assert int(dut.io_axi_r_payload_data.value) == read_data, "Forwarded R data mismatch"
    assert int(dut.io_axi_r_payload_last.value) == 1, "Forwarded RLAST mismatch"

    await RisingEdge(dut.clk)
    dut.extIo_ddrMaster_r_valid.value = 0
    print("DdrAdapter AXI4 read pass-through (AR forward / R response) validated")


def _run_sim(testcase, toplevel, sim_build, request=None):
    v_file = run_mill("spinalML.memory.MemoryAdapterTest", FILTER, toplevel)

    from utils.test_layers_utils import safe_run_sim as run

    run(
        simulator="verilator",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_memory_adapter",
        testcase=testcase,
        sim_build=sim_build,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
    )


def test_bram_adapter_readback(request):
    _run_sim("cocotb_bram_adapter_readback", "BramAdapterTestComp", "sim_build/py_bram_readback", request)


def test_sram_adapter_readback(request):
    _run_sim("cocotb_sram_adapter_readback", "SramAsicAdapterTestComp", "sim_build/py_sram_readback", request)


def test_bram_adapter_burst(request):
    _run_sim("cocotb_bram_adapter_burst", "BramAdapterTestComp", "sim_build/py_bram_burst", request)


def test_bram_adapter_clamp(request):
    _run_sim("cocotb_bram_adapter_clamp", "BramAdapterTestComp", "sim_build/py_bram_clamp", request)


def test_bram_adapter_backpressure(request):
    _run_sim("cocotb_bram_adapter_backpressure", "BramAdapterTestComp", "sim_build/py_bram_backpressure", request)


def test_bram_adapter_partial_write(request):
    _run_sim("cocotb_bram_adapter_partial_write", "BramAdapterTestComp", "sim_build/py_bram_partial_write", request)


def test_bram_adapter_write_burst(request):
    _run_sim("cocotb_bram_adapter_write_burst", "BramAdapterTestComp", "sim_build/py_bram_write_burst", request)


def test_sram_adapter_write_burst(request):
    _run_sim("cocotb_bram_adapter_write_burst", "SramAsicAdapterTestComp", "sim_build/py_sram_write_burst", request)


def test_bram_adapter_write_strobes(request):
    _run_sim("cocotb_bram_adapter_write_strobes", "BramAdapterTestComp", "sim_build/py_bram_write_strobes", request)


def test_ddr_adapter_write(request):
    _run_sim("cocotb_ddr_adapter_write", "DdrAdapterTestComp", "sim_build/py_ddr_write", request)


def test_ddr_adapter_accel_write(request):
    _run_sim("cocotb_ddr_adapter_accel_write", "DdrAdapterTestComp", "sim_build/py_ddr_accel_write", request)


def test_ddr_adapter_read(request):
    _run_sim("cocotb_ddr_adapter_read", "DdrAdapterTestComp", "sim_build/py_ddr_read", request)
