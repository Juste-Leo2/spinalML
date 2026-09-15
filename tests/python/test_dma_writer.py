# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

# Python/Cocotb co-simulation of the AXI4-Master DMAWriter (Refactoring 1.5).
#
# The generated `DMAWriterTestComp` (DMAWriterTestWrapper, maxBurstBeats = 4)
# drains a 16-element SInt16 Tensor stream (inLanes = 2) and writes it to a
# 64-bit AXI4 memory through AW/W/B bursts:
#   - 16 elements x 16 bits = 8 stream beats = 4 AXI beats (one burst max len)
#   - word w packs elements 4w..4w+3 little-endian (element b at bits 16b)
# A minimal AXI write slave captures bursts, stores the memory image and
# answers every burst with a B response, optionally under random backpressure.

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge
from cocotb_test.simulator import run

from utils.tb_utils import run_mill, seed_random

IN_LANES = 2
N_ELEMS = 16
ELEMS_PER_WORD = 4
AXI_BYTES = 8
MAX_BURST_BEATS = 4


class AxiWriteSlave:
    """Minimal AXI4 write slave: captures AW/W, stores words, answers B."""

    def __init__(self, dut, aw_stall=0.0, w_stall=0.0, b_stall=0.0, seed=1234):
        self.dut = dut
        self.aw_stall = aw_stall
        self.w_stall = w_stall
        self.b_stall = b_stall
        self.rnd = random.Random(seed)
        self.memory = {}
        self.bursts = []   # (addr, beats)
        self.beats = []    # (addr, data, last, strb)
        self.current = None
        self.b_pending = 0

    async def run(self):
        d = self.dut
        d.io_axiMaster_aw_ready.value = 0
        d.io_axiMaster_w_ready.value = 0
        d.io_axiMaster_b_valid.value = 0
        d.io_axiMaster_b_payload_resp.value = 0
        d.io_axiMaster_b_payload_id.value = 0
        d.io_axiMaster_ar_ready.value = 0
        d.io_axiMaster_r_valid.value = 0
        d.io_axiMaster_r_payload_data.value = 0
        d.io_axiMaster_r_payload_last.value = 0
        d.io_axiMaster_r_payload_resp.value = 0
        d.io_axiMaster_r_payload_id.value = 0

        while True:
            await ReadOnly()

            aw_fire = int(d.io_axiMaster_aw_valid.value) and int(d.io_axiMaster_aw_ready.value)
            w_fire = int(d.io_axiMaster_w_valid.value) and int(d.io_axiMaster_w_ready.value)
            b_fire = int(d.io_axiMaster_b_valid.value) and int(d.io_axiMaster_b_ready.value)

            if aw_fire:
                assert self.current is None, "AW issued while a burst was still in flight"
                addr = int(d.io_axiMaster_aw_payload_addr.value)
                beats = int(d.io_axiMaster_aw_payload_len.value) + 1
                assert beats <= MAX_BURST_BEATS, f"Burst of {beats} beats exceeds maxBurstBeats"
                self.bursts.append((addr, beats))
                self.current = {"addr": addr, "left": beats}

            if w_fire:
                assert self.current is not None, "W beat without an accepted AW burst"
                data = int(d.io_axiMaster_w_payload_data.value)
                last = int(d.io_axiMaster_w_payload_last.value)
                strb = int(d.io_axiMaster_w_payload_strb.value)
                self.beats.append((self.current["addr"], data, last, strb))
                self.memory[self.current["addr"]] = data
                self.current["addr"] += AXI_BYTES
                self.current["left"] -= 1
                if self.current["left"] == 0:
                    self.b_pending += 1
                    self.current = None

            if b_fire:
                self.b_pending -= 1

            await RisingEdge(d.clk)

            d.io_axiMaster_aw_ready.value = 0 if self.rnd.random() < self.aw_stall else 1
            d.io_axiMaster_w_ready.value = 0 if self.rnd.random() < self.w_stall else 1
            d.io_axiMaster_b_valid.value = (
                1 if (self.b_pending > 0 and self.rnd.random() >= self.b_stall) else 0
            )


async def reset_dut(dut):
    dut.io_cmd_valid.value = 0
    dut.io_cmd_payload_address.value = 0
    dut.io_cmd_payload_length.value = 0
    dut.io_inStream_stream_valid.value = 0
    dut.io_inStream_stream_payload_0.value = 0
    dut.io_inStream_stream_payload_1.value = 0
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)


async def issue_cmd(dut, addr, length):
    dut.io_cmd_valid.value = 1
    dut.io_cmd_payload_address.value = addr
    dut.io_cmd_payload_length.value = length
    while True:
        await ReadOnly()
        ready = int(dut.io_cmd_ready.value)
        await RisingEdge(dut.clk)
        if ready:
            break
    dut.io_cmd_valid.value = 0


async def send_stream(dut, values, lanes=IN_LANES):
    idx = 0
    while idx < len(values):
        dut.io_inStream_stream_valid.value = 1
        dut.io_inStream_stream_payload_0.value = values[idx] & 0xFFFF
        dut.io_inStream_stream_payload_1.value = values[idx + 1] & 0xFFFF
        await ReadOnly()
        ready = int(dut.io_inStream_stream_ready.value)
        await RisingEdge(dut.clk)
        if ready:
            idx += lanes
    dut.io_inStream_stream_valid.value = 0


async def wait_done(dut, timeout=5000, dump=None):
    for _ in range(timeout):
        await ReadOnly()
        if int(dut.io_done.value) == 1:
            return
        await RisingEdge(dut.clk)
    extra = ""
    if dump is not None:
        extra = (
            f" bursts={dump.bursts} beats={len(dump.beats)} b_pending={dump.b_pending}"
            f" current={dump.current} busy={int(dut.io_busy.value)}"
            f" aw_valid={int(dut.io_axiMaster_aw_valid.value)} aw_ready={int(dut.io_axiMaster_aw_ready.value)}"
            f" w_valid={int(dut.io_axiMaster_w_valid.value)} w_ready={int(dut.io_axiMaster_w_ready.value)}"
            f" b_valid={int(dut.io_axiMaster_b_valid.value)} b_ready={int(dut.io_axiMaster_b_ready.value)}"
            f" stream_ready={int(dut.io_inStream_stream_ready.value)}"
        )
    raise AssertionError(f"DMAWriter never raised done{extra}")


def expected_word(values, word_idx):
    word = 0
    for b in range(ELEMS_PER_WORD):
        word |= (values[word_idx * ELEMS_PER_WORD + b] & 0xFFFF) << (16 * b)
    return word


def check_memory_image(slave, base_addr, values):
    for w in range(len(values) // ELEMS_PER_WORD):
        addr = base_addr + w * AXI_BYTES
        assert addr in slave.memory, f"Missing memory word at 0x{addr:X}"
        assert slave.memory[addr] == expected_word(values, w), (
            f"Word {w} at 0x{addr:X}: got 0x{slave.memory[addr]:016X} "
            f"instead of 0x{expected_word(values, w):016X}"
        )


def check_beat_metadata(slave):
    assert all(strb == 0xFF for _, _, _, strb in slave.beats), "w_strb must be all-ones"
    last_flags = [last for _, _, last, _ in slave.beats]
    assert sum(last_flags) == len(slave.bursts), "w_last count must match burst count"
    beat_idx = 0
    for _, beats in slave.bursts:
        for b in range(beats):
            expected_last = 1 if b == beats - 1 else 0
            assert last_flags[beat_idx + b] == expected_last, (
                f"w_last mismatch at beat {beat_idx + b} of burst {slave.bursts}"
            )
        beat_idx += beats


async def run_case(dut, base_addr, length, values, aw_stall=0.0, w_stall=0.0, b_stall=0.0, seed=1234):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    slave = AxiWriteSlave(dut, aw_stall=aw_stall, w_stall=w_stall, b_stall=b_stall, seed=seed)
    cocotb.start_soon(slave.run())

    await reset_dut(dut)
    await issue_cmd(dut, base_addr, length)
    await send_stream(dut, values)
    await wait_done(dut, dump=slave)

    await ReadOnly()
    assert int(dut.io_busy.value) == 0, "DMAWriter still busy after done"
    await RisingEdge(dut.clk)

    return slave


@cocotb.test()
async def cocotb_dma_writer_basic(dut):
    seed_random()
    values = [(i * 1000) - 8000 for i in range(N_ELEMS)]  # SInt16 range, signed mix

    slave = await run_case(dut, 0x1000, length=3, values=values)

    assert slave.bursts == [(0x1000, 4)], f"Expected a single 4-beat burst, got {slave.bursts}"
    check_memory_image(slave, 0x1000, values)
    check_beat_metadata(slave)
    print(f"DMAWriter basic burst wrote {len(values)} elements bit-exactly: {slave.bursts}")


@cocotb.test()
async def cocotb_dma_writer_4k_boundary(dut):
    # 0x1FF0 is 16 bytes (2 beats) before the 4 KiB page boundary 0x2000:
    # the 4-beat transfer must be split into 2 + 2 beats.
    seed_random()
    values = [(i * 1000) - 8000 for i in range(N_ELEMS)]

    slave = await run_case(dut, 0x1FF0, length=3, values=values)

    assert slave.bursts == [(0x1FF0, 2), (0x2000, 2)], (
        f"Expected 4 KiB boundary split [(0x1FF0, 2), (0x2000, 2)], got {slave.bursts}"
    )
    check_memory_image(slave, 0x1FF0, values)
    check_beat_metadata(slave)
    print(f"DMAWriter 4 KiB boundary split validated: {slave.bursts}")


@cocotb.test()
async def cocotb_dma_writer_backpressure(dut):
    seed_random()
    values = [((i * 977) % 60000) - 30000 for i in range(N_ELEMS)]

    slave = await run_case(
        dut,
        0x1000,
        length=3,
        values=values,
        aw_stall=0.3,
        w_stall=0.4,
        b_stall=0.3,
        seed=4321,
    )

    assert slave.bursts == [(0x1000, 4)], f"Expected a single 4-beat burst, got {slave.bursts}"
    check_memory_image(slave, 0x1000, values)
    check_beat_metadata(slave)
    print(f"DMAWriter random AW/W/B backpressure validated: {slave.bursts}")


def _run_sim(testcase, sim_build, request=None):
    v_file = run_mill(
        "spinalML.memory.DMAWriterTest",
        "Generate Verilog for DMAWriterTestWrapper",
        "DMAWriterTestComp",
    )

    from utils.test_layers_utils import safe_run_sim as run

    run(
        simulator="verilator",
        verilog_sources=[v_file],
        toplevel="DMAWriterTestComp",
        module="test_dma_writer",
        sim_build=sim_build,
        timescale="1ns/1ps",
        testcase=testcase,
        extra_args=["-Wno-fatal"],
    )


def test_dma_writer_basic(request):
    _run_sim("cocotb_dma_writer_basic", "sim_build/py_dma_writer_basic", request)


def test_dma_writer_4k_boundary(request):
    _run_sim("cocotb_dma_writer_4k_boundary", "sim_build/py_dma_writer_boundary", request)


def test_dma_writer_backpressure(request):
    _run_sim("cocotb_dma_writer_backpressure", "sim_build/py_dma_writer_backpressure", request)
