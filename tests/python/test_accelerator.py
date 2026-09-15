# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

# Python/Cocotb SoC-level co-simulation of the Accelerator top (Phase 4 of the
# Python coverage plan). Two generated tops are exercised:
#
#   - AcceleratorPassthroughTestComp : Flatten() only (identity), so the oracle
#     is the input image itself. Covers the AXI-Lite control plane (START,
#     status, TILE_CNT, RUN/STOP continuous auto-advance) and the DMAWriter
#     write-back path (CSR 0x20 / 0x24).
#   - AcceleratorRuntimeCastTestComp : Flatten() -> Cast(BF16, runtimeScale),
#     mirroring examples/Mnist/Model.scala + inference.py. Covers CSR 0x30 and
#     the programmable dequantization boundary dynamically.
#
# Python plays both roles: AXI4-Lite master for the CSR plane, and AXI4 slave
# (AR/R reads from an image dict, AW/W/B capture for write-back).

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge
from cocotb_test.simulator import run

from golden_models.dtypes import BF16
from golden_models.ops import cast_hw, floatml_mul
from utils.tb_utils import run_mill, seed_random

IMG_BASE = 0x10000
WEIGHT_BASE = 0x20000
OUT_BASE = 0x30000
FRAME_BYTES = 16
N_ELEMS = 16
FILTER = "Generate Verilog for Python co-simulation"


def to_signed(value, bits):
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


def pack_image(values):
    """I8 image bytes -> little-endian 64-bit words (zero padded)."""
    words = []
    data = bytes(v & 0xFF for v in values)
    for i in range(0, len(data), 8):
        chunk = data[i : i + 8]
        words.append(int.from_bytes(chunk.ljust(8, b"\x00"), "little"))
    return words


def write_image(memory, base, values):
    for i, word in enumerate(pack_image(values)):
        memory[base + i * 8] = word


class AxiMemSlave:
    """Minimal AXI4 slave: AR/R reads from `memory`, AW/W/B write-back capture."""

    def __init__(self, dut, memory, r_stall=0.0, b_stall=0.0, seed=1):
        self.dut = dut
        self.memory = memory
        self.r_stall = r_stall
        self.b_stall = b_stall
        self.rnd = random.Random(seed)
        self.written = {}
        self.bursts = []  # read bursts (addr, beats)

    async def run(self):
        d = self.dut
        d.io_axiMaster_aw_ready.value = 1
        d.io_axiMaster_w_ready.value = 1
        d.io_axiMaster_b_valid.value = 0
        d.io_axiMaster_b_payload_id.value = 0
        d.io_axiMaster_b_payload_resp.value = 0
        d.io_axiMaster_r_valid.value = 0
        d.io_axiMaster_r_payload_data.value = 0
        d.io_axiMaster_r_payload_last.value = 0
        d.io_axiMaster_r_payload_id.value = 0
        d.io_axiMaster_r_payload_resp.value = 0
        d.io_axiMaster_ar_ready.value = 1

        rd_queue = []
        rd_active = None
        wr_active = None
        b_pending = 0

        while True:
            await ReadOnly()
            ar_fire = int(d.io_axiMaster_ar_valid.value) and int(d.io_axiMaster_ar_ready.value)
            r_fire = int(d.io_axiMaster_r_valid.value) and int(d.io_axiMaster_r_ready.value)
            aw_fire = int(d.io_axiMaster_aw_valid.value) and int(d.io_axiMaster_aw_ready.value)
            w_fire = int(d.io_axiMaster_w_valid.value) and int(d.io_axiMaster_w_ready.value)
            b_fire = int(d.io_axiMaster_b_valid.value) and int(d.io_axiMaster_b_ready.value)

            if ar_fire:
                addr = int(d.io_axiMaster_ar_payload_addr.value)
                beats = int(d.io_axiMaster_ar_payload_len.value) + 1
                self.bursts.append((addr, beats))
                rd_queue.append((addr, beats))

            if r_fire:
                rd_active["addr"] += 8
                rd_active["left"] -= 1
                if rd_active["left"] == 0:
                    rd_active = None

            if aw_fire:
                assert wr_active is None, "AW issued while a burst was still in flight"
                wr_active = {
                    "addr": int(d.io_axiMaster_aw_payload_addr.value),
                    "left": int(d.io_axiMaster_aw_payload_len.value) + 1,
                }

            if w_fire:
                assert wr_active is not None, "W beat without an accepted AW burst"
                self.written[wr_active["addr"]] = int(d.io_axiMaster_w_payload_data.value)
                wr_active["addr"] += 8
                wr_active["left"] -= 1
                if wr_active["left"] == 0:
                    b_pending += 1
                    wr_active = None

            if b_fire:
                b_pending -= 1

            await RisingEdge(d.clk)

            if rd_active is None and rd_queue:
                addr, beats = rd_queue.pop(0)
                rd_active = {"addr": addr, "left": beats}

            if rd_active is not None and self.rnd.random() >= self.r_stall:
                d.io_axiMaster_r_valid.value = 1
                d.io_axiMaster_r_payload_data.value = self.memory.get(rd_active["addr"], 0)
                d.io_axiMaster_r_payload_last.value = 1 if rd_active["left"] == 1 else 0
            else:
                d.io_axiMaster_r_valid.value = 0
                d.io_axiMaster_r_payload_last.value = 0

            d.io_axiMaster_b_valid.value = (
                1 if (b_pending > 0 and self.rnd.random() >= self.b_stall) else 0
            )


async def reset_accel(dut):
    dut.io_ctrlBus_aw_valid.value = 0
    dut.io_ctrlBus_aw_payload_addr.value = 0
    dut.io_ctrlBus_w_valid.value = 0
    dut.io_ctrlBus_w_payload_data.value = 0
    dut.io_ctrlBus_w_payload_strb.value = 0xF
    dut.io_ctrlBus_b_ready.value = 1
    dut.io_ctrlBus_ar_valid.value = 0
    dut.io_ctrlBus_ar_payload_addr.value = 0
    dut.io_ctrlBus_r_ready.value = 1
    dut.io_outStream_stream_ready.value = 0
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)


async def axi_lite_write(dut, addr, data):
    aw_done = False
    w_done = False
    dut.io_ctrlBus_aw_valid.value = 1
    dut.io_ctrlBus_aw_payload_addr.value = addr & 0xFF
    dut.io_ctrlBus_w_valid.value = 1
    dut.io_ctrlBus_w_payload_data.value = data & 0xFFFFFFFF
    dut.io_ctrlBus_w_payload_strb.value = 0xF

    while not (aw_done and w_done):
        await ReadOnly()
        if not aw_done and int(dut.io_ctrlBus_aw_ready.value):
            aw_done = True
        if not w_done and int(dut.io_ctrlBus_w_ready.value):
            w_done = True
        await RisingEdge(dut.clk)
        if aw_done:
            dut.io_ctrlBus_aw_valid.value = 0
        if w_done:
            dut.io_ctrlBus_w_valid.value = 0

    # Wait for the write response (b_ready is tied high)
    while True:
        await ReadOnly()
        b_valid = int(dut.io_ctrlBus_b_valid.value)
        await RisingEdge(dut.clk)
        if b_valid:
            return


async def axi_lite_read(dut, addr):
    dut.io_ctrlBus_ar_valid.value = 1
    dut.io_ctrlBus_ar_payload_addr.value = addr & 0xFF
    while True:
        await ReadOnly()
        ar_ready = int(dut.io_ctrlBus_ar_ready.value)
        await RisingEdge(dut.clk)
        if ar_ready:
            break
    dut.io_ctrlBus_ar_valid.value = 0

    while True:
        await ReadOnly()
        r_valid = int(dut.io_ctrlBus_r_valid.value)
        data = int(dut.io_ctrlBus_r_payload_data.value) if r_valid else 0
        await RisingEdge(dut.clk)
        if r_valid:
            return data


async def collect_i8(dut, n, timeout=200000):
    out = []
    for _ in range(timeout):
        await ReadOnly()
        if int(dut.io_outStream_stream_valid.value):
            out.append(to_signed(int(dut.io_outStream_stream_payload_0.value), 8))
        await RisingEdge(dut.clk)
        if len(out) == n:
            return out
    raise AssertionError(f"Timed out collecting outputs: {len(out)}/{n}")


async def collect_bf16(dut, n, timeout=200000):
    out = []
    for _ in range(timeout):
        await ReadOnly()
        if int(dut.io_outStream_stream_valid.value):
            sign = int(dut.io_outStream_stream_payload_0_sign.value)
            exp = int(dut.io_outStream_stream_payload_0_exponent.value)
            mant = int(dut.io_outStream_stream_payload_0_mantissa.value)
            out.append((sign << (BF16.exp_bits + BF16.mant_bits)) | (exp << BF16.mant_bits) | mant)
        await RisingEdge(dut.clk)
        if len(out) == n:
            return out
    raise AssertionError(f"Timed out collecting BF16 outputs: {len(out)}/{n}")


async def wait_done(dut, timeout=200000):
    for _ in range(timeout):
        await ReadOnly()
        if int(dut.io_done.value) == 1:
            return
        await RisingEdge(dut.clk)
    raise AssertionError("Accelerator never raised done")


def setup_accel(dut, toplevel):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    return clock


@cocotb.test()
async def cocotb_accel_passthrough_oneshot(dut):
    seed_random()
    setup_accel(dut, "AcceleratorPassthroughTestComp")

    values = [((i * 9) % 251) - 125 for i in range(N_ELEMS)]
    memory = {}
    write_image(memory, IMG_BASE, values)
    slave = AxiMemSlave(dut, memory)
    cocotb.start_soon(slave.run())

    await reset_accel(dut)
    await axi_lite_write(dut, 0x08, IMG_BASE)
    await axi_lite_write(dut, 0x0C, WEIGHT_BASE)
    dut.io_outStream_stream_ready.value = 1
    await axi_lite_write(dut, 0x00, 1)

    out = await collect_i8(dut, N_ELEMS)
    assert out == values, f"Passthrough mismatch: {out} != {values}"

    tile_cnt = await axi_lite_read(dut, 0x18)
    assert tile_cnt == 1, f"TILE_CNT expected 1, got {tile_cnt}"
    status = await axi_lite_read(dut, 0x04)
    assert (status >> 1) & 1 == 0, f"Accelerator still busy after one-shot: status=0x{status:X}"

    print(f"Accelerator one-shot passthrough bit-exact: {len(out)} elements, TILE_CNT=1")


@cocotb.test()
async def cocotb_accel_passthrough_continuous(dut):
    seed_random()
    setup_accel(dut, "AcceleratorPassthroughTestComp")

    frames = [
        [((i * (k + 3) + k) % 251) - 125 for i in range(N_ELEMS)]
        for k in range(3)
    ]
    memory = {}
    for k, frame in enumerate(frames):
        write_image(memory, IMG_BASE + k * FRAME_BYTES, frame)
    slave = AxiMemSlave(dut, memory)
    cocotb.start_soon(slave.run())

    await reset_accel(dut)
    await axi_lite_write(dut, 0x08, IMG_BASE)
    await axi_lite_write(dut, 0x0C, WEIGHT_BASE)
    dut.io_outStream_stream_ready.value = 1
    await axi_lite_write(dut, 0x1C, 1)  # RUN
    await axi_lite_write(dut, 0x00, 1)  # single START

    collected = []

    async def collector():
        while len(collected) < 4 * N_ELEMS:
            await ReadOnly()
            if int(dut.io_outStream_stream_valid.value):
                collected.append(to_signed(int(dut.io_outStream_stream_payload_0.value), 8))
            await RisingEdge(dut.clk)

    cocotb.start_soon(collector())

    while len(collected) < 2 * N_ELEMS:
        await RisingEdge(dut.clk)
    await axi_lite_write(dut, 0x1C, 0)  # STOP after frame 2 (in-flight completes)

    timeout = 0
    while len(collected) < 3 * N_ELEMS and timeout < 50000:
        await RisingEdge(dut.clk)
        timeout += 1
    assert len(collected) >= 3 * N_ELEMS, f"Only {len(collected)} beats after STOP"
    got_frames = [collected[k * N_ELEMS : (k + 1) * N_ELEMS] for k in range(3)]
    for k in range(3):
        assert got_frames[k] == frames[k], f"Frame {k} mismatch: {got_frames[k]} != {frames[k]}"

    # Silence after STOP
    silence = 0
    for _ in range(1000):
        await ReadOnly()
        if int(dut.io_outStream_stream_valid.value):
            silence = 0
        else:
            silence += 1
        await RisingEdge(dut.clk)
        if silence > 300:
            break
    assert silence > 300, "Accelerator kept streaming after STOP"
    assert len(collected) == 3 * N_ELEMS, f"Unexpected extra beats after STOP: {len(collected)}"

    tile_cnt = await axi_lite_read(dut, 0x18)
    assert tile_cnt == 3, f"TILE_CNT expected 3 after STOP, got {tile_cnt}"
    print("Accelerator continuous RUN/STOP: 3 frames bit-exact, clean silence after STOP")


@cocotb.test()
async def cocotb_accel_dma_write_back(dut):
    seed_random()
    setup_accel(dut, "AcceleratorPassthroughTestComp")

    values = [((i * 13) % 251) - 125 for i in range(N_ELEMS)]
    memory = {}
    write_image(memory, IMG_BASE, values)
    slave = AxiMemSlave(dut, memory)
    cocotb.start_soon(slave.run())

    await reset_accel(dut)
    await axi_lite_write(dut, 0x08, IMG_BASE)
    await axi_lite_write(dut, 0x0C, WEIGHT_BASE)
    await axi_lite_write(dut, 0x20, OUT_BASE)
    await axi_lite_write(dut, 0x24, 1)  # writeToDdr

    status_before = await axi_lite_read(dut, 0x28)
    assert status_before & 0x1 == 0, f"DMA writer busy before start: 0x{status_before:X}"

    await axi_lite_write(dut, 0x00, 1)

    saw_valid = False
    for _ in range(100000):
        await ReadOnly()
        if int(dut.io_outStream_stream_valid.value):
            saw_valid = True
        done = int(dut.io_done.value) == 1
        await RisingEdge(dut.clk)
        if done:
            break
    else:
        raise AssertionError("Timed out waiting for write-back completion")

    assert not saw_valid, "outStream.valid pulsed while writeToDdr was active"

    # 16 bytes = 2 beats at OUT_BASE, element b at bits 8b of each word
    expected_words = []
    for w in range(2):
        word = 0
        for b in range(8):
            word |= (values[w * 8 + b] & 0xFF) << (8 * b)
        expected_words.append(word)
    for w in range(2):
        addr = OUT_BASE + w * 8
        assert addr in slave.written, f"Missing write-back word at 0x{addr:X}"
        assert slave.written[addr] == expected_words[w], (
            f"Write-back word {w} at 0x{addr:X}: got 0x{slave.written[addr]:016X} "
            f"instead of 0x{expected_words[w]:016X}"
        )

    status_after = await axi_lite_read(dut, 0x28)
    assert status_after & 0x1 == 0, f"DMA writer still busy after completion: 0x{status_after:X}"
    print("Accelerator DMA write-back bit-exact (CSR 0x20/0x24/0x28), outStream silent")


@cocotb.test()
async def cocotb_accel_runtime_scale(dut):
    seed_random()
    setup_accel(dut, "AcceleratorRuntimeCastTestComp")

    values = [((i * 11) % 251) - 125 for i in range(N_ELEMS)]
    memory = {}
    write_image(memory, IMG_BASE, values)
    slave = AxiMemSlave(dut, memory)
    cocotb.start_soon(slave.run())

    await reset_accel(dut)
    await axi_lite_write(dut, 0x08, IMG_BASE)
    await axi_lite_write(dut, 0x0C, WEIGHT_BASE)
    dut.io_outStream_stream_ready.value = 1

    # Default scale must be 1.0 (encoded at elaboration)
    default_scale = await axi_lite_read(dut, 0x30) & 0xFFFF
    assert default_scale == BF16.from_float(1.0), (
        f"Default CSR 0x30 should encode 1.0, got 0x{default_scale:04X}"
    )

    for scale in [0.5, 2.0, -1.0]:
        scale_bits = BF16.from_float(scale)
        await axi_lite_write(dut, 0x30, scale_bits)
        await axi_lite_write(dut, 0x00, 1)
        out = await collect_bf16(dut, N_ELEMS)

        expected = [
            BF16.from_float(
                floatml_mul(BF16.to_float(cast_hw(x, 8, BF16)), BF16.to_float(scale_bits), BF16)
            )
            for x in values
        ]
        assert out == expected, f"Runtime scale {scale}: {out} != {expected}"
        print(f"Accelerator CSR 0x30 scale={scale}: 16 BF16 outputs bit-exact")

    print("Accelerator runtime dequantization boundary validated dynamically")


def _run_sim(testcase, toplevel, sim_build, request=None):
    v_file = run_mill("spinalML.nn.AcceleratorTest", FILTER, toplevel)

    from utils.test_layers_utils import safe_run_sim as run

    run(
        simulator="verilator",
        verilog_sources=[v_file],
        toplevel=toplevel,
        module="test_accelerator",
        testcase=testcase,
        sim_build=sim_build,
        timescale="1ns/1ps",
        extra_args=["-Wno-fatal"],
    )


def test_accel_passthrough_oneshot(request):
    _run_sim(
        "cocotb_accel_passthrough_oneshot",
        "AcceleratorPassthroughTestComp",
        "sim_build/py_accel_oneshot",
        request,
    )


def test_accel_passthrough_continuous(request):
    _run_sim(
        "cocotb_accel_passthrough_continuous",
        "AcceleratorPassthroughTestComp",
        "sim_build/py_accel_continuous",
        request,
    )


def test_accel_dma_write_back(request):
    _run_sim(
        "cocotb_accel_dma_write_back",
        "AcceleratorPassthroughTestComp",
        "sim_build/py_accel_writeback",
        request,
    )


def test_accel_runtime_scale(request):
    _run_sim(
        "cocotb_accel_runtime_scale",
        "AcceleratorRuntimeCastTestComp",
        "sim_build/py_accel_runtime_scale",
        request,
    )
