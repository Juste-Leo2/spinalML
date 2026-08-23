import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import random

from golden_models.dtypes import I8
from utils.tb_utils import run_mill, seed_random
from utils.test_layers_utils import send_tensor

# Flow-control check of the DAG tap primitive (TapBuffer tee): a single
# producer stream is mirrored into a direct passthrough and an exact-capacity
# FIFO. The deferred branch may drain arbitrarily late (random stall bursts):
# it must deliver the exact source sequence, order preserved.
#
# Both observed streams are REGISTERED inside the DUT (m2sPipe on the direct
# branch, FIFO pop register on the tapped branch): sampling them right after
# a clock edge is deterministic. Icarus on purpose: control-heavy logic.

BEATS = 8
LANES = 4
ELEMENTS = BEATS * LANES  # equals the FIFO capacity: one full tile, never overflows


async def deferred_consumer_task(dut, out_list, stall_prob, max_stall_cycles):
    """Drain the tapped branch with random stall bursts.

    Per iteration: one clock edge with ready as decided; if the (registered)
    valid reads 1 afterwards, that edge popped exactly one FIFO word -> capture.
    """
    got = 0
    while got < ELEMENTS:
        if random.random() < stall_prob:
            dut.io_deferred_stream_ready.value = 0
            for _ in range(random.randint(1, max_stall_cycles)):
                await RisingEdge(dut.clk)
        dut.io_deferred_stream_ready.value = 1
        await RisingEdge(dut.clk)
        if int(dut.io_deferred_stream_valid.value) == 1:
            out_list.extend(
                int(getattr(dut, f"io_deferred_stream_payload_{l}").value.signed_integer)
                for l in range(LANES)
            )
            got += LANES


async def direct_consumer_task(dut, out_list):
    """Direct branch drains as fast as the source offers (ready tied high)."""
    got = 0
    dut.io_direct_stream_ready.value = 1
    while got < ELEMENTS:
        await RisingEdge(dut.clk)
        if int(dut.io_direct_stream_valid.value) == 1:
            out_list.extend(
                int(getattr(dut, f"io_direct_stream_payload_{l}").value.signed_integer)
                for l in range(LANES)
            )
            got += LANES


@cocotb.test()
async def run_tap_buffer_sim(dut):
    seed_random()
    rnd = random.Random(1234)

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.reset.value = 1
    dut.io_in_stream_valid.value = 0
    dut.io_direct_stream_ready.value = 0
    dut.io_deferred_stream_ready.value = 0
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)

    TRIALS = 5
    for trial in range(TRIALS):
        src = [[rnd.randint(-128, 127) for _ in range(LANES)] for _ in range(BEATS)]
        flat_src = [v for row in src for v in row]

        direct_out = []
        deferred_out = []

        recv_d = cocotb.start_soon(direct_consumer_task(dut, direct_out))
        recv_f = cocotb.start_soon(
            deferred_consumer_task(
                dut,
                deferred_out,
                stall_prob=rnd.uniform(0.3, 0.9),
                max_stall_cycles=rnd.randint(2, 8),
            )
        )

        # Repo-proven sender (waits io_in_stream_ready after each offer).
        await send_tensor(
            dut, "io_in_stream", flat_src, (1, ELEMENTS), LANES, I8, False
        )

        await recv_d
        await recv_f

        assert direct_out == flat_src, \
            f"Trial {trial}: direct stream corrupted ({len(direct_out)} elems)"
        assert deferred_out == flat_src, \
            f"Trial {trial}: deferred stream corrupted ({len(deferred_out)} elems)"

        print(f"Trial {trial} OK (deferred stalled randomly)")

    print("TapBuffer flow control validated under random deferred backpressure (Python/Cocotb/Icarus)!")


def test_pytest_tap_buffer():
    v_file = run_mill("spinalML.memory.TapBufferTest", "cocotb", "TapBufferTestComp")

    from utils.test_layers_utils import safe_run_sim as run
    run(
        simulator="icarus",
        verilog_sources=[v_file],
        toplevel="TapBufferTestComp",
        module="test_tap_buffer",
        timescale="1ns/1ps",
        testcase="run_tap_buffer_sim"
    )
