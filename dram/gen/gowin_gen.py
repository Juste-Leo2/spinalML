#!/usr/bin/env python3
"""LiteDRAM standalone core generator for Gowin GW2A boards (CPU-free).

Upstream ``litedram.gen`` only supports SDR / ECP5 / Xilinx PHYs, so this
script composes the equivalent core by hand from LiteDRAM blocks:

  board clk27 -> CRG (GW2APLL + DHCEN/CLKDIV, copied from the litex-boards
  Tang Primer 20K target) -> GW2DDRPHY -> LiteDRAMCore (controller +
  crossbar) -> 64-bit AXI port -> top-level ``litedram_core`` Verilog.

Two deliberate v1 choices (see also dram/configs/tang-primer-20k.yml):

- ``sys_clk_freq`` defaults to 27MHz (same clock as the SpinalHDL SoC, so
  no clock-domain crossing on the AXI link). The DDR runs at 2x sys
  (54MHz): very slow for DDR3, huge timing margins, plenty for weight
  streaming. Raising it later only needs a CDC bridge on the SpinalHDL
  side (tracked follow-up, config value is already parametric).
- No CPU/BIOS exists here, so the JEDEC DDR3 init sequence (normally
  played by software through DFII) is played by the gateware
  ``DramInitSequencer`` below, driving ``DFIInjector.ext_dfi`` in
  hardware-control mode (``_control.sel`` resets to 1). ``init_done``
  switches the DFI mux to the controller and gates the SpinalHDL side.

PHY read-leveling/delay calibration stays at reset defaults (no software
to train them); at 54MHz DDR the margins absorb it. Silicon bring-up
will confirm.

Runs in the uv-managed env (migen/litex/litedram), never in the CLI env::

    ~/.spinalml_tools/.venv/bin/python dram/gen/gowin_gen.py \
        --config dram/configs/tang-primer-20k.yml --out dram/out
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from migen import *
from migen.genlib.resetsync import AsyncResetSynchronizer
from migen.fhdl import verilog

from litex.soc.cores.clock.gowin_gw2a import GW2APLL

from litedram.common import PhySettings  # noqa: F401 (re-exported for configs)
from litedram.core import LiteDRAMCore
from litedram.frontend.axi import LiteDRAMAXIPort, LiteDRAMAXI2Native
from litedram.init import cmds as DFII_CMDS, get_ddr3_phy_init_sequence
from litedram.phy.gw2ddrphy import GW2DDRPHY

from sdram_modules import H5TC1G63EFR

# Available SDRAM modules by config name (upstream LiteDRAM has no Hynix 1Gb).
SDRAM_MODULES = {
    "H5TC1G63EFR": H5TC1G63EFR,
}

try:
    import litedram.modules as _litedram_modules
    for _name in ("MT41K64M16", "IMD128M16R39CG8GNF", "MT41J128M16"):
        SDRAM_MODULES.setdefault(_name, getattr(_litedram_modules, _name))
except ImportError:  # pragma: no cover (litedram always present in managed env)
    pass


# Board pads -------------------------------------------------------------------

class TangPrimer20KDDRPads:
    """DDR3 pads of the Sipeed Tang Primer 20K (litex-boards pinout).

    The footprint is shared across fitted chip revisions (IMD/Hynix), so the
    pins hold for the H5TC1G63EFR variant; only geometry/timings differ and
    come from the SDRAM module class above.
    """
    def __init__(self):
        self.a       = Signal(14)
        self.ba      = Signal(3)
        self.ras_n   = Signal()
        self.cas_n   = Signal()
        self.we_n    = Signal()
        self.cs_n    = Signal()
        self.dm      = Signal(2)
        self.dq      = Signal(16)
        self.dqs_p   = Signal(2)
        self.dqs_n   = Signal(2)
        self.clk_p   = Signal()
        self.clk_n   = Signal()
        self.cke     = Signal()
        self.odt     = Signal()
        self.reset_n = Signal()

    def ios(self):
        return [self.a, self.ba, self.ras_n, self.cas_n, self.we_n, self.cs_n,
                self.dm, self.dq, self.dqs_p, self.dqs_n, self.clk_p, self.clk_n,
                self.cke, self.odt, self.reset_n]


DDR_PINMAP = {
    # Maps pad attribute -> FPGA pin(s) for the constraints emitter.
    "a":       "F7 A4 D6 F8 C4 E6 B1 D8 A5 F9 K3 B7 A3 C8",
    "ba":      "H4 D3 H5",
    "ras_n":   "R4",
    "cas_n":   "R6",
    "we_n":    "L2",
    "cs_n":    "P5",
    "dm":      "G1 K5",
    "dq":      "G5 F5 F4 F3 E2 C1 E1 B3 M3 K4 N2 L1 P4 H3 R1 M2",
    "dqs_p":   "G2 J5",
    "dqs_n":   "G3 K6",
    "clk_p":   "J1",
    "clk_n":   "J3",
    "cke":     "J2",
    "odt":     "R3",
    "reset_n": "B9",
}


# Clock/reset generation (from litex-boards Tang Primer 20K target) ------------

class GowinDramCRG(Module):
    """clk27 -> GW2APLL -> sys / sys2x / init clock domains."""
    def __init__(self, clk27, sys_clk_freq, device, devicename):
        self.rst     = Signal()
        self.cd_sys  = ClockDomain()
        self.cd_sys2x   = ClockDomain()
        self.cd_sys2x_i = ClockDomain()
        self.cd_init = ClockDomain()
        self.cd_por  = ClockDomain()
        self.stop  = Signal()
        self.reset = Signal()
        self.locked = Signal()

        # Power-on reset: the on-board POR is not aware of reprogramming.
        por_count = Signal(16, reset=2**16-1)
        por_done  = Signal()
        self.comb += self.cd_por.clk.eq(clk27)
        self.comb += por_done.eq(por_count == 0)
        self.sync.por += If(~por_done, por_count.eq(por_count - 1))

        # PLL: 2:1 clock needed for DDR.
        # NB: explicit self.submodules (plain attribute assignment does NOT
        # register a submodule in Migen).
        self.submodules.pll = pll = GW2APLL(devicename=devicename, device=device)
        self.comb += pll.reset.eq(~por_done)
        pll.register_clkin(clk27, 27e6)
        # with_reset=False: AsyncResetSynchronizer cannot lower without a
        # LiteX platform (see sys reset below); fast-domain reset is built
        # explicitly instead.
        pll.create_clkout(self.cd_sys2x_i, 2*sys_clk_freq, with_reset=False)
        self.specials += [
            Instance("DHCEN",
                i_CLKIN  = self.cd_sys2x_i.clk,
                i_CE     = self.stop,
                o_CLKOUT = self.cd_sys2x.clk),
            Instance("CLKDIV",
                p_DIV_MODE = "2",
                i_CALIB    = 0,
                i_HCLKIN   = self.cd_sys2x.clk,
                i_RESETN   = ~self.reset,
                o_CLKOUT   = self.cd_sys.clk),
        ]

        # Init clock domain (free-running clk27, reset by PLL reset).
        self.comb += self.cd_init.clk.eq(clk27)
        self.comb += self.cd_init.rst.eq(pll.reset)

        self.comb += self.locked.eq(pll.locked)
        # Reset release synchronizer (plain flops: verilog.convert runs
        # without a LiteX platform, where AsyncResetSynchronizer cannot
        # lower). Sources are quasi-static (POR counter, PLL lock), so a
        # synchronous 2-stage release is equivalent in practice.
        _rst0 = Signal(reset=1)
        _rst1 = Signal(reset=1)
        self.sync.sys += [
            _rst0.eq(~pll.locked | self.rst | self.reset),
            _rst1.eq(_rst0),
        ]
        self.comb += self.cd_sys.rst.eq(_rst1)
        # Fast-domain reset (self-starting from bitstream init values).
        _rst2x_0 = Signal(reset=1)
        _rst2x_1 = Signal(reset=1)
        self.sync.sys2x_i += [
            _rst2x_0.eq(~pll.locked | self.reset),
            _rst2x_1.eq(_rst2x_0),
        ]
        self.comb += self.cd_sys2x_i.rst.eq(_rst2x_1)


# JEDEC DDR3 init sequencer (replaces the BIOS) --------------------------------

def _decode_dfii_command(cmd):
    """Splits a DFII command string (e.g. from litedram.init.cmds) into levels.

    DFII_COMMAND_* bits assert the corresponding strobe low; DFII_CONTROL_*
    bits drive the level high (mirrors PhaseInjector semantics).
    """
    parts = set(cmd.split("|"))
    return {
        "ras_n":   0 if "DFII_COMMAND_RAS" in parts else 1,
        "cas_n":   0 if "DFII_COMMAND_CAS" in parts else 1,
        "we_n":    0 if "DFII_COMMAND_WE"  in parts else 1,
        "cs_n":    0 if "DFII_COMMAND_CS"  in parts else 1,
        "cke":     1 if "DFII_CONTROL_CKE"     in parts else 0,
        "odt":     1 if "DFII_CONTROL_ODT"     in parts else 0,
        "reset_n": 1 if "DFII_CONTROL_RESET_N" in parts else 0,
    }


class DramInitSequencer(Module):
    """Plays the JEDEC DDR3 init sequence through DFIInjector.ext_dfi.

    Steps (address/bank/delay) come from
    ``litedram.init.get_ddr3_phy_init_sequence`` so MR values always match
    the PHY settings; only the command decode above maps DFII strings to
    DFI levels. The command is issued on phase 0, NOP on other phases.
    Per-step wait is ``max(2*sequence_delay, 4)`` sys cycles: the x2 margin
    covers slow-clock scaling, the 4-cycle floor covers tMRD between
    back-to-back mode-register writes.
    """
    def __init__(self, dfii, phy_settings, timing_settings, nphases):
        self.init_done = Signal(reset=0)

        sequence, _ = get_ddr3_phy_init_sequence(phy_settings, timing_settings)

        # Elaborate steps: (comment, decoded_levels, address, bank, wait_cycles).
        steps = []
        cke_latched = False
        rst_latched = False
        for comment, address, bank, cmd, delay in sequence:
            levels = _decode_dfii_command(cmd)
            # CKE / reset_n stay asserted once set by an earlier step.
            cke_latched = cke_latched or ("DFII_CONTROL_CKE" in cmd)
            rst_latched = rst_latched or ("DFII_CONTROL_RESET_N" in cmd)
            levels["cke"] = 1 if cke_latched else 0
            levels["reset_n"] = 1 if rst_latched else 0
            steps.append((comment, levels, address, bank, max(2*delay, 4)))

        ext = dfii.ext_dfi
        max_wait = max(s[4] for s in steps) + 1
        wait = Signal(max=max_wait)
        idx  = Signal(max=len(steps) + 1)

        self.comb += dfii.ext_dfi_sel.eq(~self.init_done)

        # Default (idle/NOP) DFI drive.
        for phase in ext.phases:
            self.comb += [
                phase.address.eq(0),
                phase.bank.eq(0),
                phase.ras_n.eq(1),
                phase.cas_n.eq(1),
                phase.we_n.eq(1),
                phase.cs_n.eq(1),
                phase.cke.eq(0),
                phase.odt.eq(0),
                phase.reset_n.eq(0),
                phase.act_n.eq(1),
                phase.wrdata.eq(0),
                phase.wrdata_en.eq(0),
                phase.wrdata_mask.eq(0),
                phase.rddata_en.eq(0),
            ]

        fsm = FSM(reset_state="SETTLE")
        self.submodules += fsm
        # Clock stabilization grace period after sys reset release.
        fsm.act("SETTLE",
            NextValue(wait, 1024),
            NextState("WAIT"),
            NextValue(idx, 0),
        )
        for i, (comment, levels, address, bank, cycles) in enumerate(steps):
            state = "STEP%d" % i
            nxt = ("STEP%d" % (i + 1)) if (i + 1) < len(steps) else "DONE"
            fsm.act(state,
                ext.p0.address.eq(address),
                ext.p0.bank.eq(bank),
                ext.p0.ras_n.eq(levels["ras_n"]),
                ext.p0.cas_n.eq(levels["cas_n"]),
                ext.p0.we_n.eq(levels["we_n"]),
                ext.p0.cs_n.eq(levels["cs_n"]),
                ext.p0.cke.eq(levels["cke"]),
                ext.p0.odt.eq(levels["odt"]),
                ext.p0.reset_n.eq(levels["reset_n"]),
                NextValue(wait, cycles),
                NextState("WAIT"),
                NextValue(idx, i + 1),
            )
        fsm.act("WAIT",
            If(wait == 0,
                If(idx >= len(steps),
                    NextState("DONE"),
                ).Else(
                    Case(idx, {i: NextState("STEP%d" % i) for i in range(len(steps))}),
                ),
            ).Else(
                NextValue(wait, wait - 1),
            ),
        )
        fsm.act("DONE",
            self.init_done.eq(1),
        )


# Top --------------------------------------------------------------------------

class LiteDramGowinTop(Module):
    """Complete CPU-free LiteDRAM core for Gowin GW2A.

    External interface: clk27 / reset_n (board) + 64-bit AXI4 slave port
    (SpinalHDL side) + ddram pads + init_done / pll_locked status.
    Everything else (sys/sys2x/init domains, DLL/ECLK init, refresh,
    JEDEC init) is internal.
    """
    def __init__(self, config):
        sys_clk_freq = float(config["sys_clk_freq"])
        device       = config["device"]
        devicename   = config["devicename"]
        axi_id_width = int(config.get("axi_id_width", 4))

        self.clk27   = Signal()
        self.reset_n = Signal(reset=1)
        self.init_done  = Signal()
        self.pll_locked = Signal()

        # CRG ------------------------------------------------------------------
        self.submodules.crg = crg = GowinDramCRG(
            clk27=self.clk27, sys_clk_freq=sys_clk_freq,
            device=device, devicename=devicename)
        self.comb += [
            crg.rst.eq(~self.reset_n),
            self.pll_locked.eq(crg.locked),
        ]
        self.clock_domains.cd_sys     = crg.cd_sys
        self.clock_domains.cd_sys2x   = crg.cd_sys2x
        self.clock_domains.cd_sys2x_i = crg.cd_sys2x_i
        self.clock_domains.cd_init    = crg.cd_init
        self.clock_domains.cd_por     = crg.cd_por

        # SDRAM module + PHY ---------------------------------------------------
        module_cls = SDRAM_MODULES[config["sdram_module"]]
        module = module_cls(sys_clk_freq, "1:2")
        pads = TangPrimer20KDDRPads()
        self.submodules.ddrphy = phy = GW2DDRPHY(
            pads,
            sys_clk_freq = sys_clk_freq,
            dll_off      = (2*sys_clk_freq <= 125e6),
        )
        self.pads = pads

        # Controller + crossbar ------------------------------------------------
        self.submodules.core = core = LiteDRAMCore(
            phy             = phy,
            geom_settings   = module.geom_settings,
            timing_settings = module.timing_settings,
            clk_freq        = sys_clk_freq,
        )

        # JEDEC init (BIOS replacement) ----------------------------------------
        self.submodules.initseq = initseq = DramInitSequencer(
            dfii=core.dfii,
            phy_settings=phy.settings,
            timing_settings=module.timing_settings,
            nphases=phy.settings.nphases,
        )
        self.comb += self.init_done.eq(initseq.init_done)

        # AXI slave port (SpinalHDL DdrAdapter side, 64-bit) --------------------
        port = core.crossbar.get_port(data_width=64)
        axi_address_width = port.address_width + log2_int(64//8)
        # NB: plain attribute, NOT a submodule (AXIInterface is a Record).
        axi = LiteDRAMAXIPort(
            data_width    = 64,
            address_width = axi_address_width,
            id_width      = axi_id_width,
        )
        self.submodules.axi2native = LiteDRAMAXI2Native(axi=axi, port=port)
        self.axi = axi

    def ios(self):
        # verilog.convert only accepts leaf Signals (no Records).
        ios = [self.clk27, self.reset_n, self.init_done, self.pll_locked]
        for channel in (self.axi.aw, self.axi.w, self.axi.b, self.axi.ar, self.axi.r):
            ios += channel.flatten()
        ios += self.pads.ios()
        return set(ios)


# Main -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gowin GW2A LiteDRAM core generator")
    parser.add_argument("--config", required=True, help="Board YAML config (dram/configs/*.yml)")
    parser.add_argument("--out", required=True, help="Output directory for litedram_core.v")
    parser.add_argument("--name", default="litedram_core", help="Top module name")
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key in ("sys_clk_freq", "device", "devicename", "sdram_module"):
        if key not in config:
            sys.exit("config %s: missing required key '%s'" % (args.config, key))
    if config["sdram_module"] not in SDRAM_MODULES:
        sys.exit("config %s: unknown sdram_module '%s' (known: %s)" % (
            args.config, config["sdram_module"], sorted(SDRAM_MODULES)))

    top = LiteDramGowinTop(config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    v = verilog.convert(top, name=args.name, ios=top.ios())
    out_path = out_dir / (args.name + ".v")
    out_path.write_text(str(v), encoding="utf-8")
    print("Wrote %s" % out_path)


if __name__ == "__main__":
    main()
