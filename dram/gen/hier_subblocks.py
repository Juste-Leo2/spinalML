"""Voie B dissection: emit one standalone flat Verilog per LiteDRAM sub-block.

Every *boundary + internal* signal of the block is exposed as a port, so
GowinSynthesis sees live inputs and observed outputs: a healthy block keeps
its size, a block with a Gowin-indigestible construct collapses.

Run with the managed env (same link as `cli/main.py dram-gen`)::

    C:\\Users\\leona\\.spinalml_tools\\.venv\\Scripts\\python.exe \\
        E:\\spinalML\\dram\\gen\\hier_subblocks.py \\
        --config E:\\spinalML\\dram\\configs\\tang-primer-20k.yml \\
        --out E:\\spinalML\\dram\\out\\hier
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from migen import Module, Signal
from migen.fhdl import verilog
from migen.fhdl.tools import list_signals, list_special_ios

import gowin_gen

BLOCKS = ("crg", "ddrphy", "core", "initseq", "axi2native")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    top = gowin_gen.LiteDramGowinTop(config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in BLOCKS:
        sub = getattr(top, name)
        # Every signal referenced by the block fragment (statements AND
        # specials like rPLL/Instance ports) becomes a port: live inputs,
        # observed outputs. Attribute-walking misses statement-local
        # signals (e.g. initseq's ext_dfi), fragment listing does not.
        frag = sub.get_fragment()
        ios = list_signals(frag) | list_special_ios(frag, True, True, True)
        # NB: pass the fragment itself: Module.get_fragment() may only be
        # called once, and convert() would call it a second time.
        v = verilog.convert(frag, name="sub_" + name, ios=ios)
        path = out_dir / ("sub_%s.v" % name)
        path.write_text(str(v), encoding="utf-8")
        print("Wrote %s (signals=%d)" % (path, len(ios)))


if __name__ == "__main__":
    main()
