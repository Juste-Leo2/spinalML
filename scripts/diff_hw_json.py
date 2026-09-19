#!/usr/bin/env python3
# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""
Structural diff of two SpinalML Gowin hardware builds.

Compares the Yosys netlist (synth.json), the placed & routed design
(pnr.json) and the Yosys pass/warning log (build.log) of a known-bad build
against a known-good reference. The goal is to separate:

  * functional differences : cell connections / parameters / memory lanes
  * placement differences  : NEXTPNR_BEL changes
  * routing differences    : net ROUTING strings

Only stdlib is required.

Usage:
  python scripts/diff_hw_json.py --ko hw_build/bisect-rne/b823d07-rne \
                                 --ok hw_build/bisect-rne/b823d07-trunc
  python scripts/diff_hw_json.py --matrix [--root DIR] [--out-dir DIR]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "hw_build" / "bisect-rne"

# Pixel-level ground truth for the MNIST RNE/DSP regression: b823d07-rne
# saturates the logits, the three other builds are bit-exact.
MATRIX_PAIRS = [
    ("b823d07-rne", "b823d07-rne-nodsp"),
    ("b823d07-rne", "b823d07-trunc"),
    ("b823d07-rne", "7a96779"),
    ("b823d07-trunc", "7a96779"),  # control: two working builds
]

PRIMITIVE_TYPES = {
    "MULT18X18", "MULT9X9", "RAM16SDP4", "SDP", "DP", "ALU",
    "MUX2_LUT5", "MUX2_LUT6", "MUX2_LUT7", "MUX2_LUT8",
}
RAM_TYPES = {"RAM16SDP4", "SDP", "DP"}
# Types whose cell identity (hdlname) is stable and meaningful across builds.
ANCHOR_TYPES = {"MULT18X18", "MULT9X9"}

PASS_RE = re.compile(r"^\s*\d+(?:\.\d+)*\. Executing (.+?)(?:\s*\(|$)")
WARN_RE = re.compile(r"Warning:\s*(.*)")

_json_cache = {}


def load_json(path):
    path = str(path)
    if path not in _json_cache:
        with open(path) as f:
            _json_cache[path] = json.load(f)
    return _json_cache[path]


def pick_module(data):
    """synth.json tops on UartSoC, pnr.json on `top`."""
    mods = data.get("modules", {})
    if "top" in mods:
        return "top", mods["top"]
    best_name, best_mod = None, None
    for name, mod in mods.items():
        n = len(mod.get("cells", {}))
        if best_mod is None or n > len(best_mod.get("cells", {})):
            best_name, best_mod = name, mod
    return best_name, best_mod


def bit_table(mod):
    tbl = {}
    for name, net in mod.get("netnames", {}).items():
        for b in net.get("bits", []):
            tbl[b] = name
    return tbl


def port_nets(cell, tbl):
    out = {}
    for port, bits in cell.get("connections", {}).items():
        if bits:
            out[port] = tuple(tbl.get(b, f"#{b}") for b in bits)
    return out


def cell_hid(cell):
    attrs = cell.get("attributes", {})
    return attrs.get("hdlname") or attrs.get("src") or ""


def cell_bel(cell):
    return cell.get("attributes", {}).get("NEXTPNR_BEL")


def canon(name):
    """Rename-insensitive net name (digit runs collapsed)."""
    return re.sub(r"\d+", "#", name)


def sig_strict(conns):
    return tuple(sorted((port, nets) for port, nets in conns.items()))


def sig_loose(conns):
    return tuple(sorted((port, tuple(canon(n) for n in nets))
                        for port, nets in conns.items()))


def collect_primitives(module, types):
    tbl = bit_table(module)
    out = defaultdict(list)
    for name, cell in module.get("cells", {}).items():
        t = cell.get("type")
        if types and t not in types:
            continue
        conns = port_nets(cell, tbl)
        out[(t, cell_hid(cell))].append({
            "name": name,
            "params": tuple(sorted(cell.get("parameters", {}).items())),
            "conns": conns,
            "strict": sig_strict(conns),
            "loose": sig_loose(conns),
            "bel": cell_bel(cell),
        })
    return out


def type_histogram(data, modules=None):
    hist = Counter()
    mods = data.get("modules", {})
    for name, mod in mods.items():
        if modules is not None and name not in modules:
            continue
        for cell in mod.get("cells", {}).values():
            hist[(cell.get("type"),
                  tuple(sorted(cell.get("parameters", {}).items())))] += 1
    return hist


def compare_primitives(ko_mod, ok_mod, types):
    ko = collect_primitives(ko_mod, types)
    ok = collect_primitives(ok_mod, types)
    keys = set(ko) | set(ok)
    report = {
        "only_ko": [], "only_ok": [],
        "strict_diff": [], "loose_diff": [],
        "moved": [], "param_diff": [],
    }
    for key in sorted(keys, key=str):
        a, b = ko.get(key, []), ok.get(key, [])
        if not a:
            report["only_ok"].append((key, len(b)))
            continue
        if not b:
            report["only_ko"].append((key, len(a)))
            continue
        a_strict = Counter(x["strict"] for x in a)
        b_strict = Counter(x["strict"] for x in b)
        a_loose = Counter(x["loose"] for x in a)
        b_loose = Counter(x["loose"] for x in b)
        a_params = Counter(x["params"] for x in a)
        b_params = Counter(x["params"] for x in b)
        if a_params != b_params:
            report["param_diff"].append((key, a_params, b_params))
        if a_loose != b_loose:
            report["loose_diff"].append((key, a_loose, b_loose))
        elif a_strict != b_strict:
            report["strict_diff"].append((key, a_strict, b_strict))
        # placement: match equal signatures pair-wise, keep equal counts
        a_bels = sorted(x["bel"] or "" for x in a)
        b_bels = sorted(x["bel"] or "" for x in b)
        if a_bels != b_bels:
            report["moved"].append((key, a_bels, b_bels))
    return report


def summarize_pair_counts(a, b):
    """multiset keys difference for (params, loose, strict)."""
    if a == b:
        return 0
    diff = 0
    for k in set(a) | set(b):
        diff += abs(a.get(k, 0) - b.get(k, 0))
    return diff


def ram_lane_groups(module):
    """RAM cells grouped by their shared (canonicalised) address buses.

    Returns {addr_signature: Counter of (canon(DI_lane), canon(DO_lane))}.
    A coherent build keeps DI[i] <-> DO[i] on one physical cell; comparing
    the pair multiset across builds exposes lane permutations.
    """
    tbl = bit_table(module)
    groups = defaultdict(Counter)
    for cell in module.get("cells", {}).values():
        if cell.get("type") not in RAM_TYPES:
            continue
        conns = port_nets(cell, tbl)
        addr = tuple(canon(n) for p in ("WCLK", "WAD", "RAD", "WRE")
                     for n in conns.get(p, ()))
        di, do = conns.get("DI", ()), conns.get("DO", ())
        for a, b in zip(di, do):
            groups[addr][(canon(a), canon(b))] += 1
    return groups


def compare_ram_lanes(ko_mod, ok_mod):
    ko, ok = ram_lane_groups(ko_mod), ram_lane_groups(ok_mod)
    only_ko = sorted(set(ko) - set(ok))
    only_ok = sorted(set(ok) - set(ko))
    changed = []
    for addr in sorted(set(ko) & set(ok)):
        if ko[addr] != ok[addr]:
            changed.append((addr, ko[addr], ok[addr]))
    return changed, only_ko, only_ok


def routing_map(module):
    out = {}
    for name, net in module.get("netnames", {}).items():
        out[name] = net.get("attributes", {}).get("ROUTING", "")
    return out


def compare_routing(ko_mod, ok_mod, types, max_examples=8):
    ko = collect_primitives(ko_mod, types)
    ok = collect_primitives(ok_mod, types)
    ko_routes = routing_map(ko_mod)
    ok_routes = routing_map(ok_mod)
    changed = Counter()
    examples = []
    for key in set(ko) & set(ok):
        for ca, cb in zip(sorted(ko[key], key=lambda x: x["name"]),
                          sorted(ok[key], key=lambda x: x["name"])):
            for port in set(ca["conns"]) & set(cb["conns"]):
                for na, nb in zip(ca["conns"][port], cb["conns"][port]):
                    ra, rb = ko_routes.get(na, ""), ok_routes.get(nb, "")
                    if ra != rb:
                        changed[ca.get("type") or key[0]] += 1
                        if len(examples) < max_examples:
                            examples.append((key, port, na, nb,
                                             ra[:60], rb[:60]))
    return changed, examples


def parse_yosys_log(path):
    passes, warns = [], Counter()
    if not path or not Path(path).exists():
        return None, passes, warns
    text = Path(path).read_text(errors="ignore")
    in_yosys = False
    for line in text.splitlines():
        if "--- Yosys Synthesis Command ---" in line:
            in_yosys = True
        if "--- nextpnr Command ---" in line:
            in_yosys = False
        m = PASS_RE.match(line)
        if m:
            passes.append(m.group(1).strip())
        m = WARN_RE.search(line)
        if m:
            warns[(Path(path).name, m.group(1)[:140])] += 1
    return path, passes, warns


def find_log(build_dir, name):
    candidates = [build_dir / "build.log"]
    candidates += sorted(build_dir.glob("*build.log"))
    candidates += sorted(build_dir.parent.glob(f"{name}*build.log"))
    for p in candidates:
        if p.exists():
            return p
    return None


def fmt_key(key):
    t, hid = key
    hid = hid.split("/")[-1] if hid else "?"
    return f"{t} {hid}"


def cell_examples(a, b, max_examples=5):
    """Concrete port diffs for two connection signatures."""
    out = []
    pa = dict((p, n) for p, n in a)
    pb = dict((p, n) for p, n in b)
    for port in sorted(set(pa) | set(pb)):
        na, nb = pa.get(port), pb.get(port)
        if na != nb:
            out.append((port, na, nb))
            if len(out) >= max_examples:
                break
    return out


def analyze(ko_dir, ok_dir, labels=None):
    ko_label = (labels or ("KO", "OK"))[0]
    ok_label = (labels or ("KO", "OK"))[1]
    metrics = {}
    md = [f"# Hardware build diff: {ko_label} (KO) vs {ok_label} (OK)",
          ""]
    for d, label in ((ko_dir, ko_label), (ok_dir, ok_label)):
        fs = sorted(p.name for p in Path(d).glob("*") if p.is_file())
        md.append(f"- **{label}** `{d}`: {', '.join(fs)}")
    md.append("")

    # ---- synth / pnr primitives ---------------------------------------
    for kind in ("synth", "pnr"):
        ko_path = Path(ko_dir) / f"{kind}.json"
        ok_path = Path(ok_dir) / f"{kind}.json"
        if not (ko_path.exists() and ok_path.exists()):
            md.append(f"## {kind.upper()} — missing file, skipped\n")
            continue
        ko_name, ko_mod = pick_module(load_json(ko_path))
        ok_name, ok_mod = pick_module(load_json(ok_path))
        ko_types = type_histogram(load_json(ko_path))
        ok_types = type_histogram(load_json(ok_path))
        md.append(f"## {kind.upper()} (`{ko_name}` vs `{ok_name}`)")
        md.append("")
        md.append("| type | KO | OK | diff |")
        md.append("|---|---:|---:|---:|")
        type_totals = Counter()
        for (t, _p), n in ko_types.items():
            type_totals[t] += n
        ok_totals = Counter()
        for (t, _p), n in ok_types.items():
            ok_totals[t] += n
        for t in sorted(set(type_totals) | set(ok_totals)):
            n_ko, n_ok = type_totals.get(t, 0), ok_totals.get(t, 0)
            if n_ko != n_ok:
                md.append(f"| {t} | {n_ko} | {n_ok} | {n_ok - n_ko:+d} |")
        md.append("")
        # parameter diffs (same type, different params)
        param_only_ko = ko_types - ok_types
        param_only_ok = ok_types - ko_types
        n_param = sum(param_only_ko.values()) + sum(param_only_ok.values())
        md.append(f"Parameter-set histogram cells only in one build: "
                  f"**{n_param}**")
        for k, v in param_only_ko.most_common(5):
            if ok_totals.get(k[0], 0):
                md.append(f"- KO only: `{k[0]}` {dict(k[1])} x{v}")
        for k, v in param_only_ok.most_common(5):
            if type_totals.get(k[0], 0):
                md.append(f"- OK only: `{k[0]}` {dict(k[1])} x{v}")
        md.append("")

        prim = compare_primitives(ko_mod, ok_mod, ANCHOR_TYPES)
        metrics[f"{kind}_anchor_params"] = len(prim["param_diff"])
        metrics[f"{kind}_anchor_structural"] = len(prim["loose_diff"])
        metrics[f"{kind}_anchor_moved"] = len(prim["moved"])
        md.append("### Anchor cells (stable `hdlname`: multipliers)")
        md.append("")
        md.append(f"- anchors KO: **{sum(len(v) for v in collect_primitives(ko_mod, ANCHOR_TYPES).values())}**, "
                  f"OK: **{sum(len(v) for v in collect_primitives(ok_mod, ANCHOR_TYPES).values())}**")
        md.append(f"- keys only in KO: **{len(prim['only_ko'])}** "
                  f"({', '.join(fmt_key(k) for k, _ in prim['only_ko'][:4])})")
        md.append(f"- keys only in OK: **{len(prim['only_ok'])}** "
                  f"({', '.join(fmt_key(k) for k, _ in prim['only_ok'][:4])})")
        md.append(f"- parameter differences: **{len(prim['param_diff'])}**")
        md.append(f"- structural connection diffs (after digit "
                  f"canonicalisation): **{len(prim['loose_diff'])}**")
        md.append(f"- rename-only connection diffs: "
                  f"**{len(prim['strict_diff'])}**")
        md.append(f"- placement (BEL) differences on anchors: "
                  f"**{len(prim['moved'])}**")
        md.append("")
        for key, a, b in prim["loose_diff"][:4]:
            md.append(f"- **structural** `{fmt_key(key)}`")
            a_sig = next(iter(a))
            b_sig = next(iter(b))
            for port, na, nb in cell_examples(a_sig, b_sig):
                md.append(f"    - `{port}`: KO {na} vs OK {nb}")
        for key, _, _ in prim["param_diff"][:4]:
            md.append(f"- **params** `{fmt_key(key)}`")
        for key, a_bels, b_bels in prim["moved"][:8]:
            md.append(f"- **moved** `{fmt_key(key)}`: KO {a_bels} vs OK {b_bels}")
        md.append("")

        if kind == "pnr":
            changed, only_ko, only_ok = compare_ram_lanes(ko_mod, ok_mod)
            metrics["ram_changed"] = len(changed)
            metrics["ram_only_ko"] = len(only_ko)
            metrics["ram_only_ok"] = len(only_ok)
            md.append("### RAM lane coherence")
            md.append("")
            md.append(f"- shared address groups with a different "
                      f"DI->DO lane multiset: **{len(changed)}**")
            md.append(f"- address groups only in KO: **{len(only_ko)}**, "
                      f"only in OK: **{len(only_ok)}**")
            for addr, a, b in changed[:6]:
                short = "/".join(n.split(".")[-1] for n in addr[:2])
                md.append(f"- group `{short}`")
                only_a = set(a) - set(b)
                only_b = set(b) - set(a)
                for pair in list(only_a)[:2]:
                    md.append(f"    - KO only: DI {pair[0]} -> DO {pair[1]}")
                for pair in list(only_b)[:2]:
                    md.append(f"    - OK only: DI {pair[0]} -> DO {pair[1]}")
            md.append("")
            changed, examples = compare_routing(
                ko_mod, ok_mod, ANCHOR_TYPES, max_examples=3)
            md.append("### Anchor routing (informational: placement moved)")
            md.append("")
            for t, n in changed.most_common():
                md.append(f"- {t}: **{n}** nets re-routed")
            for key, port, na, nb, ra, rb in examples:
                md.append(f"- `{fmt_key(key)}.{port}` {na} -> {nb}")
                md.append(f"    - KO route: `{ra}`")
                md.append(f"    - OK route: `{rb}`")
            md.append("")

    # ---- Yosys logs ----------------------------------------------------
    ko_log = find_log(Path(ko_dir), Path(ko_dir).name)
    ok_log = find_log(Path(ok_dir), Path(ok_dir).name)
    _, ko_passes, ko_warns = parse_yosys_log(ko_log)
    _, ok_passes, ok_warns = parse_yosys_log(ok_log)
    md.append("## Yosys logs")
    md.append("")
    md.append(f"- logs: KO `{ko_log}` / OK `{ok_log}`")
    if ko_passes and ok_passes:
        same = ko_passes == ok_passes
        md.append(f"- pass sequence identical: **{same}** "
                  f"({len(ko_passes)} vs {len(ok_passes)} passes)")
        if not same:
            for i, (a, b) in enumerate(zip(ko_passes, ok_passes)):
                if a != b:
                    md.append(f"    - pass {i}: KO `{a}` vs OK `{b}`")
                    break
    ko_w_tot = sum(ko_warns.values())
    ok_w_tot = sum(ok_warns.values())
    md.append(f"- warnings: KO **{ko_w_tot}** / OK **{ok_w_tot}**")
    for w, n in (ko_warns - ok_warns).most_common(5):
        md.append(f"    - KO only x{n}: {w[1][:120]}")
    for w, n in (ok_warns - ko_warns).most_common(5):
        md.append(f"    - OK only x{n}: {w[1][:120]}")
    md.append("")
    metrics["yosys_pass_same"] = bool(
        ko_passes and ok_passes and ko_passes == ok_passes)
    metrics["warnings_ko"] = ko_w_tot
    metrics["warnings_ok"] = ok_w_tot
    return "\n".join(md), metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ko", type=Path, help="known-bad build directory")
    ap.add_argument("--ok", type=Path, help="known-good build directory")
    ap.add_argument("--label-ko", default=None)
    ap.add_argument("--label-ok", default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the report to this markdown file")
    ap.add_argument("--matrix", action="store_true",
                    help="run the default 4-pair matrix")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"matrix build root (default: {DEFAULT_ROOT})")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write one markdown report per pair")
    args = ap.parse_args()

    if args.matrix:
        out_dir = args.out_dir or (args.root / "diff")
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for ko_name, ok_name in MATRIX_PAIRS:
            ko_dir, ok_dir = args.root / ko_name, args.root / ok_name
            if not ko_dir.exists() or not ok_dir.exists():
                print(f"[skip] {ko_name} vs {ok_name}: missing directory",
                      file=sys.stderr)
                continue
            md, m = analyze(ko_dir, ok_dir, (ko_name, ok_name))
            report = out_dir / f"{ko_name}__vs__{ok_name}.md"
            report.write_text(md, encoding="utf-8")
            rows.append((ko_name, ok_name, m))
            print(f"[ok] {report}")
        summary = ["# Hardware JSON/log diff matrix", "",
                   "| KO | OK | anchors struct/params (synth, pnr) | "
                   "anchors moved (pnr) | RAM lane groups diff | "
                   "passes same | warnings KO/OK |",
                   "|---|---|---:|---:|---:|---|---|"]
        for ko_name, ok_name, m in rows:
            summary.append(
                f"| {ko_name} | {ok_name} | "
                f"{m.get('synth_anchor_structural', 0)}/"
                f"{m.get('synth_anchor_params', 0)}, "
                f"{m.get('pnr_anchor_structural', 0)}/"
                f"{m.get('pnr_anchor_params', 0)} | "
                f"{m.get('pnr_anchor_moved', 0)} | "
                f"{m.get('ram_changed', 0)} "
                f"(+{m.get('ram_only_ko', 0)}/-{m.get('ram_only_ok', 0)}) | "
                f"{m.get('yosys_pass_same')} | "
                f"{m.get('warnings_ko')}/{m.get('warnings_ok')} |")
        summary_path = out_dir / "summary.md"
        summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
        print(f"[ok] {summary_path}")
        return

    if not args.ko or not args.ok:
        ap.error("--ko and --ok are required without --matrix")
    md, _ = analyze(args.ko, args.ok,
                    (args.label_ko or args.ko.name,
                     args.label_ok or args.ok.name))
    if getattr(args, "out", None):
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"[ok] {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
