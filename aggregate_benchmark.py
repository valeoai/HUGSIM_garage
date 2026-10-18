"""Per-agent means of the three HD-Score variants over a benchmark set, from realized_score.json files.

    python aggregate_benchmark.py --out-root $HUGSIM_OUT --data $HUGSIM_DATA --set official436 --agents uniad vad ...

Every scenario of the set is accounted for:
  scored     realized_score.json with scores
  crashed    the agent crashed or refused ({"crashed": true}): counts as 0 in every mean
  too_short  fewer than two frames ({"too_short": true}, or a `null` file): counts as 0
  agent_died no realized_score.json, but the folder shows the agent died: run.log holds closed_loop's
             "the agent exited (code" or agent_crash.txt is there. Counts as 0, like crashed
  missing    no realized_score.json and no sign of an agent death (the episode failed on the
             simulator's side): left out of the means and listed, since it says nothing about the
             agent. A warning line is printed for every agent with any.
A crashed record carrying "gpu_crash": true is an agent that died of the GPU's memory on the last
attempt run_benchmark.py allowed: it counts as 0 like any crash and is reported on its own line.

Denominators. The three HD-Score columns (hdscore, realized, arclen) average every episode that is
not missing, scoring crashed, too_short and agent_died episodes as 0. For `hdscore` that differs
from get_score.py, which scores a crashed episode from its eval.json (a crash there is not
necessarily 0); here the crash is 0 because eval.json may not exist. The component columns (nc, dac,
ttc, c, rc, rc_arc) average the scored episodes only (a crash has no components). The n behind each
column is printed under the table.
`hdscore` (HUGSIM's own, from eval.json) can be null on a scored episode whose eval.json is absent;
such an episode is left out of the hdscore mean only, and counted.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

KEYS = ("hdscore", "hdscore_realized", "hdscore_realized_arclength")
COMPONENTS = ("nc", "dac", "ttc", "c", "rc", "rc_arclength")
DATASETS = ("kitti360", "nuscenes", "pandaset", "waymo")


def episode_name(path):
    txt = open(path).read()
    scene = re.search(r"^scene_name:\s*'?([^'\n]+?)'?\s*$", txt, re.M).group(1).strip()
    mode = re.search(r"^mode:\s*'?([^'\n]+?)'?\s*$", txt, re.M).group(1).strip()
    return f"{scene}_{mode}"


def agent_died(folder):
    """The episode folder shows its agent died: closed_loop's FifoAgent._gone message, or a crash note."""
    if os.path.isfile(os.path.join(folder, "agent_crash.txt")):
        return True
    try:
        with open(os.path.join(folder, "run.log"), errors="replace") as f:
            return "the agent exited (code" in f.read()
    except OSError:
        return False


def collect(args, agent):
    rows = defaultdict(list)  # dataset -> list of (episode, status, record)
    for ds in args.datasets:
        for sc in sorted(glob.glob(os.path.join(args.data, "HUGSIM", "ss", "scenarios", args.set, ds, "*.yaml"))):
            ep = episode_name(sc)
            path = os.path.join(args.out_root, args.set, f"{ds}_{agent}", ep, "realized_score.json")
            if not os.path.isfile(path):
                rows[ds].append((ep, "agent_died" if agent_died(os.path.dirname(path)) else "missing", None))
                continue
            try:
                rec = json.load(open(path))
            except ValueError:
                rec = None
            if not isinstance(rec, dict):  # `null` (closed_loop's old too-short record), or a torn file
                rec = {"too_short": True}
            status = "crashed" if rec.get("crashed") else "too_short" if rec.get("too_short") else "scored"
            rows[ds].append((ep, status, rec))
    return rows


def summarise(entries):
    out = {"n": len(entries)}
    for status in ("scored", "crashed", "too_short", "agent_died", "missing"):
        out[status] = sum(1 for _, s, _ in entries if s == status)
    out["gpu_crash"] = sum(1 for _, s, r in entries if s == "crashed" and r.get("gpu_crash"))
    counted = [(s, r) for _, s, r in entries if s != "missing"]
    for k in KEYS:
        vals = []
        for s, r in counted:
            if s != "scored":
                vals.append(0.0)
            elif r.get(k) is not None:
                vals.append(float(r[k]))
        out[k] = sum(vals) / len(vals) if vals else None
        out[f"{k}_n"] = len(vals)
    scored = [r for s, r in counted if s == "scored"]
    for k in COMPONENTS:
        vals = [float(r[k]) for r in scored if r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
        out[f"{k}_n"] = len(vals)
    return out


def fmt(x):
    return "   -  " if x is None else f"{x:6.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=os.environ.get("HUGSIM_OUT"))
    ap.add_argument("--data", default=os.environ.get("HUGSIM_DATA"))
    ap.add_argument("--set", default="official436")
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--json", help="write the full table here")
    args = ap.parse_args()

    table, notes = {}, []
    head = f"{'agent':9s} {'dataset':9s} {'n':>4s} {'scored':>6s} {'crash':>5s} {'short':>5s} {'died':>4s} {'miss':>4s}  " \
           f"{'hdscore':>7s} {'realized':>8s} {'arclen':>7s} | {'nc':>6s} {'dac':>6s} {'ttc':>6s} {'c':>6s} {'rc':>6s} {'rc_arc':>6s}"
    print(head)
    for agent in args.agents:
        rows = collect(args, agent)
        table[agent] = {}
        for ds in list(args.datasets) + ["all"]:
            entries = [e for d in args.datasets for e in rows[d]] if ds == "all" else rows[ds]
            s = summarise(entries)
            table[agent][ds] = s
            print(f"{agent:9s} {ds:9s} {s['n']:4d} {s['scored']:6d} {s['crashed']:5d} {s['too_short']:5d} {s['agent_died']:4d} {s['missing']:4d}  "
                  f"{fmt(s['hdscore']):>7s} {fmt(s['hdscore_realized']):>8s} {fmt(s['hdscore_realized_arclength']):>7s} | "
                  + " ".join(fmt(s[k]) for k in COMPONENTS))
        missing = [f"{d}/{ep}" for d in args.datasets for ep, st, _ in rows[d] if st == "missing"]
        table[agent]["missing_episodes"] = missing
        notes.append((agent, table[agent]["all"], missing))
    print("\nn behind each column (all datasets):")
    for agent, s, missing in notes:
        print(f"  {agent:9s} hdscore {s['hdscore_n']}, realized {s['hdscore_realized_n']}, arclen "
              f"{s['hdscore_realized_arclength_n']} | " + ", ".join(f"{k} {s[f'{k}_n']}" for k in COMPONENTS))
    for agent, s, missing in notes:
        if s["gpu_crash"]:
            print(f"note: {agent}: {s['gpu_crash']} episode(s) crashed of the GPU's memory on the last attempt (counted as 0)")
        if s["missing"] > 0:
            print(f"WARNING: {agent}: {s['missing']} episode(s) missing (no score, no sign of an agent death), "
                  f"left out of the means: " + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
    if args.json:
        json.dump(table, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
