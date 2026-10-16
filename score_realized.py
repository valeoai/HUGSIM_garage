"""The realized score (TerraTransfer's closed-loop HD-Score) of whole runs, into a JSON cache.

See sim/utils/realized_score.py for what it measures.

    python score_realized.py --runs official436/nuscenes_uniad official436/nuscenes_vad --out realized.json

A run is an output folder under --out-root (``$HUGSIM_OUT`` by default), named as closed_loop.py
names them: ``<dataset>_<agent>``, possibly below a folder of its own (``official436/nuscenes_uniad``,
say: only the last part names the dataset). Each of its episodes is scored once and kept in --out, which is
updated in place after every episode, so an interrupted pass resumes where it stopped. An episode
whose AD side crashed (agents/client.py's agent_crash.txt, or Pictura's pictura_crash.json) is
recorded as ``{"crashed": true, ...}`` rather than scored, and a cached crash is scored again on
the next pass, since the scene may have been re-run into that folder since.
"""

import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sim.utils.realized_score import DATASETS, find_scene_dir, read_crash_marker, score_episode  # noqa: E402


def save(cache, path):
    with open(path + ".tmp", "w") as f:
        json.dump(cache, f)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="output folders under --out-root, e.g. nuscenes_uniad")
    ap.add_argument("--out-root", default=os.environ.get("HUGSIM_OUT"),
                    help="the folder the runs are in (default: $HUGSIM_OUT)")
    ap.add_argument("--data-root", default=os.environ.get("HUGSIM_DATA"),
                    help="the data tree holding HUGSIM/ss/scenes (default: $HUGSIM_DATA)")
    ap.add_argument("--out", required=True, help="JSON cache of per-episode scores (updated in place)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these episode folder names")
    args = ap.parse_args()
    for flag, value, var in (("--out-root", args.out_root, "HUGSIM_OUT"), ("--data-root", args.data_root, "HUGSIM_DATA")):
        if not value:
            raise SystemExit(f"pass {flag} or set {var}")

    cache = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    for run in args.runs:
        # closed_loop.py names a run <dataset>_<agent> under $HUGSIM_OUT; one written with
        # --output_dir carries no dataset, and its scenes are looked up in all of them.
        prefix = os.path.basename(run.rstrip("/")).split("_")[0]
        datasets = (prefix,) if prefix in DATASETS else DATASETS
        # data.pkl, not eval.json: HUGSIM writes it when the episode ends, before its own scorer
        # runs, and that scorer sometimes dies; everything this reads is there all the same.
        for ep in sorted(glob.glob(os.path.join(args.out_root, run, "*", "data.pkl"))):
            ep_dir = os.path.dirname(ep)
            name = os.path.basename(ep_dir)
            if args.only and name not in args.only:
                continue
            key = f"{run}/{name}"
            crash = read_crash_marker(ep_dir)
            if crash is not None:
                cache[key] = {"crashed": True, "crash_step": crash["step"], "error": crash["error"],
                              "marker": crash["marker"]}
                save(cache, args.out)
                continue
            if key in cache and not cache[key].get("crashed"):
                continue
            try:
                scene_dir = find_scene_dir(args.data_root, datasets, name.rsplit("_", 2)[0])
                cache[key] = score_episode(ep_dir, scene_dir)
            except Exception as exc:  # keep going; the error is kept in the cache
                cache[key] = {"error": repr(exc)}
                print(f"[error] {key}: {exc!r}", file=sys.stderr, flush=True)
            save(cache, args.out)
        print(f"{run}: done", flush=True)


if __name__ == "__main__":
    main()
