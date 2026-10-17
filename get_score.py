import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# -------------------------
# Config
# -------------------------
# Episodes are filed by closed_loop.py as $HUGSIM_OUT/<set>/<dataset>_<agent>/<scene>_<mode>.
# Score one set per run: `python get_score.py <set> <model>`, e.g. `official436 pictura_vector`
# (the --ad name). Only the episodes that are scenarios of the set count; see expected_episodes.
USAGE = "Usage: python get_score.py <set> <model>   (set: a folder under $HUGSIM_DATA/HUGSIM/ss/scenarios)"
if len(sys.argv) != 3:
    raise SystemExit(USAGE)
SCENARIO_SET, MODEL_NAME = sys.argv[1], sys.argv[2]
for _var in ("HUGSIM_OUT", "HUGSIM_DATA"):
    if _var not in os.environ:
        raise SystemExit(f"{_var} is not set (README.md, Environment)\n{USAGE}")
SCENARIO_ROOT = Path(os.environ["HUGSIM_DATA"]) / "HUGSIM" / "ss" / "scenarios"
if not (SCENARIO_ROOT / SCENARIO_SET).is_dir():
    sets = sorted(p.name for p in SCENARIO_ROOT.glob("*") if p.is_dir()) if SCENARIO_ROOT.is_dir() else []
    raise SystemExit(f"'{SCENARIO_SET}' is not a set under {SCENARIO_ROOT} (found: {sets})\n{USAGE}")
BENCH_ROOT = Path(os.environ["HUGSIM_OUT"]) / SCENARIO_SET
BASE_MODEL_NAME = MODEL_NAME

DATASET_ORDER = ["kitti360", "nuscenes", "pandaset", "waymo"]
DIFF_ORDER = ["easy", "medium", "hard", "extreme"]
DIFF_RE = re.compile(r"(easy|medium|hard|extreme)", re.IGNORECASE)

DISPLAY_NAMES = {
    "kitti360": "KITTI360",
    "nuscenes": "nuScenes",
    "pandaset": "Pandaset",
    "waymo": "Waymo",
}

# Canonical score order and key mapping.
SCORE_KEYS = [
    ("nc", "NC"),
    ("dac", "DAC"),
    ("ttc", "TTC"),
    ("c", "COM"),
    ("rc", "RC"),
    # ("pdms", "PDMS"),
    ("hdscore", "HDSCORE"),
]


# -------------------------
# Helpers
# -------------------------
def avg_dict(acc):
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) > 0}


def ensure_floatable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def to_float_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt(v):
    return "" if v is None or np.isnan(v) else f"{v:.4f}"


def expected_episodes(ds):
    """The episode folder names ``{scene_name}_{mode}`` of the set's scenarios of ``ds``, as
    closed_loop.py names them."""
    names = set()
    for path in sorted((SCENARIO_ROOT / SCENARIO_SET / ds).glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        names.add(f"{cfg['scene_name']}_{cfg['mode']}")
    return names


def discover_dataset_roots(model_name):
    roots = {}
    for ds in DATASET_ORDER:
        candidate = BENCH_ROOT / f"{ds}_{model_name}"
        if candidate.is_dir():
            roots[ds] = [candidate]
    return roots


def ds_root_of(roots):
    return roots[0]


def aggregate_from_roots(dataset_roots):
    # -------------------------
    # Accumulators
    # -------------------------
    global_totals = defaultdict(list)
    global_details = defaultdict(lambda: defaultdict(list))

    per_dataset_totals = defaultdict(lambda: defaultdict(list))
    per_dataset_details = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    per_diff_totals = defaultdict(lambda: defaultdict(list))
    per_diff_details = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    per_dataset_diff_totals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_dataset_diff_details = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    missing_per_dataset = defaultdict(list)
    ignored_per_dataset = defaultdict(list)
    present_per_dataset = defaultdict(list)
    all_eval_files = []

    # -------------------------
    # Aggregate data
    # -------------------------
    for ds, roots in dataset_roots.items():
        scene_dirs = []
        for ds_root in roots:
            scene_dirs.extend([p for p in ds_root.iterdir() if p.is_dir()])

        expected = expected_episodes(ds)
        by_name = {p.name: p for p in scene_dirs}
        ignored_per_dataset[ds] = [p for n, p in sorted(by_name.items()) if n not in expected]
        # A scenario never run has no folder at all; it counts as missing like one without eval.json.
        scene_dirs = [by_name.get(n, ds_root_of(roots) / n) for n in sorted(expected)]

        for scene_dir in scene_dirs:
            eval_fp = scene_dir / "eval.json"
            diff_match = DIFF_RE.search(scene_dir.name)
            diff = diff_match.group(1).lower() if diff_match else "unknown"

            if not eval_fp.exists():
                missing_per_dataset[ds].append(scene_dir)
                continue

            present_per_dataset[ds].append(scene_dir)
            all_eval_files.append(eval_fp)

            data = json.loads(eval_fp.read_text())
            top = ensure_floatable(data)

            # Global
            for k, v in top.items():
                global_totals[k].append(v)
            for th, subs in data.get("details", {}).items():
                for k, v in subs.items():
                    fv = to_float_or_none(v)
                    if fv is not None:
                        global_details[th][k].append(fv)

            # Per-dataset
            for k, v in top.items():
                per_dataset_totals[ds][k].append(v)
            for th, subs in data.get("details", {}).items():
                for k, v in subs.items():
                    fv = to_float_or_none(v)
                    if fv is not None:
                        per_dataset_details[ds][th][k].append(fv)

            # Per-difficulty
            for k, v in top.items():
                per_diff_totals[diff][k].append(v)
            for th, subs in data.get("details", {}).items():
                for k, v in subs.items():
                    fv = to_float_or_none(v)
                    if fv is not None:
                        per_diff_details[diff][th][k].append(fv)

            # Per (dataset, difficulty)
            for k, v in top.items():
                per_dataset_diff_totals[ds][diff][k].append(v)
            for th, subs in data.get("details", {}).items():
                for k, v in subs.items():
                    fv = to_float_or_none(v)
                    if fv is not None:
                        per_dataset_diff_details[ds][diff][th][k].append(fv)

    # -------------------------
    # Compute means
    # -------------------------
    global_means = avg_dict(global_totals)
    global_details_mean = {th: avg_dict(sub) for th, sub in global_details.items()}
    per_dataset_means = {ds: avg_dict(tot) for ds, tot in per_dataset_totals.items()}
    per_dataset_details_mean = {
        ds: {th: avg_dict(sub) for th, sub in details.items()} for ds, details in per_dataset_details.items()
    }
    per_diff_means = {diff: avg_dict(tot) for diff, tot in per_diff_totals.items()}
    per_diff_details_mean = {
        diff: {th: avg_dict(sub) for th, sub in details.items()} for diff, details in per_diff_details.items()
    }
    per_dataset_diff_means = {
        ds: {diff: avg_dict(tot) for diff, tot in diff_map.items()} for ds, diff_map in per_dataset_diff_totals.items()
    }
    per_dataset_diff_details_mean = {
        ds: {diff: {th: avg_dict(sub) for th, sub in th_map.items()} for diff, th_map in diff_map.items()}
        for ds, diff_map in per_dataset_diff_details.items()
    }

    return {
        "dataset_roots": dataset_roots,
        "eval_files": all_eval_files,
        "missing_per_dataset": missing_per_dataset,
        "ignored_per_dataset": ignored_per_dataset,
        "present_per_dataset": present_per_dataset,
        "averages": {
            "global": global_means,
            "per_dataset": per_dataset_means,
            "per_difficulty": per_diff_means,
            "per_dataset_difficulty": per_dataset_diff_means,
        },
        "details": {
            "global": global_details_mean,
            "per_dataset": per_dataset_details_mean,
            "per_difficulty": per_diff_details_mean,
            "per_dataset_difficulty": per_dataset_diff_details_mean,
        },
    }


def get_dataset_diff_mean(stats, ds, diff, key):
    try:
        return float(stats["averages"]["per_dataset_difficulty"][ds][diff][key])
    except Exception:
        return np.nan


def get_dataset_mean(stats, ds, key):
    try:
        return float(stats["averages"]["per_dataset"][ds][key])
    except Exception:
        return np.nan


def get_diff_mean(stats, diff, key):
    try:
        return float(stats["averages"]["per_difficulty"][diff][key])
    except Exception:
        return np.nan


def get_global_mean(stats, key):
    try:
        return float(stats["averages"]["global"][key])
    except Exception:
        return np.nan


def print_tables(section_name, stats):
    print(f"\n\n========== {section_name} ==========")
    print("Found dataset roots:")
    for ds in DATASET_ORDER:
        roots = stats["dataset_roots"].get(ds, [])
        if roots:
            for p in roots:
                print(f"  {ds}: {p}")

    print("\n=== PER (DATASET, DIFFICULTY) MEANS ===")
    header_dataset_diff = [f"{DISPLAY_NAMES.get(ds, ds)} {d.capitalize()}" for ds in DATASET_ORDER for d in DIFF_ORDER]
    print("\t".join(header_dataset_diff))

    for key, _ in SCORE_KEYS:
        row = [fmt(get_dataset_diff_mean(stats, ds, d, key)) for ds in DATASET_ORDER for d in DIFF_ORDER]
        print("\t".join(row))

    print("\n=== PER-DATASET MEANS ===")
    header_dataset = [DISPLAY_NAMES.get(ds, ds) for ds in DATASET_ORDER]
    print("\t".join(header_dataset))

    for key, _ in SCORE_KEYS:
        row = [fmt(get_dataset_mean(stats, ds, key)) for ds in DATASET_ORDER]
        print("\t".join(row))

    print("\n=== PER-DIFFICULTY MEANS ===")
    header_diff = [d.capitalize() for d in DIFF_ORDER]
    print("\t".join(header_diff))

    for key, _ in SCORE_KEYS:
        row = [fmt(get_diff_mean(stats, d, key)) for d in DIFF_ORDER]
        print("\t".join(row))

    print("\n=== GLOBAL MEAN ===")
    row = [fmt(get_global_mean(stats, key)) for key, _ in SCORE_KEYS]
    print("\n".join(row))

    print("\n=== Episodes of the set missing (no folder, or no eval.json), per dataset ===")
    total_missing = 0
    for ds in DATASET_ORDER:
        missing = stats["missing_per_dataset"].get(ds, [])
        total_missing += len(missing)
        print(f"\n[{ds}]  missing: {len(missing)}")
        for d in missing:
            print(f"  {d}")

    print("\n=== Folders not scenarios of this set (ignored) ===")
    total_ignored = 0
    for ds in DATASET_ORDER:
        ignored = stats["ignored_per_dataset"].get(ds, [])
        total_ignored += len(ignored)
        if ignored:
            print(f"\n[{ds}]  ignored: {len(ignored)}")
            for d in ignored:
                print(f"  {d}")

    print(f"\nTotal eval.json files found: {len(stats['eval_files'])}")
    print(f"Total missing episodes: {total_missing}")
    print(f"Total ignored folders: {total_ignored}")


def save_outputs(section_slug, stats):
    # Save numeric means as TSV.
    tsv_path = BENCH_ROOT / f"aggregate_eval_{BASE_MODEL_NAME}_{section_slug}_means_for_figure.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        # Section 1
        header_dataset_diff = [
            f"{DISPLAY_NAMES.get(ds, ds)} {d.capitalize()}" for ds in DATASET_ORDER for d in DIFF_ORDER
        ]
        f.write("## Per (dataset, difficulty)\n")
        f.write("\t".join(header_dataset_diff) + "\n")
        for key, _ in SCORE_KEYS:
            row = [fmt(get_dataset_diff_mean(stats, ds, d, key)) for ds in DATASET_ORDER for d in DIFF_ORDER]
            f.write("\t".join(row) + "\n")

        # Section 2
        f.write("\n## Per-dataset means\n")
        f.write("\t".join(DATASET_ORDER) + "\n")
        for key, _ in SCORE_KEYS:
            row = [fmt(get_dataset_mean(stats, ds, key)) for ds in DATASET_ORDER]
            f.write("\t".join(row) + "\n")

        # Section 3
        f.write("\n## Per-difficulty means\n")
        f.write("\t".join(DIFF_ORDER) + "\n")
        for key, _ in SCORE_KEYS:
            row = [fmt(get_diff_mean(stats, d, key)) for d in DIFF_ORDER]
            f.write("\t".join(row) + "\n")

        # Section 4
        f.write("\n## Global mean\n")
        row = [fmt(get_global_mean(stats, key)) for key, _ in SCORE_KEYS]
        f.write("\n".join(row) + "\n")

    print(f"\nSaved all numeric means to: {tsv_path}")

    # Save full JSON report.
    report = {
        "section": section_slug,
        "base_model": BASE_MODEL_NAME,
        "dataset_roots": {ds: [str(p) for p in stats["dataset_roots"].get(ds, [])] for ds in DATASET_ORDER},
        "averages": stats["averages"],
        "details": stats["details"],
        "ignored_folders": {
            ds: [str(p) for p in stats["ignored_per_dataset"].get(ds, [])] for ds in DATASET_ORDER
        },
        "missing_folders": {
            ds: [str(p) for p in stats["missing_per_dataset"].get(ds, [])] for ds in DATASET_ORDER
        },
        "present_folders": {
            ds: [str(p) for p in stats["present_per_dataset"].get(ds, [])] for ds in DATASET_ORDER
        },
        "eval_file_count": len(stats["eval_files"]),
    }

    out_path = BENCH_ROOT / f"aggregate_eval_{BASE_MODEL_NAME}_{section_slug}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved aggregate report to: {out_path}")


# -------------------------
# Main
# -------------------------
roots = discover_dataset_roots(BASE_MODEL_NAME)

if not roots:
    raise SystemExit(f"No dataset folders found for '{BASE_MODEL_NAME}' under {BENCH_ROOT}")

stats = aggregate_from_roots(roots)
print_tables(SCENARIO_SET.upper(), stats)
save_outputs(SCENARIO_SET, stats)
