"""Lay out the HUGSIM release as one scene tree and two named benchmark sets.

Why two sets. The release (https://huggingface.co/datasets/XDimLab/HUGSIM) changed under the
people who report on it. On 2025-07-07 it held 49 scenes and a ``scenarios.zip`` of 345
scenarios, because the RealADSim @ ICCV 2025 competition held the rest back. Papers from that
period score those 345: the *pre-challenge protocol*. On 2025-10-31 the authors uploaded the 33
held-back scenes and replaced ``scenarios.zip`` with the current official release of 436
scenarios (nuScenes 88, PandaSet 127, KITTI-360 113, Waymo 108). The new zip retired 94 of the
345 and moved the actors in 15 of the rest, so it is a different benchmark, and nothing in the
files says which scenario belongs to which. The dataset's git history does, so both revisions
are pinned here:

* ``PRE_REVISION``  (2025-07-07): its ``scenarios.zip`` is the 345 scenarios of ``prechallenge345``.
* ``FULL_REVISION`` (2025-10-31): its scene zips are all 82 scenes (the 49 above and the 33 added
  later) and its ``scenarios.zip`` is the 436 scenarios of ``official436``.

The two sets overlap, and where they overlap the scenario file can differ, so each set holds its
own copy of every file. The scenes do not differ (a scene zip is never replaced between the
revisions), so there is one scene tree for both::

    <root>/HUGSIM/3DRealCar/<id>/{gs.pth,wlh.json}
    <root>/HUGSIM/nusc_map_cache/nusc_trainval/...
    <root>/HUGSIM/ss/scenes/<dataset>/<scene>/scene.pth ...
    <root>/HUGSIM/ss/scenarios/prechallenge345/<dataset>/<scenario>.yaml
    <root>/HUGSIM/ss/scenarios/official436/<dataset>/<scenario>.yaml

and ``HUGSIM_DATA=<root>``. ``closed_loop.py`` reads the set off the scenario's path, so an
episode lands in ``$HUGSIM_OUT/<set>/<dataset>_<agent>/`` and a scenario that is in both sets
never overwrites the other set's run.

The release does not run as shipped, so every scenario goes through three repairs:

* Every vehicle asset is written as ``<id>/postprocess/shadow.pth``, but the 3DRealCar release
  holds ``<id>/gs.pth`` and ``<id>/wlh.json``, and HUGSIM joins ``gs.pth`` onto the path
  itself (``sim/utils/plan.py``). It is reduced back to ``<id>``.
* Fifteen pandaset ``medium-02`` scenarios (in the official zip) pass ``max_t: 25`` to ``ConstantPlanner``, whose
  ``__init__`` takes no arguments. It is dropped.
* ``waymo/scene-140045460035-hard-00.yaml`` has ``start_velo: 1R``. It becomes ``1``.

Every scene's ``cfg.yaml`` also carries the authors' machine paths (``model_path:
/datasets_local/...``, ``source_path: /data3/hyzhou/...``). ``--from-hf`` rewrites them to the
install location (``localize_scene_cfg``). ``closed_loop.py`` overrides ``model_path`` in memory,
so a scene that was not localized still runs; ``--from-install`` leaves the old install's files
alone, and ``--verify`` only notes such scenes.

Two ways to build the tree:

``--from-hf``
    Download the pinned revisions into ``--root``: the scene zips (about 41 GB for all four
    datasets, unpacked and the zips removed one by one), the 3DRealCar vehicles and the trajdata
    nuScenes map cache.
``--from-install OLD_ROOT``
    A one-time migration of an install laid out as ``HUGSIM-public/`` + ``HUGSIM-private/`` (the layout
    this script replaces; nothing else reads it) without downloading data. The scenes, 3DRealCar and the map cache
    become symlinks into OLD_ROOT, which is only read. Only the two ``scenarios.zip`` files
    (about 200 KB each) are fetched, unless ``--scenarios-zip-pre`` / ``--scenarios-zip-official``
    name local copies, for a machine without access to Hugging Face.

Each step skips work that is already done, so an interrupted run can be restarted. Needs
``pyyaml``, and ``huggingface_hub`` unless both zips are local and ``--from-hf`` is not used.
Examples::

    python prepare_benchmark.py --root /data/hugsim --from-hf --dry-run
    python prepare_benchmark.py --root /data/hugsim --from-hf --datasets nuscenes
    python prepare_benchmark.py --root /data/hugsim --from-install /data/old_hugsim
    python prepare_benchmark.py --root /data/hugsim --verify
"""

import argparse
import collections
import copy
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

REPO_ID = "XDimLab/HUGSIM"
# 2025-07-07 "Upload 4 files": the last commit before the held-back scenes were published.
PRE_REVISION = "fb8ecd45b6d9377b70c59c2b03804b00a1259cb5"
# 2025-10-31 "Upload 4 files": the last commit of the full release, pinned rather than `main`
# so a later upload cannot silently change the sets.
FULL_REVISION = "38433e301bef99dca09f7bbd10f3145449fb66a2"

DATASETS = ("kitti360", "nuscenes", "pandaset", "waymo")
#: The set names are also read by closed_loop.py (SCENARIO_SETS), which files episodes by them.
SETS = ("prechallenge345", "official436")
REVISIONS = {"prechallenge345": PRE_REVISION, "official436": FULL_REVISION}
# What the pinned revisions contain. A mismatch means the layout was derived from something
# else, so it stops the run.
EXPECTED_SCENES = {"kitti360": 26, "nuscenes": 19, "pandaset": 15, "waymo": 22}
EXPECTED_SCENARIOS = {
    "prechallenge345": {"kitti360": 79, "nuscenes": 87, "pandaset": 51, "waymo": 128},
    "official436": {"kitti360": 113, "nuscenes": 88, "pandaset": 127, "waymo": 108},
}
EXPECTED_VEHICLES = 110
SHADOW_SUFFIX = "/postprocess/shadow.pth"


def hugsim_dir(root):
    return Path(root) / "HUGSIM"


def scene_dir(root, dataset):
    return hugsim_dir(root) / "ss" / "scenes" / dataset


def scenario_dir(root, set_name, dataset):
    return hugsim_dir(root) / "ss" / "scenarios" / set_name / dataset


def realcar_dir(root):
    return hugsim_dir(root) / "3DRealCar"


def map_cache_dir(root):
    return hugsim_dir(root) / "nusc_map_cache"


# ---------------------------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------------------------


def repair_cfg(cfg, name):
    """Apply the three repairs described in the module docstring, in place."""
    for actor in cfg["plan_list"]:
        asset = str(actor[5])
        if asset.endswith(SHADOW_SUFFIX):
            actor[5] = asset[: -len(SHADOW_SUFFIX)]
        elif "/" in asset:
            raise ValueError(f"{name}: unexpected vehicle asset path {asset!r}")
        if actor[6] == "ConstantPlanner" and actor[7]:
            actor[7] = {}
    velo = cfg["start_velo"]
    if isinstance(velo, str):
        match = re.fullmatch(r"(\d+(?:\.\d+)?)R", velo)
        if match is None:
            raise ValueError(f"{name}: unexpected start_velo {velo!r}")
        cfg["start_velo"] = int(match.group(1)) if "." not in match.group(1) else float(match.group(1))
    return cfg


def normalise(text, name):
    """Repair a scenario and re-serialise it, so the file runs as the tree does."""
    try:
        return yaml.safe_dump(repair_cfg(yaml.safe_load(text), name), sort_keys=False)
    except (ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as e:
        sys.exit(f"{name}: cannot repair this scenario ({type(e).__name__}: {e})")


def read_scenarios_zip(path):
    """{dataset: {filename: yaml text}} of a ``scenarios.zip``. The two revisions disagree on the
    inner prefix (``scenarios/<dataset>/<file>`` or ``<dataset>/<file>``), so the dataset is the
    path component that names one."""
    out = collections.defaultdict(dict)
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".yaml"):
                continue
            parts = name.strip("/").split("/")
            ds = next((p for p in parts[:-1] if p in DATASETS), None)
            if ds is None:
                sys.exit(f"{path}: cannot place {name} in a dataset")
            out[ds][parts[-1]] = zf.read(name).decode()
    return out


def check_counts(what, got, expected):
    counts = {ds: len(got.get(ds, ())) for ds in DATASETS}
    if counts != expected:
        sys.exit(f"{what}: expected {expected}, got {counts}. The HF history no longer matches this script.")


def hf_zip(cache, revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO_ID, "scenarios.zip", repo_type="dataset", revision=revision, cache_dir=cache)


def resolve_scenarios(args, cache):
    """{set: {dataset: {filename: repaired yaml text}}}, from the two zips (local or pinned)."""
    zips = {
        "prechallenge345": args.scenarios_zip_pre or hf_zip(cache, PRE_REVISION),
        "official436": args.scenarios_zip_official or hf_zip(cache, FULL_REVISION),
    }
    out = {}
    for set_name in SETS:
        raw = read_scenarios_zip(zips[set_name])
        check_counts(f"{set_name} scenarios ({zips[set_name]})", raw, EXPECTED_SCENARIOS[set_name])
        out[set_name] = {
            ds: {f: normalise(text, f"{set_name}/{ds}/{f}") for f, text in raw[ds].items()} for ds in DATASETS
        }
    return out


def install_scenarios(args, scenarios):
    for set_name in SETS:
        for ds in args.datasets:
            out = scenario_dir(args.root, set_name, ds)
            out.mkdir(parents=True, exist_ok=True)
            written = 0
            for fname, text in scenarios[set_name][ds].items():
                path = out / fname
                if path.exists() and path.read_text() == text:
                    continue
                path.write_text(text)
                written += 1
            print(f"[scenarios] {set_name}/{ds}: {len(scenarios[set_name][ds])} files ({written} written) -> {out}")


# ---------------------------------------------------------------------------------------------
# Scenes, vehicles, map cache: downloaded
# ---------------------------------------------------------------------------------------------


def list_files(api, revision):
    """{path: size} of every file in the repo at ``revision``."""
    from huggingface_hub.hf_api import RepoFile

    return {
        entry.path: entry.size
        for entry in api.list_repo_tree(REPO_ID, repo_type="dataset", revision=revision, recursive=True)
        if isinstance(entry, RepoFile)
    }


def scene_zips(files):
    """{dataset: {scene_name: (path, size)}} of the scene zips in a file listing."""
    out = collections.defaultdict(dict)
    for path, size in files.items():
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == "scenes" and parts[2].endswith(".zip"):
            out[parts[1]][parts[2][: -len(".zip")]] = (path, size)
    return out


def extract_scene(zip_path, dest):
    """Unpack a scene zip to ``dest``. The zips disagree on their inner prefix (``<scene>/``
    or ``<dataset>/<scene>/``), so move whichever folder holds ``scene.pth``."""
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".extract_") as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        found = [p.parent for p in Path(tmp).rglob("scene.pth")]
        if len(found) != 1:
            sys.exit(f"{zip_path}: expected one scene.pth, found {len(found)}")
        if dest.exists() and not (dest / "scene.pth").exists():
            # An incomplete folder would make the move nest the scene inside it.
            shutil.rmtree(dest)
        shutil.move(str(found[0]), str(dest))


def stale_paths(scene_path):
    """The author-machine paths still in ``<scene>/cfg.yaml``, as {key: value}."""
    text = (scene_path / "cfg.yaml").read_text()
    stale = {}
    match = re.search(r"^model_path: *(.*)$", text, flags=re.MULTILINE)
    if match and match.group(1) != str(scene_path):
        stale["model_path"] = match.group(1)
    match = re.search(r"^source_path: *(.*)$", text, flags=re.MULTILINE)
    if match and match.group(1) not in ("", "null"):
        stale["source_path"] = match.group(1)
    return stale


def localize_scene_cfg(scene_path):
    """Replace the authors' paths in ``<scene>/cfg.yaml``; returns whether anything changed.

    ``model_path`` becomes the installed scene folder. ``closed_loop.py`` overrides it in
    memory anyway, but the render/GUI tools and anyone reading the file take it as written.
    ``source_path`` is the authors' raw reconstruction input, which is not released, so it
    becomes null; its old value is kept as ``release_source_path`` because it records which
    raw sequence and frame range the scene was built from. Only those lines change.
    """
    cfg = scene_path / "cfg.yaml"
    stale = stale_paths(scene_path)
    if not stale:
        return False
    text = cfg.read_text()
    if "model_path" in stale:
        text = re.sub(r"^model_path:.*$", f"model_path: {scene_path}", text, count=1, flags=re.MULTILINE)
    if "source_path" in stale:
        text = re.sub(
            r"^source_path:.*$",
            f"source_path: null\nrelease_source_path: {stale['source_path']}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    cfg.write_text(text)
    return True


def install_scenes_hf(args, scene_zip_listing, cache):
    from huggingface_hub import hf_hub_download

    for ds in args.datasets:
        out = scene_dir(args.root, ds)
        out.mkdir(parents=True, exist_ok=True)
        # A killed extraction leaves its temp dir behind.
        for leftover in out.glob(".extract_*"):
            shutil.rmtree(leftover, ignore_errors=True)
        localized = 0
        linked = 0
        for scene, (path, _) in sorted(scene_zip_listing[ds].items()):
            if not (out / scene / "scene.pth").exists():
                print(f"[scene] {ds}/{scene}", flush=True)
                zip_path = hf_hub_download(REPO_ID, path, repo_type="dataset", revision=FULL_REVISION, cache_dir=cache)
                extract_scene(zip_path, out / scene)
                if not args.keep_zips:
                    # Remove the blob behind the snapshot symlink too, or the disk use doubles.
                    blob = Path(zip_path).resolve()
                    Path(zip_path).unlink()
                    blob.unlink(missing_ok=True)
            # Also run on scenes that were already there, so an older tree is fixed. Never through
            # a link: on a --from-install root that would rewrite the old install's files.
            if out.resolve() != out or (out / scene).is_symlink():
                linked += 1
                continue
            localized += localize_scene_cfg(out / scene)
        if localized:
            print(f"[scene] {ds}: replaced the authors' paths in {localized} cfg.yaml")
        if linked:
            print(f"[scene] {ds}: {linked} scenes are links into another install, cfg.yaml left as it is")


def vehicle_ids(root):
    """The 3DRealCar ids that have both ``gs.pth`` and ``wlh.json``."""
    root = Path(root)
    return {p.parent.name for p in root.glob("*/gs.pth")} & {p.parent.name for p in root.glob("*/wlh.json")}


def install_vehicles_hf(args, cache):
    from huggingface_hub import snapshot_download

    root = realcar_dir(args.root)
    # Both files of every vehicle, not just gs.pth: a run that died between them must be repaired.
    if len(vehicle_ids(root)) >= EXPECTED_VEHICLES:
        return
    print(f"[3DRealCar] -> {root}", flush=True)
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=FULL_REVISION,
        allow_patterns=["3DRealCar/*/gs.pth", "3DRealCar/*/wlh.json"],
        local_dir=hugsim_dir(args.root),
        cache_dir=cache,
    )


def install_map_cache_hf(args, cache):
    from huggingface_hub import hf_hub_download

    target = map_cache_dir(args.root) / "nusc_trainval"
    # Extract beside the target and rename, so an interrupted extraction is never taken for a
    # finished one (the target exists only once it is complete).
    for leftover in hugsim_dir(args.root).glob(".extract_*"):
        shutil.rmtree(leftover, ignore_errors=True)
    if not target.is_dir():
        print(f"[nusc_map_cache] -> {target}", flush=True)
        zip_path = hf_hub_download(
            REPO_ID, "nusc_map_cache.zip", repo_type="dataset", revision=FULL_REVISION, cache_dir=cache
        )
        hugsim_dir(args.root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=hugsim_dir(args.root), prefix=".extract_") as tmp:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            found = [p for p in Path(tmp).rglob("nusc_trainval") if p.is_dir()]
            if len(found) != 1:
                sys.exit(f"{zip_path}: expected one nusc_trainval folder, found {len(found)}")
            target.parent.mkdir(parents=True, exist_ok=True)
            found[0].rename(target)


# ---------------------------------------------------------------------------------------------
# Scenes, vehicles, map cache: linked from an old install
# ---------------------------------------------------------------------------------------------


def old_layout(old_root):
    """The pieces of a ``HUGSIM-public`` + ``HUGSIM-private`` install this script links to."""
    old = Path(old_root).expanduser().resolve()
    pub, priv = old / "HUGSIM-public", old / "HUGSIM-private"
    for need in (pub / "3DRealCar", pub / "ss" / "scenes", priv / "ss" / "scenes"):
        if not need.is_dir():
            sys.exit(f"{need} does not exist: --from-install takes a root holding HUGSIM-public/ and HUGSIM-private/")
    return pub, priv


def link(src, dst):
    """``dst`` -> ``src``, unless it is already that link. Returns True if it created one."""
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return False
        sys.exit(f"{dst} is a link to {os.readlink(dst)}, not {src}; remove it to relink")
    if dst.exists():
        sys.exit(f"{dst} exists and is not a link; remove it to link {src} there")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)
    return True


def install_links(args):
    pub, priv = old_layout(args.from_install)
    made = link(pub / "3DRealCar", realcar_dir(args.root))
    print(f"[3DRealCar] {'linked' if made else 'already linked'}: {realcar_dir(args.root)} -> {pub / '3DRealCar'}")
    if (pub / "nusc_map_cache").is_dir() and "nuscenes" in args.datasets:
        made = link(pub / "nusc_map_cache", map_cache_dir(args.root))
        print(f"[nusc_map_cache] {'linked' if made else 'already linked'}: {map_cache_dir(args.root)}")
    for ds in args.datasets:
        n = 0
        for tree in (pub, priv):
            src = tree / "ss" / "scenes" / ds
            for scene in sorted(p for p in src.glob("*") if (p / "scene.pth").exists()):
                n += link(scene, scene_dir(args.root, ds) / scene.name)
        print(f"[scenes] {ds}: {n} new links -> {scene_dir(args.root, ds)}")


def link_home_map_cache(args):
    """Point ``~/.unified_data_cache/nusc_trainval`` at the map cache, which is where HUGSIM's
    trajdata loader looks (``sim/utils/plan.py``). Only nuScenes scenarios with
    ``load_HD_map: true`` read it."""
    target = map_cache_dir(args.root) / "nusc_trainval"
    if not target.is_dir():
        return
    home = Path(os.path.expanduser("~/.unified_data_cache/nusc_trainval"))
    if home.is_symlink() and not home.exists():
        print(f"[nusc_map_cache] {home} was a dangling link to {os.readlink(home)}, replaced")
        home.unlink()
    if home.exists() or home.is_symlink():
        print(f"[nusc_map_cache] {home} already exists, left alone")
    else:
        home.parent.mkdir(parents=True, exist_ok=True)
        home.symlink_to(target)
        print(f"[nusc_map_cache] {home} -> {target}")


# ---------------------------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------------------------


def verify(args):
    """Check an installed tree: the counts of both sets, that every scenario's scene and vehicles
    exist and that none still needs a repair. Needs no network. Returns the number of problems."""
    problems = 0
    cars = {p.parent.name for p in realcar_dir(args.root).glob("*/gs.pth")}
    wlh = {p.parent.name for p in realcar_dir(args.root).glob("*/wlh.json")}
    bad = len(cars) < EXPECTED_VEHICLES or cars != wlh
    problems += bad
    print(f"{'BAD' if bad else 'OK '} 3DRealCar {len(cars)}/{EXPECTED_VEHICLES} vehicles")
    if "nuscenes" in args.datasets:
        bad = not (map_cache_dir(args.root) / "nusc_trainval").is_dir()
        problems += bad
        print(f"{'BAD' if bad else 'OK '} nusc_map_cache nusc_trainval {'missing' if bad else 'present'}")
    for ds in args.datasets:
        sdir = scene_dir(args.root, ds)
        have = {p.parent.name for p in sdir.glob("*/scene.pth")} if sdir.is_dir() else set()
        stale = sorted(s for s in have if (sdir / s / "cfg.yaml").exists() and stale_paths(sdir / s))
        bad = len(have) != EXPECTED_SCENES[ds]
        problems += bad
        print(f"{'BAD' if bad else 'OK '} scenes    {ds:8s} {len(have)}/{EXPECTED_SCENES[ds]}")
        if stale:
            print(f"      note: {len(stale)} scene cfg.yaml still hold the authors' paths (harmless: closed_loop.py overrides them)")
    for set_name in SETS:
        for ds in args.datasets:
            cdir = scenario_dir(args.root, set_name, ds)
            files = sorted(cdir.glob("*.yaml")) if cdir.is_dir() else []
            sdir = scene_dir(args.root, ds)
            issues = collections.defaultdict(list)
            for path in files:
                try:
                    cfg = yaml.safe_load(path.read_text())
                    scene_name, plan_list = str(cfg["scene_name"]), cfg["plan_list"]
                except (KeyError, TypeError, yaml.YAMLError) as e:
                    issues[f"unreadable scenario ({type(e).__name__}: {e})"].append(path.name)
                    continue
                if not (sdir / scene_name / "scene.pth").exists():
                    issues["scene not under ss/scenes"].append(path.name)
                for actor in plan_list:
                    if actor[6] != "UnicyclePlanner" and not (realcar_dir(args.root) / str(actor[5]) / "gs.pth").exists():
                        issues["vehicle not in 3DRealCar"].append(path.name)
                try:
                    if repair_cfg(copy.deepcopy(cfg), path.name) != cfg:
                        issues["still needs a repair"].append(path.name)
                except (ValueError, KeyError, TypeError, IndexError) as e:
                    issues[f"unrepairable ({type(e).__name__}: {e}) in {path}"].append(path.name)
            want = EXPECTED_SCENARIOS[set_name][ds]
            bad = len(files) != want or bool(issues)
            problems += bad
            print(f"{'BAD' if bad else 'OK '} {set_name:15s} {ds:8s} scenarios {len(files)}/{want}")
            for label, items in issues.items():
                print(f"      {label}: {sorted(set(items))[:6]}{' ...' if len(set(items)) > 6 else ''}")
    return problems


# ---------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="the tree HUGSIM_DATA will point at; HUGSIM/ is created in it")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--from-hf", action="store_true", help="download the pinned revisions into --root")
    source.add_argument("--from-install", metavar="OLD_ROOT", help="link an existing HUGSIM-public/-private install")
    parser.add_argument("--scenarios-zip-pre", help="local scenarios.zip of the pre-challenge revision (prechallenge345)")
    parser.add_argument("--scenarios-zip-official", help="local scenarios.zip of the full release (official436)")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--keep-zips", action="store_true", help="--from-hf: keep downloaded scene zips in the HF cache")
    parser.add_argument(
        "--no-home-link", action="store_true", help="do not link ~/.unified_data_cache/nusc_trainval to the map cache"
    )
    parser.add_argument("--dry-run", action="store_true", help="print what would be built and its size, change nothing")
    parser.add_argument("--verify", action="store_true", help="check an existing tree against the expected sets")
    args = parser.parse_args()
    args.root = Path(args.root).expanduser().resolve()
    if not (args.verify or args.from_hf or args.from_install):
        parser.error("say where the data comes from: --from-hf or --from-install OLD_ROOT (or --verify)")

    if args.verify:
        problems = verify(args)
        print("tree is complete" if not problems else f"{problems} problem(s)")
        sys.exit(1 if problems else 0)

    if args.dry_run or args.from_install:
        # Reading the two scenarios.zip files (~200 kB each) must not create the HF cache in --root.
        cache = Path(tempfile.mkdtemp(prefix="hugsim_sets_"))
        try:
            build(args, cache)
        finally:
            shutil.rmtree(cache, ignore_errors=True)
    else:
        args.root.mkdir(parents=True, exist_ok=True)
        build(args, args.root / ".hf_cache")


def build(args, cache):
    scenarios = resolve_scenarios(args, cache)
    print(f"{REPO_ID}: prechallenge345 = scenarios.zip at {PRE_REVISION[:10]}, official436 = scenarios.zip at {FULL_REVISION[:10]}")

    scene_zip_listing = None
    if args.from_hf:
        from huggingface_hub import HfApi

        api = HfApi()
        files = list_files(api, FULL_REVISION)
        scene_zip_listing = scene_zips(files)
        check_counts("scenes", scene_zip_listing, EXPECTED_SCENES)
        vehicles = {p.split("/")[1] for p in files if p.startswith("3DRealCar/") and p.endswith("/gs.pth")}
        for set_name in SETS:
            for ds in DATASETS:
                for fname, text in scenarios[set_name][ds].items():
                    cfg = yaml.safe_load(text)
                    if cfg["scene_name"] not in scene_zip_listing[ds]:
                        sys.exit(f"{set_name}/{ds}/{fname}: scene {cfg['scene_name']!r} has no zip in the release")
                    for actor in cfg["plan_list"]:
                        if actor[6] != "UnicyclePlanner" and str(actor[5]) not in vehicles:
                            sys.exit(f"{set_name}/{ds}/{fname}: vehicle {actor[5]!r} is not in 3DRealCar")
        for ds in DATASETS:
            gb = sum(size for _, size in scene_zip_listing[ds].values()) / 1e9
            print(
                f"  {ds:8s} {len(scene_zip_listing[ds]):3d} scenes ({gb:5.1f} GB zipped)"
                f"  scenarios: prechallenge345 {len(scenarios['prechallenge345'][ds]):3d}"
                f"  official436 {len(scenarios['official436'][ds]):3d}"
            )
        realcar_gb = sum(s for p, s in files.items() if p.startswith("3DRealCar/")) / 1e9
        print(f"  shared: 3DRealCar {realcar_gb:.1f} GB, nusc_map_cache {files['nusc_map_cache.zip'] / 1e9:.1f} GB")
        if args.dry_run:
            sel = sum(size for ds in args.datasets for _, size in scene_zip_listing[ds].values()) / 1e9
            extra = f"{realcar_gb:.1f} GB of 3DRealCar"
            if "nuscenes" in args.datasets:
                extra += f" and {files['nusc_map_cache.zip'] / 1e9:.1f} GB of nusc_map_cache"
            print(
                f"would download {sel:.1f} GB of scene zips for {', '.join(args.datasets)}, plus {extra}, "
                f"to {hugsim_dir(args.root)}"
            )
            return
    elif args.dry_run:
        pub, priv = old_layout(args.from_install)
        for ds in args.datasets:
            n = sum(1 for t in (pub, priv) for p in (t / "ss" / "scenes" / ds).glob("*") if (p / "scene.pth").exists())
            print(
                f"  {ds:8s} {n:3d} scenes to link"
                f"  scenarios: prechallenge345 {len(scenarios['prechallenge345'][ds]):3d}"
                f"  official436 {len(scenarios['official436'][ds]):3d}"
            )
        print(f"would link 3DRealCar, nusc_map_cache and the scenes from {args.from_install} into {hugsim_dir(args.root)}")
        return

    install_scenarios(args, scenarios)
    if args.from_hf:
        install_vehicles_hf(args, cache)
        if "nuscenes" in args.datasets:
            install_map_cache_hf(args, cache)
        install_scenes_hf(args, scene_zip_listing, cache)
    else:
        install_links(args)
    if "nuscenes" in args.datasets and not args.no_home_link:
        link_home_map_cache(args)

    print()
    print(f"export HUGSIM_DATA={args.root}")
    problems = verify(args)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
