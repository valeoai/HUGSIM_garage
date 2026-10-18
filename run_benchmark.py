"""Run agents over a benchmark set, several episodes at once per GPU, and keep going until done.

    python run_benchmark.py --agents uniad vad ltf --set official436 --slots 0:6 1:2

Each job is one closed_loop.py episode (agent x scenario of $HUGSIM_DATA/HUGSIM/ss/scenarios/<set>/),
run with --outputs scoring --realized_score, so its folder holds data.pkl, eval.json,
realized_score.json and the cropped clouds. A job whose folder already holds realized_score.json
is skipped, so a run can be stopped and restarted. Agents listed in agents/registry.py run
--in_process; the others over the pipes, on the same GPU. A job past --timeout is killed with its
whole process group and retried once. An episode folder that is a symlink (reused from another set)
is done if its target holds a realized_score.json, and is otherwise skipped, never run into.
Episodes land in $HUGSIM_OUT/<set>/<dataset>_<agent>/, and
aggregate_benchmark.py turns them into a table of the three HD-Score variants.

Measured on toro (two H200s shared with other jobs): a NAVSIM-family episode takes 1 to 4 minutes,
a UniAD or VAD one 7 to 9, with about 3.5 GB and 6 GB of GPU memory each.
"""

import argparse
import glob
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from agents.registry import IN_PROCESS  # noqa: E402  the agents that run in this environment
DATASETS = ("kitti360", "nuscenes", "pandaset", "waymo")


def episode_name(path):
    txt = open(path).read()
    scene = re.search(r"^scene_name:\s*'?([^'\n]+?)'?\s*$", txt, re.M).group(1).strip()
    mode = re.search(r"^mode:\s*'?([^'\n]+?)'?\s*$", txt, re.M).group(1).strip()
    return f"{scene}_{mode}"


def jobs(args):
    out = []
    for ds in args.datasets:
        scenarios = sorted(glob.glob(os.path.join(args.data, "HUGSIM", "ss", "scenarios", args.set, ds, "*.yaml")))
        for sc in scenarios[:args.limit] if args.limit else scenarios:
            for agent in args.agents:
                ep = os.path.join(args.out, args.set, f"{ds}_{agent}", episode_name(sc))
                out.append({"agent": agent, "dataset": ds, "scenario": sc, "episode": ep})
    return out


def done(job):
    """A realized_score.json is there (through a symlinked folder too): scored, crashed or too short."""
    return os.path.isfile(os.path.join(job["episode"], "realized_score.json"))


def linked(job, args):
    """The episode folder, or a folder above it under <out>/<set>, is a symlink (reused from another set)."""
    path = os.path.join(args.out, args.set)
    for part in os.path.relpath(job["episode"], path).split(os.sep):
        path = os.path.join(path, part)
        if os.path.islink(path):
            return True
    return False


def read_record(job):
    """The episode's realized_score.json as a dict, or None (absent, unreadable, or `null`)."""
    try:
        with open(os.path.join(job["episode"], "realized_score.json")) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def pre_toad_drivor(rec):
    """A `drivor` record from before agents/toad existed: it carries no "agent" tag, and was TOAD's drive.

    Until agents/drivor and agents/toad were separated, `--ad drivor` ran DrivoR with CEM refinement,
    which is now `toad`; closed_loop.py tags every record it writes with its --ad since. Such a record
    must not be taken for the new `drivor` (see README.md, *Results written before agents/toad*).
    """
    return isinstance(rec, dict) and "agent" not in rec


def refuse_pre_toad_drivor(folders, out, set_name):
    """Stop if any of the `drivor` episode folders holds an untagged record, with the rename to run."""
    if folders:
        dirs = sorted({os.path.dirname(f) for f in folders})
        raise SystemExit(
            f"{len(folders)} drivor record(s) under {', '.join(os.path.relpath(d, out) for d in dirs)} "
            "predate agents/toad and are TOAD's drive, not DrivoR's: rename each <dataset>_drivor folder "
            f"of {os.path.join(out, set_name)} to <dataset>_toad (and re-point any symlink to it), "
            "as README.md, *Results written before agents/toad*, describes")


def command(job, args):
    ds = job["dataset"]
    cmd = [args.python, "closed_loop.py", "--scenario_path", job["scenario"],
           "--base_path", f"configs/sim/{ds}_base.yaml", "--camera_path", f"configs/sim/{ds}_camera.yaml",
           "--kinematic_path", "configs/sim/kinematic.yaml", "--ad", job["agent"], "--ad_cuda", "0",
           "--outputs", "scoring", "--realized_score"] + args.extra
    if job["agent"] in IN_PROCESS and not args.no_in_process:
        cmd.append("--in_process")
    return cmd


#: Free GPU memory an episode needs to start, in MiB: the simulator, plus the agent's own process for
#: an agent behind a launcher (UniAD and VAD hold about 4.6 GB besides the simulator's ~6 GB).
NEED_MB = {"uniad": 16000, "vad": 16000}
DEFAULT_NEED_MB = 12000
#: Seconds a GPU is held after a launch, while the new episode loads and allocates: two workers
#: reading the same free memory would otherwise both start.
LOAD_HOLD_S = 45
GPU_LOCKS, GPU_LAST_START = {}, {}


#: Longest an episode waits for room on its GPU, in seconds, before it gives the GPU back and is requeued.
MEMORY_WAIT_MAX_S = 1800.0


def free_mb(gpu, log=None):
    """Free MiB on the GPU, or None when nvidia-smi cannot say (logged as an error, not as 0 MiB free)."""
    try:
        res = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            raise RuntimeError(f"nvidia-smi exit {res.returncode}: {res.stderr.strip() or res.stdout.strip()}")
        return int(res.stdout.strip().splitlines()[0])
    except Exception as e:
        (log or print)(f"ERROR      nvidia-smi on gpu{gpu} failed: {type(e).__name__}: {e}")
        return None


def wait_for_memory(job, gpu, log):
    """Wait until the GPU has room for this episode and its previous start is LOAD_HOLD_S old.

    One thread checks a GPU at a time (its lock is held only while checking); the start time it
    records makes the next one wait for the new episode to allocate before reading free memory.
    Gives up after MEMORY_WAIT_MAX_S, lock wait included: it releases the lock and returns False,
    and the caller requeues the job.
    """
    need = NEED_MB.get(job["agent"], DEFAULT_NEED_MB)
    lock = GPU_LOCKS.setdefault(str(gpu), threading.Lock())
    t0 = time.time()
    deadline = t0 + MEMORY_WAIT_MAX_S
    warned, last_err = False, 0.0
    if not lock.acquire(timeout=max(0.0, MEMORY_WAIT_MAX_S)):
        log(f"memory-wait timeout gpu{gpu} for {job['agent']}: could not get the GPU check lock in {int(MEMORY_WAIT_MAX_S)}s")
        return False
    try:
        while True:
            if time.time() >= deadline:
                log(f"memory-wait timeout gpu{gpu} for {job['agent']}: no {need} MiB free after {int(time.time() - t0)}s; requeued")
                return False
            hold = GPU_LAST_START.get(str(gpu), 0.0) + LOAD_HOLD_S - time.time()
            if hold > 0:
                time.sleep(min(hold, max(0.0, deadline - time.time())))
                continue
            # nvidia-smi errors are logged at most every 5 minutes, as errors
            free = free_mb(gpu, log if time.time() - last_err > 300 else (lambda _m: None))
            if free is None:
                last_err = time.time()
            elif free >= need:
                GPU_LAST_START[str(gpu)] = time.time()
                return True
            elif not warned and time.time() - t0 > 300:
                log(f"waiting    gpu{gpu} {free} MiB free, {need} needed, for {job['agent']} ({int(time.time() - t0)}s)")
                warned = True
            time.sleep(min(15, max(0.0, deadline - time.time())))
    finally:
        lock.release()


#: An agent or a simulator that dies of these did not fail the episode: the GPU had no room for it.
GPU_RESOURCE_ERRORS = ("out of memory", "cublas_status_alloc_failed", "cudnn_status_not_initialized",
                       "cusolver_status_internal_error")


def out_of_memory(job):
    for name in ("run.log", "output.txt", "agent_crash.txt"):
        try:
            with open(os.path.join(job["episode"], name), errors="replace") as f:
                text = f.read().lower()
        except OSError:
            continue
        if any(e in text for e in GPU_RESOURCE_ERRORS):
            return True
    return False


def gpu_crash(job, can_retry):
    """A crash record whose cause was the GPU's memory, not the agent: dropped, to be re-run.

    On a GPU shared with other jobs an agent can fail to allocate at start-up or mid-episode; the
    episode then ends as an agent crash and would be scored 0, which says nothing about the agent.
    The record is dropped only while a retry remains (`can_retry`); on the last attempt it is kept,
    marked "gpu_crash": true, so aggregate_benchmark.py reports it apart. Returns True if dropped.
    """
    rec = read_record(job)
    if rec is None or not rec.get("crashed") or rec.get("gpu_crash"):
        return False
    if not any(e in (rec.get("error") or "").lower() for e in GPU_RESOURCE_ERRORS):
        return False
    path = os.path.join(job["episode"], "realized_score.json")
    if can_retry:
        os.remove(path)
        return True
    rec["gpu_crash"] = True
    with open(path, "w") as f:
        json.dump(rec, f)
    return False


def child_env(gpu, args):
    """closed_loop.py reads its output and data trees from the environment, not from --out / --data."""
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                PYTHONPATH=f"{REPO}/sim:{REPO}", HUGSIM_OUT=os.path.abspath(args.out),
                HUGSIM_DATA=os.path.abspath(args.data))


def run(job, gpu, args, log, can_retry=lambda: True):
    """Run one episode. Returns (done, gpu_trouble, started): gpu_trouble when the GPU, not the agent, ended it."""
    if linked(job, args):  # reused from another set: never launch into it
        log(f"{'symlinked':12s} {'':8s}gpu{gpu} {job['agent']:8s} {os.path.relpath(job['episode'], args.out)} -> skipped")
        return done(job), False, False
    os.makedirs(job["episode"], exist_ok=True)
    env = child_env(gpu, args)
    if not wait_for_memory(job, gpu, log):
        return False, False, False
    t0 = time.time()
    with open(os.path.join(job["episode"], "run.log"), "w") as f:
        p = subprocess.Popen(command(job, args), cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = "timeout"
    try:  # whatever the episode left behind goes with it
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # An agent behind a launcher runs in a session of its own (sim/utils/launch_ad.py), which the
    # episode kills when it ends normally; one killed from outside (--timeout) leaves it blocked on
    # a pipe, holding its GPU memory, and attached to the folder a retry would reuse. Its command
    # line names the episode folder.
    subprocess.run(["pkill", "-9", "-f", f"{re.escape(job['episode'])}( |$)"], check=False)
    trouble = gpu_crash(job, can_retry())
    if trouble:
        status = "oom"
    else:
        status = "ok" if done(job) else ("oom" if out_of_memory(job) else f"fail({rc})")
        trouble = status == "oom"
        if done(job) and (read_record(job) or {}).get("gpu_crash"):
            status = "gpu_crash"  # the last attempt: kept, scored 0, reported apart
    log(f"{status:12s} {time.time() - t0:7.0f}s gpu{gpu} {job['agent']:8s} {os.path.relpath(job['episode'], args.out)}")
    return done(job), trouble, True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--set", default="official436")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--slots", nargs="+", default=["0:1"], help="gpu:episodes_at_once, e.g. 0:6 1:2")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per episode")
    ap.add_argument("--data", default=os.environ.get("HUGSIM_DATA"))
    ap.add_argument("--out", default=os.environ.get("HUGSIM_OUT"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--no-in-process", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N scenarios per dataset (a pilot)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="after --: more closed_loop.py arguments")
    args = ap.parse_args()
    if not args.data or not args.out:
        raise SystemExit("set HUGSIM_DATA and HUGSIM_OUT (or pass --data / --out)")

    all_jobs = jobs(args)
    refuse_pre_toad_drivor([j["episode"] for j in all_jobs if j["agent"] == "drivor" and pre_toad_drivor(read_record(j))],
                           args.out, args.set)
    # A crash record of the GPU's memory written before gpu_crash() existed (no "gpu_crash" mark) is
    # a failed attempt, not a score: drop it so the job is re-run. A symlinked folder is not ours to touch.
    swept = [] if args.dry_run else [j for j in all_jobs if not linked(j, args) and gpu_crash(j, True)]  # a dry run deletes nothing
    if swept:
        print(f"dropped {len(swept)} stale GPU-memory crash records: {dict(Counter(j['agent'] for j in swept))}", flush=True)
    skipped = [j for j in all_jobs if linked(j, args) and not done(j)]
    for j in skipped:
        print(f"symlinked, no realized_score.json at its target: skipped {os.path.relpath(j['episode'], args.out)}", flush=True)
    todo = [j for j in all_jobs if not done(j) and not linked(j, args)]
    random.Random(0).shuffle(todo)  # mix agents and scenes, so the GPUs see a steady load
    print(f"{len(todo)} episodes to run: {dict(Counter(j['agent'] for j in todo))}", flush=True)
    if args.dry_run:
        for j in todo[:5]:
            print(" ".join(command(j, args)))
        env = child_env("0", args)
        print({k: env[k] for k in ("HUGSIM_OUT", "HUGSIM_DATA", "CUDA_DEVICE_ORDER")})
        return

    lock, queue, retried, ooms = threading.Lock(), list(todo), set(), Counter()
    log_path = os.path.join(args.out, args.set, "run_benchmark.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(line):
        with lock:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{stamp} {line}", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{stamp} {line}\n")

    def can_retry(job):
        with lock:
            return ooms[job["episode"]] < 5 or job["episode"] not in retried

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                job = queue.pop()
            try:
                ok, trouble, started = run(job, gpu, args, log, lambda: can_retry(job))
            except Exception:
                log(f"EXCEPTION  gpu{gpu} {job['agent']} {os.path.relpath(job['episode'], args.out)}:\n{traceback.format_exc()}")
                ok, trouble, started = False, False, True  # a failed attempt
            if ok:
                continue
            with lock:
                if not started:
                    if not linked(job, args):  # no room on the GPU in time: the back of the queue, no attempt spent
                        queue.insert(0, job)
                # An out-of-memory start says nothing about the agent: it goes back in the queue
                # without spending the one retry (up to 5 times).
                elif trouble and ooms[job["episode"]] < 5:
                    ooms[job["episode"]] += 1
                    queue.insert(0, job)
                elif job["episode"] not in retried:
                    retried.add(job["episode"])
                    queue.insert(0, job)

    threads = []
    for spec in args.slots:
        gpu, n = spec.split(":")
        threads += [threading.Thread(target=worker, args=(gpu,)) for _ in range(int(n))]
    for t in threads:
        t.start()
        time.sleep(5)  # stagger the scene loads
    for t in threads:
        t.join()
    left = [j for j in todo if not done(j)]
    log(f"finished: {len(todo) - len(left)} done, {len(left)} not done")


if __name__ == "__main__":
    main()
