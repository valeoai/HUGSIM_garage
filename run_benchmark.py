"""Run agents over a benchmark set, several episodes at once per GPU, and keep going until done.

    python run_benchmark.py --agents uniad vad ltf --set official436 --slots 0:6 1:2

Each job is one closed_loop.py episode (agent x scenario of $HUGSIM_DATA/HUGSIM/ss/scenarios/<set>/),
run with --outputs scoring --realized_score, so its folder holds data.pkl, eval.json,
realized_score.json and the cropped clouds. A job whose folder already holds realized_score.json
is skipped, so a run can be stopped and restarted. Agents listed in agents/registry.py run
--in_process; the others over the pipes, on the same GPU. A job past --timeout is killed with its
whole process group and retried once. Episodes land in $HUGSIM_OUT/<set>/<dataset>_<agent>/, and
aggregate_benchmark.py turns them into a table of the three HD-Score variants.

Measured on toro (two H200s shared with other jobs): a NAVSIM-family episode takes 1 to 4 minutes,
a UniAD or VAD one 7 to 9, with about 3.5 GB and 6 GB of GPU memory each.
"""

import argparse
import glob
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
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
    return os.path.isfile(os.path.join(job["episode"], "realized_score.json"))


def command(job, args):
    ds = job["dataset"]
    cmd = [args.python, "closed_loop.py", "--scenario_path", job["scenario"],
           "--base_path", f"configs/sim/{ds}_base.yaml", "--camera_path", f"configs/sim/{ds}_camera.yaml",
           "--kinematic_path", "configs/sim/kinematic.yaml", "--ad", job["agent"], "--ad_cuda", "0",
           "--outputs", "scoring", "--realized_score"] + args.extra
    if job["agent"] in IN_PROCESS and not args.no_in_process:
        cmd.append("--in_process")
    return cmd


def run(job, gpu, args, log):
    os.makedirs(job["episode"], exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=f"{REPO}/sim:{REPO}")
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
    subprocess.run(["pkill", "-9", "-f", f"{job['episode']}( |$)"], check=False)
    status = "ok" if done(job) else f"fail({rc})"
    log(f"{status:12s} {time.time() - t0:7.0f}s gpu{gpu} {job['agent']:8s} {os.path.relpath(job['episode'], args.out)}")
    return done(job)


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

    todo = [j for j in jobs(args) if not done(j)]
    random.Random(0).shuffle(todo)  # mix agents and scenes, so the GPUs see a steady load
    print(f"{len(todo)} episodes to run: {dict(Counter(j['agent'] for j in todo))}", flush=True)
    if args.dry_run:
        for j in todo[:5]:
            print(" ".join(command(j, args)))
        return

    lock, queue, retried = threading.Lock(), list(todo), set()
    log_path = os.path.join(args.out, args.set, "run_benchmark.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(line):
        with lock:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{stamp} {line}", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{stamp} {line}\n")

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                job = queue.pop()
            if not run(job, gpu, args, log):
                with lock:
                    if job["episode"] not in retried:
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
