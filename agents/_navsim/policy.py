"""The five NAVSIM-family agents behind one ``step(obs, info) -> plan`` function.

Each agent reproduces, step for step, what its HUGSIM client script in hyzhou404/NAVSIM did:
``hugsim/dataparser.py`` + ``ltf_e2e.py`` (LTF), ``dynamo_e2e.py`` (DrivoR) and
``gtrs_e2e.py``/``ztrs_e2e.py``/``gtrs_aug_e2e.py``, each with its own ``parse_raw``. The three
GTRS-family clients and LTF share the ego-status quirk (velocity and acceleration rotated by the
steering angle, the driving command fixed to "straight"); DrivoR passes the longitudinal speed and
a bicycle-model lateral acceleration instead. What distinguishes an agent (its :class:`AgentSpec`) is
in ``agents/<name>/spec.py``; this module loads it by name, so deleting an agent's folder leaves the
others working. ``step`` is what ``run.py`` serves and ``tools/plans.py`` replays over recorded frames.
"""

import contextlib
import importlib
import os
import time
from dataclasses import dataclass
from typing import Callable

import torch

from . import AGENTS

def to_plan(poses, subsample):
    """NAVSIM poses (x forward, y left, heading) -> HUGSIM's ``(right, forward)``.

    ``subsample`` keeps every fifth 0.1 s pose (from the fourth) for the 4 s plans of GTRS & co, so
    the waypoints are 0.5 s apart as HUGSIM expects.
    """
    imu_way_points = poses[:, :3]
    way_points = imu_way_points[:, [1, 0, 2]]
    way_points[:, 0] *= -1
    if subsample:
        way_points = way_points[4::5]
    return way_points


@dataclass
class AgentSpec:
    name: str
    files: dict  # role -> path relative to the checkpoint root
    build: Callable  # (paths) -> torch module with load_checkpoint
    parse: Callable
    compute: str  # the method of the module that turns an AgentInput into a Trajectory
    subsample: bool
    network: Callable  # the module whose output holds the discrete choice
    choice: Callable  # that output -> the discrete choices (proposal / vocabulary indices)
    # The original client ran this agent on the CPU (it never moved the model to the GPU). On a
    # GPU, cuDNN's default TF32 convolutions then move its plans by up to 2.5 cm on scene-0383
    # against 2e-5 m with TF32 off, so the step turns TF32 off for itself (see Policy.step).
    fp32: bool = False


def load_spec(name):
    """The :class:`AgentSpec` of ``agents/<name>/spec.py``; only the NAVSIM-family names are accepted."""
    if name not in AGENTS:
        raise KeyError(f'unknown NAVSIM-family agent {name!r}; expected one of {", ".join(AGENTS)}')
    return importlib.import_module(f'agents.{name}.spec').SPEC


def resolve_device(device=None):
    """``device`` or ``$HUGSIM_AGENT_DEVICE`` or ``cuda``; ``cuda`` must be available."""
    device = device or os.environ.get('HUGSIM_AGENT_DEVICE') or 'cuda'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f'device {device} requested but no CUDA device is visible; pass --device cpu '
            f'(or set HUGSIM_AGENT_DEVICE=cpu) to run on the CPU'
        )
    return torch.device(device)


def ckpt_paths(name, ckpt_root=None):
    """The files the agent needs under the checkpoint root, or an error naming the first missing one."""
    root = ckpt_root or os.environ.get('HUGSIM_AGENT_CKPTS')
    if not root:
        raise RuntimeError('no checkpoint root: pass --ckpt-root or set HUGSIM_AGENT_CKPTS')
    paths = {role: os.path.join(root, rel) for role, rel in load_spec(name).files.items()}
    for role, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{name}: expected {role} file {path} (checkpoint root {root}), not found')
    return paths


@contextlib.contextmanager
def full_fp32(enabled):
    """TF32 off for cuDNN and matmuls inside the block, and the caller's flags back after it.

    Scoped to the step rather than set at import: in closed_loop.py's --in_process mode the agent
    shares the simulator's process, whose own flags must not change.
    """
    if not enabled:
        yield
        return
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


class Policy:
    """An agent on a device: ``step(obs, info)`` is the function ``serve`` calls.

    ``seed``, when given, seeds torch before each step with ``seed + step index``. The original
    clients never seeded; this only matters for DrivoR, whose CEM refinement samples noise, and
    makes its plans repeatable (``tools/plans.py`` uses it).
    """

    def __init__(self, name, ckpt_root=None, device=None, seed=None):
        self.spec = load_spec(name)
        self.device = resolve_device(device)
        paths = ckpt_paths(name, ckpt_root)
        self.model = self.spec.build(paths).to(self.device).eval()
        self.seed = seed
        self.count = 0
        self.last_raw_imgs = None
        self.last_way_points = None
        self.last_choice = None
        self.spec.network(self.model).register_forward_hook(
            lambda module, args, output: setattr(self, 'last_choice', self.spec.choice(output))
        )

    def step(self, obs, info):
        agent_input, raw_imgs = self.spec.parse(obs, info)
        t = time.time()
        # A seeded step draws from a forked RNG: in closed_loop.py's --in_process mode the global
        # one is the simulator's, whose adversary and planner sample from it.
        devices = [self.device] if self.seed is not None and self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices, enabled=self.seed is not None):
            if self.seed is not None:
                torch.manual_seed(self.seed + self.count)
            with torch.no_grad(), full_fp32(self.spec.fp32 and self.device.type == 'cuda'):
                traj = getattr(self.model, self.spec.compute)(agent_input)
        way_points = to_plan(traj.poses, self.spec.subsample)
        print(f'[{self.spec.name}] step {self.count}: {len(way_points)} waypoints in {time.time() - t:.2f} s', flush=True)
        self.count += 1
        self.last_raw_imgs = raw_imgs
        self.last_way_points = way_points
        return way_points[:, :2]
