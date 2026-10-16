"""Shared plumbing of the five agents: checkpoint loading and the batch-of-one forward.

Written for this port. ``load_checkpoint`` and the batching reproduce what NAVSIM's
``AbstractAgent.compute_trajectory`` and the GTRS agents' ``get_trajectory_from_agent_input`` did
for the HUGSIM clients, plus a device argument: the original ``compute_trajectory`` never moved
anything off the CPU.
"""

import os
import pickle

import torch


def load_checkpoint(path, name):
    """The ``state_dict`` of a Lightning checkpoint, on the CPU, or a message naming ``path``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{name}: checkpoint not found at {path}. Pass --ckpt-root or set HUGSIM_AGENT_CKPTS '
            f'to a folder holding it at that relative path.'
        )
    try:
        return torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    except pickle.UnpicklingError:  # a checkpoint that stores more than tensors and plain containers
        return torch.load(path, map_location='cpu', weights_only=False)['state_dict']


def to_batch(features, device):
    """Add the batch dimension to every tensor (and every list of tensors) and move it to ``device``."""
    batch = {}
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            batch[k] = [t.unsqueeze(0).to(device) for t in v]
    return batch


def model_device(module):
    return next(module.parameters()).device
