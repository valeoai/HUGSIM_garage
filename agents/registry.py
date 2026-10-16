"""Agents that closed_loop.py can run in the simulator's process (``--in_process``).

Maps an agent name to ``(module, agent argument)``: the module defines ``make_step(agent,
episode_dir, device, ckpt_root=None) -> (step, on_done)``. Nothing heavy is imported here; the
module is imported by :func:`load_in_process`.
"""

import importlib

IN_PROCESS = {
    'ltf': ('agents.navsim.run', 'ltf'),
    'drivor': ('agents.navsim.run', 'drivor'),
    'gtrs': ('agents.navsim.run', 'gtrs'),
    'ztrs': ('agents.navsim.run', 'ztrs'),
    'gtrs_aug': ('agents.navsim.run', 'gtrs_aug'),
}


def load_in_process(name, episode_dir, device='cuda'):
    """``(step, on_done)`` of the named agent; ``on_done`` may be None."""
    if name not in IN_PROCESS:
        raise KeyError(f'no in-process agent {name!r}; supported: {", ".join(sorted(IN_PROCESS))}')
    module, agent = IN_PROCESS[name]
    return importlib.import_module(module).make_step(agent, episode_dir, device)
