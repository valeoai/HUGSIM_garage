"""The part of hydra's ``instantiate`` that loading a VaVAM checkpoint uses (added for the garage).

The checkpoint's ``vam_conf`` holds ``gpt_config`` and ``action_config``, each a mapping with a
``_target_`` class path (``vam.video_pretraining...``, ``vam.action_expert...``) and its keyword
arguments. This imports the class and calls it with them, nested configs instantiated first, which is
what ``hydra.utils.instantiate`` does for them, without making hydra a dependency of the agent.
"""

import importlib

from omegaconf import DictConfig, ListConfig, OmegaConf


def _resolve(path):
    module, _, name = path.rpartition('.')
    return getattr(importlib.import_module(module), name)


def instantiate(config, **kwargs):
    if isinstance(config, (DictConfig, ListConfig)):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, list):
        return [instantiate(c) for c in config]
    if not isinstance(config, dict) or '_target_' not in config:
        return config
    args = {k: instantiate(v) for k, v in config.items() if not k.startswith('_')}
    args.update(kwargs)
    return _resolve(config['_target_'])(**args)
