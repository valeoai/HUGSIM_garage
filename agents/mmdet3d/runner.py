"""UniAD and VAD as HUGSIM agents: model build, checkpoint load and the per-frame step.

This is the old ``tools/closeloop/e2e.py`` of UniAD_SIM and VAD_SIM minus the FIFO loop (that is
``agents/client.py``'s ``serve``) and minus the visualisation unless ``HUGSIM_AGENT_VIZ=1``. The
image preparation, can_bus, lidar2img, command handling and output selection are the vendored
``tools/closeloop/dataparser.py`` and the lines below, unchanged.

One process runs one agent: ``uniad/`` and ``vad/`` both define a ``projects`` package with the
same registry names, so :func:`load_agent` puts exactly one of them on ``sys.path`` and refuses a
second. It also changes into that tree, because the configs and the visualiser read
``data/...``, ``sources/...`` relative to it. Paths given to it must therefore be absolute.

Environment variables:

* ``HUGSIM_VAD_EVAL=1``: call ``model.eval()`` on VAD. Upstream never does, so GridMask (p=0.7)
  and dropout stay active and its plans are stochastic; that is the default, because the
  published HUGSIM VAD numbers come from it. UniAD is always in eval mode, as upstream.
* ``HUGSIM_AGENT_VIZ=1``: write the visualisation to ``<episode>/uniad`` or ``<episode>/vad``.
"""

import multiprocessing
import os
import sys
import warnings
from queue import Queue

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
IMG_NORM_CFG = {
    'mean': np.array([103.530, 116.280, 123.675]),
    'std': np.array([1.0, 1.0, 1.0]),
    'to_rgb': False,
}
CONFIGS = {
    'uniad': 'projects/configs/stage2_e2e/base_e2e.py',
    'vad': 'projects/configs/VAD/VAD_base_e2e.py',
}
CKPT_NAMES = {'uniad': 'uniad/uniad_base_e2e.pth', 'vad': 'vad/VAD_base.pth'}


def env_flag(name):
    return os.environ.get(name, '0') not in ('', '0', 'false', 'False')


def _import_plugin(cfg):
    """The plugin import of the old e2e.py, for ``plugin_dir`` configs."""
    import importlib

    if cfg.get('plugin', False):
        module_path = os.path.dirname(cfg.plugin_dir).replace('/', '.')
        print(module_path)
        importlib.import_module(module_path)


class Agent:
    """``step(obs, info)`` -> ``(6, 2)`` float32 plan (right, forward), 0.5 s apart."""

    def __init__(self, name, ckpt, device='cuda', viz_dir=None, vad_eval=False, seed=0):
        if 'projects' in sys.modules:
            raise RuntimeError('one agent per process: a `projects` package is already imported')
        if name not in CONFIGS:
            raise ValueError(f'unknown agent {name!r}, expected one of {sorted(CONFIGS)}')
        warnings.filterwarnings('ignore')
        tree = os.path.join(HERE, name)
        ckpt = os.path.abspath(ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f'{name} checkpoint not found: {ckpt}')
        sys.path.insert(0, tree)
        os.chdir(tree)

        import torch
        from mmcv import Config
        from mmcv.cnn import fuse_conv_bn  # noqa: F401  (kept importable, as upstream)
        from mmcv.parallel import MMDataParallel
        from mmcv.runner import load_checkpoint, wrap_fp16_model
        from mmdet.apis import set_random_seed
        from mmdet3d.models import build_model

        from tools.closeloop.dataparser import parse_raw

        self.name, self.device, self.parse_raw = name, device, parse_raw
        self.torch = torch
        cfg = Config.fromfile(CONFIGS[name])
        _import_plugin(cfg)
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            cfg.data.test.pop('samples_per_gpu', 1)
        set_random_seed(seed, deterministic=False)

        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        if cfg.get('fp16', None) is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
        meta = checkpoint.get('meta', {})
        if 'CLASSES' in meta:
            model.CLASSES = meta['CLASSES']
        if 'PALETTE' in meta:
            model.PALETTE = meta['PALETTE']

        if device == 'cuda':
            model = MMDataParallel(model, device_ids=[0])
        elif device != 'cpu':
            raise ValueError(f'device must be cuda or cpu, got {device!r}')
        # UniAD upstream is in eval mode; VAD upstream never calls eval() (see module docstring).
        if name == 'uniad' or vad_eval:
            model = model.eval()
        self.model = model
        self.history = Queue()  # VAD's ego history (UniAD's parse_raw ignores it)
        self.viz_dir = viz_dir
        self.count = 0
        self.vis_pool = []
        if viz_dir is not None:
            os.makedirs(viz_dir, exist_ok=True)
            from tools.closeloop import visualizer

            self.visualizer = visualizer

    def step(self, obs, info):
        torch = self.torch
        data = self.parse_raw((obs, info), CAMERAS, IMG_NORM_CFG, self.history, device=self.device)
        raw_images = data['raw_imgs']
        del data['raw_imgs']
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)

        if self.name == 'uniad':
            plan = results[0]['planning']['result_planning']['sdc_traj'][0].detach().cpu().numpy()
        else:
            out = results[0]['pts_bbox']
            plan_cmd = np.argmax(out['ego_fut_cmd'])
            plan = np.cumsum(out['ego_fut_preds'][plan_cmd], axis=0)
        plan = np.asarray(plan, dtype=np.float32)

        if self.viz_dir is not None:
            self._visualize(results, data, raw_images, plan, info)
        self.count += 1
        return plan

    def _visualize(self, results, data, raw_images, plan, info):
        save_fn = os.path.join(self.viz_dir, str(self.count).zfill(4))
        if self.name == 'uniad':
            r = results[0]
            r['planning_traj'] = r['planning']['result_planning']['sdc_traj'].cpu().detach().numpy()
            r['pts_bbox']['score_list'] = r['pts_bbox']['score_list'].detach().cpu().numpy()
            r['pts_bbox']['lane_score'] = r['pts_bbox']['lane_score'].detach().cpu().numpy()
            r['command'] = data['command'][0].cpu().detach().numpy()
            # one process per frame, as upstream
            process = multiprocessing.Process(target=self.visualizer.save_visualize_img, args=(r, raw_images, save_fn))
            self.vis_pool.append(process)
            process.start()
        else:
            self.visualizer.save_frame(raw_images, plan, info['cam_params'], save_fn)

    def finish(self):
        if self.viz_dir is not None:
            for process in self.vis_pool:
                process.join()
            self.visualizer.to_video(self.viz_dir)


def load_agent(name, ckpt, device='cuda', output=None):
    """Build an agent from the environment-variable switches. ``output`` is the episode folder."""
    viz_dir = None
    if env_flag('HUGSIM_AGENT_VIZ'):
        if output is None:
            raise ValueError('HUGSIM_AGENT_VIZ=1 needs an output folder')
        viz_dir = os.path.join(os.path.abspath(output), name)
    return Agent(name, ckpt, device=device, viz_dir=viz_dir, vad_eval=env_flag('HUGSIM_VAD_EVAL'))
