# agents/_vavam: VaVAM and ECO

Shared by `vavam` and `vavam_eco`. Two parts with different standing:

| file | what | from |
|---|---|---|
| `vendor/vam/action_expert/{video_action_model,joint_model,mup_action_expert}.py`, `vendor/vam/video_pretraining/{mup_gpt2,prepare_token_sequence}.py` | VaVAM's model, inference path only | [valeoai/VideoActionModel](https://github.com/valeoai/VideoActionModel) at `738050e`, `vam/`, MIT (`LICENSE`). Changed: `video_action_model.py` imports `instantiate` from `vendor/vam/_instantiate.py` instead of hydra, and logs with `logging` instead of the repo's lightning `RankedLogger`. Nothing else |
| `vendor/vam/__init__.py`, `vendor/vam/*/__init__.py`, `vendor/vam/_instantiate.py` | package files: only the modules above, and hydra's `instantiate` for `_target_` configs | written here |
| `transforms.py` | `NeuroNCAPTransform`, `SafeResize`, `Normalize`: VaVAM's image preprocessing for closed loop | VideoActionModel `vam/datalib/transforms.py`, unchanged |
| `run.py` | the agent: VaVAM's closed-loop client stepped by HUGSIM | follows VideoActionModel `inference/runner.py` (`VAMRunner.forward_inference`); see below |
| `eco.py` | ECO | **our reimplementation** of arXiv 2609.31383 (no code released); see below |

The package keeps the name `vam` because the released checkpoints name their classes by path
(hydra `_target_`s and pickled objects); only `vendor/` is put on `sys.path`.

## Weights

VaVAM-B (`VAM_width_1024_pretrained_139k.pt`, 318M + 38M parameters) and the image tokenizer
(`VQ_ds16_16384_llamagen_encoder.jit`) from the VideoActionModel v1.0.0 release, under its
**research-only RAIL-M licence** (`LICENSE_MODEL` there), not MIT. VaVAM-B is the checkpoint the ECO
paper uses ("the released VaVAM-B checkpoint throughout").

## VaVAM: what follows the original client, and what is ours

As `VAMRunner.forward_inference`: the front camera only (`CAM_FRONT`); `NeuroNCAPTransform` (any size
-> 900x1600 -> 288x512, [-1, 1]); the context frames tokenized together each step; the command as
given (HUGSIM's `info['command']`: 0 right, 1 left, 2 straight, VaVAM's encoding); the action expert's
default 10 Euler steps of flow matching (also the ECO paper's "deployed" sampler); bf16 autocast (the
client server's default dtype); 6 waypoints 0.5 s apart, ego frame x forward, y left, sent to HUGSIM as
(right, forward) = (-y, x). The noise is unseeded, as in the client (`HUGSIM_AGENT_SEED` seeds it).

Ours: the client was called at 2 Hz (NeuroNCAP), HUGSIM steps every 0.25 s, so the context is the
frames 0.5 s apart ending at the current frame (VaVAM's training rate), up to the model's
`context_length`, rather than every simulator frame. Neither the client nor the ECO paper says how
VaVAM was fed at HUGSIM's rate.

## ECO: what the paper fixes, and what is ours

From the paper (Sec. III and its appendix): the objective `J = 3 S + 8 C + 0.05 D` on the plan with
the endpoint fixed and the K = 2 last executed poses plus the current pose prepended as fixed anchors;
`S` second differences of positions and wrapped headings, `C` the turn penalty `(|c| - 0.5)^2` on
cross products above 0.5 m^2 with both segments over 0.1 m, `D` the squared deviation from the plan;
L-BFGS-B from the plan; headings of a position-only policy taken from its segments; history poses
interpolated at k * dt before the step, in the current ego frame; fewer anchors at an episode's start.

Ours, because the paper does not give them: scipy's default L-BFGS-B tolerances and iteration cap; an
exact gradient (torch autograd, float64) with the turn penalty's indicator held at the current iterate;
`D`'s heading difference wrapped like `S`'s; a solve with a non-finite cost leaves the plan unchanged;
the executed poses are HUGSIM's `info['ego_box']` (x, y, yaw), linearly interpolated (yaw on the
circle). Results of this agent are "VaVAM + ECO, our reimplementation", not the paper's.
