"""The one high-risk combination: the unfixed simulator (--as_drivor_paper, --as_upstream) at a step other than 0.25 s.

Those flags bring back upstream HUGSIM's step-count bugs, which only cancel out at its shipped dt = 0.25 s:
the episode ends after 400 steps (40 s at dt 0.1, so a route that needs longer can never be completed),
and the AttackPlanner's actors advance and replan a fixed amount per step (2.5x as far as intended at
dt 0.1). An episode run like this measures the bugs as much as the agent, so its scores are not
comparable with any other run (README, Scoring: --as_drivor_paper). closed_loop.py
warns and records the step in realized_score.json, run_benchmark.py warns before it starts, and
aggregate_benchmark.py warns about every agent whose records carry it.
"""

import math

#: HUGSIM's shipped step (configs/sim/kinematic.yaml), the only one the step-count code is right at.
STANDARD_DT = 0.25
#: Upstream HUGSIM's episode cap, a step count (closed_loop.UPSTREAM_MAX_STEPS).
UPSTREAM_MAX_STEPS = 400
#: realized_score.json key: the step of an episode run unfixed at a non-standard step.
RECORD_KEY = "unfixed_step_dt"


def is_unfixed_step(dt, as_drivor_paper=False, as_upstream=False):
    """True for the unfixed simulator (either flag) at a step other than the shipped 0.25 s."""
    return bool(as_drivor_paper or as_upstream) and not math.isclose(float(dt), STANDARD_DT)


def warning(dt, as_drivor_paper=False, as_upstream=False):
    """The warning for that combination, or None when the run is safe."""
    if not is_unfixed_step(dt, as_drivor_paper, as_upstream):
        return None
    dt = float(dt)
    flag = "--as_upstream" if as_upstream else "--as_drivor_paper"
    return (
        f"WARNING: {flag} at dt = {dt:g} s runs upstream HUGSIM's step-count bugs, which are only right at "
        f"dt = {STANDARD_DT:g} s: episodes end after {UPSTREAM_MAX_STEPS} steps ({UPSTREAM_MAX_STEPS * dt:g} s "
        f"instead of {UPSTREAM_MAX_STEPS * STANDARD_DT:g} s), and the aggressive vehicles move "
        f"{STANDARD_DT / dt:.2g}x as far as intended. Its scores are NOT comparable with the standard "
        f"benchmark or with any fixed run; use it only to reproduce results published on that simulator, "
        f"and keep its episodes in an output folder of their own."
    )


def kinematic_dt(path):
    """The step `dt` of a kinematic config (configs/sim/kinematic*.yaml), read without OmegaConf."""
    import yaml

    with open(path) as f:
        return float(yaml.safe_load(f)["dt"])
