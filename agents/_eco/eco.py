"""ECO: endpoint-constrained optimization of a waypoint plan (our reimplementation).

Zhang, Golchoubian, Gilitschenski, Ivanovic and Chitta, "Guiding End-to-End Driving Models with
Endpoint-Constrained Trajectory Optimization", arXiv 2609.31383 (v1, 2026-09-25), Sec. III. No code
was released with it (its project page says "Code (Coming soon)", October 2026), so this follows the
paper's equations; what the paper leaves open is marked "not in the paper" below.

The plan's intermediate waypoints x_0 .. x_{N-2}, each (p_i, theta_i), are re-solved while the policy's
endpoint x_{N-1} stays fixed and the K = 2 most recently executed poses h_{-2}, h_{-1} and the current
pose h_0 = (0, 0, 0) are prepended as fixed anchors. On z = (h_{-K}, .., h_0, x_0, .., x_{N-1}), with
z_i = h_{i+1} for i < 0:

    J(x) = 3 S(z) + 8 C(z) + 0.05 D(x, x0)
    S(z) = sum_{i=-K-1}^{N-3} |p_{i+2} - 2 p_{i+1} + p_i|^2 + wrap(th_{i+2} - 2 th_{i+1} + th_i)^2
    C(z) = sum_{i=-K}^{N-2} 1_i (|c_i| - 0.5)^2,  c_i = (p_i - p_{i-1}) x (p_{i+1} - p_i)
           1_i = 1 when |c_i| > 0.5 m^2 and both segments are longer than 0.1 m
    D    = sum_{i=0}^{N-1} |x_i - x0_i|^2           (the endpoint's term is 0: it is fixed)

solved with L-BFGS-B from x0. A policy that outputs headings (DrivoR) gives theta0_i itself; a position-only
policy's (VaVAM) are those of its segments: theta_i is the heading of p_{i-1} -> p_i, with p_{-1} = h_0 (the
paper's convention for such policies).

Not in the paper, chosen here: the solver's tolerances and iteration cap are scipy's L-BFGS-B
defaults; D's heading difference is wrapped to [-pi, pi] like S's; the gradient is exact (torch
autograd, float64), the indicator 1_i being held at its value at the current iterate; a solve that
fails (scipy's ``success`` false with a non-finite cost) returns the plan unchanged.
"""

import math

import numpy as np
import torch
from scipy.optimize import minimize

W_SMOOTH, W_TURN, W_DEV = 3.0, 8.0, 0.05
TURN_THRESHOLD = 0.5  # m^2
MIN_SEGMENT = 0.1  # m
HISTORY = 2  # K


def segment_headings(points, start=(0.0, 0.0)):
    """Heading of each segment p_{i-1} -> p_i, with p_{-1} = ``start``; ``points`` is (N, 2)."""
    prev = np.vstack([np.asarray(start, dtype=np.float64)[None, :2], points[:-1]])
    d = points - prev
    return np.arctan2(d[:, 1], d[:, 0])


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def objective_terms(z, x, x0):
    """(S, C, D) on torch float64 tensors: ``z`` (K+1+N, 3) the whole sequence, ``x`` (N, 3) the
    plan part of it, ``x0`` (N, 3) the policy's plan."""
    p, th = z[:, :2], z[:, 2]
    dd = p[2:] - 2 * p[1:-1] + p[:-2]
    S = (dd**2).sum() + (_wrap(th[2:] - 2 * th[1:-1] + th[:-2]) ** 2).sum()
    a, b = p[1:-1] - p[:-2], p[2:] - p[1:-1]  # segments into and out of each interior point
    c = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    with torch.no_grad():
        active = (c.abs() > TURN_THRESHOLD) & (a.norm(dim=1) > MIN_SEGMENT) & (b.norm(dim=1) > MIN_SEGMENT)
    C = ((c.abs() - TURN_THRESHOLD) ** 2 * active).sum()
    dx = x - x0
    D = (dx[:, :2] ** 2).sum() + (_wrap(dx[:, 2]) ** 2).sum()
    return S, C, D


def eco(plan, history=(), headings=None):
    """ECO on a position plan.

    ``plan``: (N, 2) waypoints, ego frame (x forward, y left), the policy's fixed-period output.
    ``history``: up to K executed poses (x, y, heading) in the current ego frame, oldest first
    (h_{-K} .. h_{-1}); the current pose (0, 0, 0) is added here. Fewer than K at an episode's start.
    ``headings``: (N,) the policy's own waypoint headings (ego frame), else those of the segments.
    Returns ``(points (N, 2), info)``; the last point is ``plan[-1]`` exactly.
    """
    plan = np.asarray(plan, dtype=np.float64)
    n = len(plan)
    theta0 = segment_headings(plan) if headings is None else np.asarray(headings, dtype=np.float64).reshape(-1)
    x0 = np.column_stack([plan, theta0])
    hist = np.asarray(list(history)[-HISTORY:], dtype=np.float64).reshape(-1, 3)
    anchors = np.vstack([hist, np.zeros((1, 3))])
    if n < 3:
        return plan.copy(), {'iterations': 0, 'success': True, 'skipped': 'fewer than 3 waypoints'}
    A = torch.from_numpy(anchors)
    X0 = torch.from_numpy(x0)
    end = X0[-1:]

    def fun(v):
        free = torch.from_numpy(v.reshape(n - 1, 3)).requires_grad_(True)
        x = torch.cat([free, end], dim=0)
        S, C, D = objective_terms(torch.cat([A, x], dim=0), x, X0)
        J = W_SMOOTH * S + W_TURN * C + W_DEV * D
        J.backward()
        return J.item(), free.grad.numpy().reshape(-1).copy()

    v0 = x0[:-1].reshape(-1)
    res = minimize(fun, v0, jac=True, method='L-BFGS-B')
    if not np.isfinite(res.fun):
        return plan.copy(), {'iterations': int(res.nit), 'success': False, 'cost': float(res.fun)}
    out = np.vstack([res.x.reshape(n - 1, 3)[:, :2], plan[-1:]])
    return out, {'iterations': int(res.nit), 'success': bool(res.success), 'cost': float(res.fun),
                 'cost0': float(fun(v0)[0])}


def history_poses(past, now, dt, k=HISTORY):
    """The executed poses k*dt, .., dt before ``now``, in ``now``'s ego frame, oldest first.

    ``past``: list of (t, x, y, yaw) of executed world poses (ego_box order, any spacing, increasing t),
    ``now`` = (t, x, y, yaw). Each is linearly interpolated (yaw on the circle) between the recorded
    poses around it; a time before the first recorded pose is left out, as at an episode's start.
    """
    if not past:
        return []
    ts = np.array([p[0] for p in past])
    out = []
    t0, x0, y0, yaw0 = now
    cos0, sin0 = math.cos(yaw0), math.sin(yaw0)
    for j in range(k, 0, -1):
        t = t0 - j * dt
        if t < ts[0] - 1e-6:
            continue
        i = int(np.searchsorted(ts, t, side='right')) - 1
        if i >= len(past) - 1:
            a = b = past[-1]
            w = 0.0
        else:
            a, b = past[i], past[i + 1]
            span = b[0] - a[0]
            w = 0.0 if span <= 0 else (t - a[0]) / span
        x = a[1] + w * (b[1] - a[1])
        y = a[2] + w * (b[2] - a[2])
        yaw = a[3] + w * math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3]))
        dx, dy = x - x0, y - y0
        out.append((cos0 * dx + sin0 * dy, -sin0 * dx + cos0 * dy, math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))))
    return out
