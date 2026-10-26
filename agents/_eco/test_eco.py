"""Checks of agents/_eco/eco.py against the ECO paper's definitions. Run from the repository root:

    python -m agents._eco.test_eco

The objective is recomputed here with explicit loops over the paper's index ranges (arXiv 2609.31383,
Sec. III-B), independently of eco.py's vectorized version.
"""

import math

import numpy as np
import torch

from agents._eco.eco import HISTORY, W_DEV, W_SMOOTH, W_TURN, eco, history_poses, objective_terms, segment_headings


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def paper_objective(hist, x, x0):
    """J on z = (h_{-K}, .., h_{-1}, h_0, x_0, .., x_{N-1}) with z_i = h_{i+1} for i < 0, by the paper's sums."""
    K = len(hist)
    seq = list(hist) + [(0.0, 0.0, 0.0)] + list(x)
    N = len(x)

    def z(i):  # paper index i in [-K-1, N-1]
        return seq[i + K + 1]

    S = 0.0
    for i in range(-K - 1, N - 2):  # i = -K-1 .. N-3
        a, b, c = z(i), z(i + 1), z(i + 2)
        S += (c[0] - 2 * b[0] + a[0]) ** 2 + (c[1] - 2 * b[1] + a[1]) ** 2 + wrap(c[2] - 2 * b[2] + a[2]) ** 2
    C = 0.0
    for i in range(-K, N - 1):  # i = -K .. N-2
        pm, p, pp = z(i - 1), z(i), z(i + 1)
        u = (p[0] - pm[0], p[1] - pm[1])
        v = (pp[0] - p[0], pp[1] - p[1])
        c = u[0] * v[1] - u[1] * v[0]
        if abs(c) > 0.5 and math.hypot(*u) > 0.1 and math.hypot(*v) > 0.1:
            C += (abs(c) - 0.5) ** 2
    D = 0.0
    for i in range(N):
        D += (x[i][0] - x0[i][0]) ** 2 + (x[i][1] - x0[i][1]) ** 2 + wrap(x[i][2] - x0[i][2]) ** 2
    return S, C, D


def check_objective(rng):
    for trial in range(200):
        n = 6
        K = int(rng.integers(0, HISTORY + 1))
        hist = [(float(-2.0 * (K - j) + rng.normal(0, 0.3)), float(rng.normal(0, 0.5)), float(rng.normal(0, 0.2))) for j in range(K)]
        x0 = np.cumsum(rng.normal([3, 0], [1, 1.2], size=(n, 2)), axis=0)
        x0 = np.column_stack([x0, segment_headings(x0)])
        x = x0 + rng.normal(scale=[0.4, 0.4, 0.1], size=x0.shape)
        x[-1] = x0[-1]
        want = paper_objective(hist, [tuple(r) for r in x], [tuple(r) for r in x0])
        A = torch.tensor(np.vstack([np.array(hist).reshape(-1, 3), np.zeros((1, 3))]), dtype=torch.float64)
        X = torch.tensor(x, dtype=torch.float64)
        got = [t.item() for t in objective_terms(torch.cat([A, X]), X, torch.tensor(x0, dtype=torch.float64))]
        assert np.allclose(got, want, rtol=1e-10, atol=1e-10), (trial, got, want)
    print('objective terms equal the paper\'s sums on 200 random cases')


def check_endpoint_and_descent(rng):
    worst = 0.0
    for trial in range(100):
        plan = np.cumsum(rng.normal([2.5, 0], [0.8, 1.0], size=(6, 2)), axis=0)
        hist = [(-2.0 + rng.normal(0, 0.2), rng.normal(0, 0.2), rng.normal(0, 0.05)),
                (-1.0 + rng.normal(0, 0.2), rng.normal(0, 0.2), rng.normal(0, 0.05))]
        out, info = eco(plan, hist)
        assert np.array_equal(out[-1], plan[-1]), 'the endpoint moved'
        assert info['cost'] <= info['cost0'] + 1e-12, (info['cost'], info['cost0'])
        worst = max(worst, info['cost'] / max(info['cost0'], 1e-12))
    print(f'endpoint fixed bit for bit and J never rises on 100 random plans (worst J/J0 {worst:.3f})')


def check_fixed_point():
    v, dt = 5.0, 0.5
    plan = np.array([[v * dt * (i + 1), 0.0] for i in range(6)])
    hist = [(-2 * v * dt, 0.0, 0.0), (-v * dt, 0.0, 0.0)]
    out, info = eco(plan, hist)
    assert np.allclose(out, plan, atol=1e-9), out
    print(f'a straight constant-speed plan matching its history is left as it is ({info["iterations"]} iterations)')


def check_zigzag():
    plan = np.array([[2.5, 0.6], [5.0, -0.6], [7.5, 0.6], [10.0, -0.6], [12.5, 0.6], [15.0, 0.0]])
    hist = [(-5.0, 0.0, 0.0), (-2.5, 0.0, 0.0)]
    out, info = eco(plan, hist)
    lat0, lat = np.abs(plan[:-1, 1]).max(), np.abs(out[:-1, 1]).max()
    assert lat < 0.5 * lat0, (lat, lat0)
    print(f'a zig-zag plan is smoothed: max lateral swing {lat0:.2f} -> {lat:.2f} m, J {info["cost0"]:.2f} -> {info["cost"]:.2f}')


def check_history():
    # constant speed 4 m/s along a heading of 30 degrees, recorded every 0.25 s
    yaw = math.radians(30)
    past = [(0.25 * k, 4 * 0.25 * k * math.cos(yaw), 4 * 0.25 * k * math.sin(yaw), yaw) for k in range(9)]
    now = past[-1]
    hist = history_poses(past[:-1], now, 0.5)
    want = [(-4.0, 0.0, 0.0), (-2.0, 0.0, 0.0)]
    assert np.allclose(hist, want, atol=1e-9), hist
    assert history_poses([], now, 0.5) == []
    assert len(history_poses(past[-2:-1], now, 0.5)) == 0  # 0.25 s of past: neither 0.5 s nor 1.0 s back exists
    assert len(history_poses(past[-3:-1], now, 0.5)) == 1
    print('history poses: interpolated in the current ego frame, and left out before the episode began')


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    print(f'weights: smooth {W_SMOOTH}, turn {W_TURN}, deviation {W_DEV}; K = {HISTORY}')
    check_objective(rng)
    check_endpoint_and_descent(rng)
    check_fixed_point()
    check_zigzag()
    check_history()
    print('all ECO checks passed')
