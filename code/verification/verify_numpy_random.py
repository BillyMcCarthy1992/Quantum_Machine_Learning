"""
Second opinion, written from scratch in plain numpy.

No PennyLane anywhere in here. It simulates the exact same seeded circuits
as grant_fig1a_random.py by hand, so if the two disagree, one of them has a
bug and I want to know before building anything on top. (They agree to
every printed digit.)

Also checks the parameter-shift rule against a plain finite difference,
because it would be embarrassing to get gradients subtly wrong.
"""

import sys
import time

import numpy as np

SEED = 190305076
N_LAYERS = 120
N_SAMPLES = 200

_H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
SQRT_H = np.exp(1j * np.pi / 4) * (
    np.cos(np.pi / 4) * np.eye(2) - 1j * np.sin(np.pi / 4) * _H
)
assert np.allclose(SQRT_H @ SQRT_H, _H)


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])


ROT = (rx, ry, rz)


def apply_1q(state, U, q):
    state = np.tensordot(U, state, axes=([1], [q]))
    return np.moveaxis(state, 0, q)


def apply_cz(state, q, n):
    idx = [slice(None)] * n
    idx[q] = 1
    idx[q + 1] = 1
    state[tuple(idx)] *= -1.0
    return state


def energy(params, axes, n):
    """<psi| Z_0 Z_1 |psi> for the sampled circuit."""
    state = np.ones(1, dtype=complex).reshape((1,) * n)
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    for q in range(n):
        state = apply_1q(state, SQRT_H, q)
    for l in range(N_LAYERS):
        for q in range(n):
            state = apply_1q(state, ROT[axes[l, q]](params[l, q]), q)
        for q in range(n - 1):
            state = apply_cz(state, q, n)
    p = np.abs(state.reshape(2, 2, -1)) ** 2
    return float(p[0, 0].sum() + p[1, 1].sum() - p[0, 1].sum() - p[1, 0].sum())


def watched_grad(params, axes, n):
    plus = params.copy()
    minus = params.copy()
    plus[0, 0] += np.pi / 2
    minus[0, 0] -= np.pi / 2
    return 0.5 * (energy(plus, axes, n) - energy(minus, axes, n))


def main():
    n_values = [int(a) for a in sys.argv[1:]] or [2, 4, 6]
    full_range = list(range(2, 13))
    children = np.random.SeedSequence(SEED).spawn(len(full_range))
    child_of = dict(zip(full_range, children))

    for n in n_values:
        t0 = time.time()
        rng = np.random.default_rng(child_of[n])
        grads = np.empty(N_SAMPLES)
        fd_check = None
        for s in range(N_SAMPLES):
            axes = rng.integers(0, 3, size=(N_LAYERS, n))
            params = rng.uniform(0.0, 2.0 * np.pi, size=(N_LAYERS, n))
            grads[s] = watched_grad(params, axes, n)
            if s == 0:  # validate shift rule once per n
                h = 1e-5
                pp, pm = params.copy(), params.copy()
                pp[0, 0] += h
                pm[0, 0] -= h
                fd = (energy(pp, axes, n) - energy(pm, axes, n)) / (2 * h)
                fd_check = abs(fd - grads[0])
        var = float(np.var(grads))
        print(
            f"n={n:2d}  Var={var:.6e}  log10={np.log10(var):+.3f}  "
            f"|shift-FD|={fd_check:.2e}  [{time.time() - t0:.1f}s]",
            flush=True,
        )


if __name__ == "__main__":
    main()