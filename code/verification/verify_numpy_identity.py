"""
Same second-opinion trick, but for the identity-block half -- which is the
fiddly one, so it gets the most checking.

What it proves, without any PennyLane involved:
  - at init the circuit really is the identity (to ~1e-16, not "close")
  - the watched gradient hits the three values the closed form predicts:
    axis X gives <Y><Z>, axis Y gives -<X><Z>, axis Z gives exactly 0
  - therefore the variance is a three-point-distribution variance and does
    not depend on n at all, which is why the line in Fig 1a is flat

That last point is worth dwelling on: the flatness is a theorem for this
parameter, not a lucky measurement. It also shows why the sqrt(H) input
matters -- with a plain Hadamard all three numbers would be zero.
"""

import sys
import time

import numpy as np

SEED_IDENTITY = 190305077  # distinct seed branch from the random-init runs
L_HALF = 60                # L; total depth 2L = 120
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


def apply_cz_chain(state, n):
    for q in range(n - 1):
        idx = [slice(None)] * n
        idx[q] = 1
        idx[q + 1] = 1
        state[tuple(idx)] *= -1.0
    return state


def input_state(n):
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0
    for q in range(n):
        state = apply_1q(state, SQRT_H, q)
    return state


def run_circuit(p1, p2, axes, n):
    """Input prep, first half (rot, CZ) x L, second half (CZ, rot) mirrored."""
    state = input_state(n)
    for l in range(L_HALF):
        for q in range(n):
            state = apply_1q(state, ROT[axes[l, q]](p1[l, q]), q)
        state = apply_cz_chain(state, n)
    for l in reversed(range(L_HALF)):
        state = apply_cz_chain(state, n)
        for q in range(n):
            state = apply_1q(state, ROT[axes[l, q]](p2[l, q]), q)
    return state


def energy_zz(state):
    p = np.abs(state.reshape(2, 2, -1)) ** 2
    return float(p[0, 0].sum() + p[1, 1].sum() - p[0, 1].sum() - p[1, 0].sum())


def watched_grad(p1, p2, axes, n):
    plus, minus = p1.copy(), p1.copy()
    plus[0, 0] += np.pi / 2
    minus[0, 0] -= np.pi / 2
    e_p = energy_zz(run_circuit(plus, p2, axes, n))
    e_m = energy_zz(run_circuit(minus, p2, axes, n))
    return 0.5 * (e_p - e_m)


# ------------------------------------------- the checks that matter
def single_qubit_expectations():
    s = SQRT_H @ np.array([1.0, 0.0])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return tuple(float(np.real(s.conj() @ M @ s)) for M in (X, Y, Z))


def unit_tests(n=3, seed=0):
    rng = np.random.default_rng(seed)
    axes = rng.integers(0, 3, size=(L_HALF, n))
    p1 = rng.uniform(0, 2 * np.pi, size=(L_HALF, n))
    p2 = -p1

    # 1. does nothing at all at step zero, as advertised
    diff = np.max(np.abs(run_circuit(p1, p2, axes, n) - input_state(n)))
    assert diff < 1e-10, f"circuit not identity at init (max diff {diff:.2e})"

    # 2. force each axis and compare against the hand-derived value
    ex, ey, ez = single_qubit_expectations()
    expected = {0: ey * ez, 1: -ex * ez, 2: 0.0}  # RX, RY, RZ on qubit 0
    for a in (0, 1, 2):
        axes_f = axes.copy()
        axes_f[0, 0] = a
        g = watched_grad(p1, p2, axes_f, n)
        assert abs(g - expected[a]) < 1e-10, (
            f"axis {a}: grad {g:.6f} != analytic {expected[a]:.6f}"
        )

    var_analytic = float(np.var(list(expected.values())))
    print(f"unit tests passed (n={n}): identity exact, boundary gradient")
    print(f"  <X>,<Y>,<Z> on sqrt(H)|0> = {ex:+.4f}, {ey:+.4f}, {ez:+.4f}")
    print(f"  3-point gradient values   = "
          f"{expected[0]:+.6f}, {expected[1]:+.6f}, {expected[2]:+.6f}")
    print(f"  analytic flat-line Var    = {var_analytic:.6f}\n")
    return var_analytic


def main():
    n_values = [int(a) for a in sys.argv[1:]] or [2, 4, 6]
    var_analytic = unit_tests()

    full_range = list(range(2, 13))
    children = np.random.SeedSequence(SEED_IDENTITY).spawn(len(full_range))
    child_of = dict(zip(full_range, children))

    for n in n_values:
        t0 = time.time()
        rng = np.random.default_rng(child_of[n])
        grads = np.empty(N_SAMPLES)
        for s in range(N_SAMPLES):
            axes = rng.integers(0, 3, size=(L_HALF, n))
            p1 = rng.uniform(0.0, 2.0 * np.pi, size=(L_HALF, n))
            grads[s] = watched_grad(p1, -p1, axes, n)
        var = float(np.var(grads))
        print(
            f"n={n:2d}  Var={var:.6e}  (analytic {var_analytic:.6e})  "
            f"ratio={var / var_analytic:.3f}  [{time.time() - t0:.1f}s]",
            flush=True,
        )


if __name__ == "__main__":
    main()