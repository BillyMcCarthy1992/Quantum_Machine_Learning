"""
The other half of Figure 1a: identity-block init, where the variance is
supposed to stop caring about how many qubits you have.

How the circuit goes together (Grant, Eq. 7-8): M = 1 block, L = 60.
First half is L rounds of (rotations, CZ chain). Second half is the same
thing mirrored -- CZ chain FIRST, then rotations, layers in reverse order --
with its own separate parameter array that just happens to start at minus
the first half. Everything cancels, the circuit is exactly the identity at
step zero, and after that the two halves are free to wander apart.

The fun bit: for the parameter we're watching, the flat line isn't just an
observation, it's forced. Everything after the first gate cancels against
its own inverse, so the gradient collapses to

    (i/2) <psi| [P, Z0 Z1] |psi>

which only depends on which axis P got drawn for that one gate. Three
possible values, so the variance is a three-point-distribution variance and
n literally cannot appear. Comes out to 0.022025 for the sqrt(H) input.
Also explains why plain Hadamard would have been a disaster: all three
values would be zero and the line would sit on the floor.

Seeds line up with verify_numpy_identity.py, so both should spit out the
same numbers to the last digit.

  python grant_fig1a_identity.py            tests + full sweep
  python grant_fig1a_identity.py --quick    tests + a fast sanity sweep
"""

import sys
import time

import numpy as np
import pennylane as qml

SEED_IDENTITY = 190305077  # distinct seed branch from the random-init runs
L_HALF = 60                # L; total depth 2L = 120

_H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
SQRT_H = np.exp(1j * np.pi / 4) * (
    np.cos(np.pi / 4) * np.eye(2) - 1j * np.sin(np.pi / 4) * _H
)
assert np.allclose(SQRT_H @ SQRT_H, _H)

_ROT = (qml.RX, qml.RY, qml.RZ)


def _prep(n_qubits):
    for q in range(n_qubits):
        qml.QubitUnitary(SQRT_H, wires=q)


def _first_half(p1, axes, n_qubits):
    for l in range(L_HALF):
        for q in range(n_qubits):
            _ROT[axes[l, q]](p1[l, q], wires=q)
        for q in range(n_qubits - 1):
            qml.CZ(wires=[q, q + 1])


def _second_half(p2, axes, n_qubits):
    for l in reversed(range(L_HALF)):
        for q in range(n_qubits - 1):
            qml.CZ(wires=[q, q + 1])
        for q in range(n_qubits):
            _ROT[axes[l, q]](p2[l, q], wires=q)


def _device(n_qubits):
    try:
        return qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        return qml.device("default.qubit", wires=n_qubits)


def make_energy_qnode(dev, n_qubits, axes):
    @qml.qnode(dev)
    def energy(p1, p2):
        _prep(n_qubits)
        _first_half(p1, axes, n_qubits)
        _second_half(p2, axes, n_qubits)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return energy


def watched_grad(energy, p1, p2):
    """Derivative w.r.t. the first angle. Exact, two circuit calls."""
    plus, minus = p1.copy(), p1.copy()
    plus[0, 0] += np.pi / 2
    minus[0, 0] -= np.pi / 2
    return 0.5 * (float(energy(plus, p2)) - float(energy(minus, p2)))


# -------------------- checks that must pass or nothing below is trustworthy
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

    # 1. at init the blocks should do precisely nothing to the input state
    dev = qml.device("default.qubit", wires=n)  # need qml.state()

    @qml.qnode(dev)
    def full_state(p1_, p2_):
        _prep(n)
        _first_half(p1_, axes, n)
        _second_half(p2_, axes, n)
        return qml.state()

    @qml.qnode(dev)
    def prep_state():
        _prep(n)
        return qml.state()

    diff = np.max(np.abs(np.asarray(full_state(p1, p2)) - np.asarray(prep_state())))
    assert diff < 1e-9, f"circuit not identity at init (max diff {diff:.2e})"

    # 2. force each axis in turn and check we hit the three predicted values
    ex, ey, ez = single_qubit_expectations()
    expected = {0: ey * ez, 1: -ex * ez, 2: 0.0}
    for a in (0, 1, 2):
        axes_f = axes.copy()
        axes_f[0, 0] = a
        energy = make_energy_qnode(_device(n), n, axes_f)
        g = watched_grad(energy, p1, p2)
        assert abs(g - expected[a]) < 1e-9, (
            f"axis {a}: grad {g:.6f} != analytic {expected[a]:.6f}"
        )

    var_analytic = float(np.var(list(expected.values())))
    print("unit tests passed: identity at init, analytic boundary gradient")
    print(f"analytic flat-line variance = {var_analytic:.6f}\n")
    return var_analytic


# --------------------------------------------- the actual sweep over n
def identity_block_gradients(n_qubits, n_samples=200, rng=None):
    rng = np.random.default_rng(rng)
    dev = _device(n_qubits)
    grads = np.empty(n_samples)
    for s in range(n_samples):
        axes = rng.integers(0, 3, size=(L_HALF, n_qubits))
        p1 = rng.uniform(0.0, 2.0 * np.pi, size=(L_HALF, n_qubits))
        energy = make_energy_qnode(dev, n_qubits, axes)
        grads[s] = watched_grad(energy, p1, -p1)
    return grads


def main():
    quick = "--quick" in sys.argv
    n_samples = 50 if quick else 200
    qubit_range = range(2, 9, 2) if quick else range(2, 13)

    var_analytic = unit_tests()

    full_range = list(range(2, 13))
    children = np.random.SeedSequence(SEED_IDENTITY).spawn(len(full_range))
    child_of = dict(zip(full_range, children))

    print("Grant et al. Fig. 1(a), identity-block half")
    print(f"M=1, L={L_HALF} (120 layers), {n_samples} circuits/point, "
          f"seed={SEED_IDENTITY}\n")

    ns, variances, all_grads = [], [], {}
    for n in qubit_range:
        t0 = time.time()
        g = identity_block_gradients(n, n_samples, rng=child_of[n])
        var = float(np.var(g))
        ns.append(n)
        variances.append(var)
        all_grads[f"n{n}"] = g
        print(
            f"n={n:2d}  Var={var:.4e}  analytic={var_analytic:.4e}  "
            f"ratio={var / var_analytic:.3f}   [{time.time() - t0:.1f}s]"
        )

    out = "grant_fig1a_identity.npz"
    np.savez(
        out,
        n=np.array(ns, dtype=float),
        variance=np.array(variances),
        var_analytic=var_analytic,
        seed=SEED_IDENTITY,
        L=L_HALF,
        n_samples=n_samples,
        **all_grads,
    )
    print(f"\nraw gradients + summary saved to {out}")

    try:  # draw both curves together if the random run already happened
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.semilogy(ns, variances, "o-", label="identity block")
        ax.axhline(var_analytic, ls=":", color="gray",
                   label=f"analytic {var_analytic:.4f}")
        try:
            rnd = np.load("grant_fig1a_random.npz")
            ax.semilogy(rnd["n"], rnd["variance"], "s-", label="random init")
        except FileNotFoundError:
            pass
        ax.set_xlabel("n qubits")
        ax.set_ylabel(r"Var[$\partial_{\theta_{1,1,1}} E$]")
        ax.set_title("Grant Fig. 1(a) reproduction")
        ax.legend()
        fig.tight_layout()
        fig.savefig("grant_fig1a.png", dpi=150)
        print("plot saved to grant_fig1a.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()