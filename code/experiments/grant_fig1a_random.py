"""
Redoing the "random init is doomed" half of Figure 1a from the Grant paper.

This is the boring-on-purpose starting point: if I can't reproduce a known
result, nothing downstream is worth believing. Everything is copied from
their Section 4.1 as literally as I could manage:

  - 120 layers of (random X/Y/Z rotation on each qubit, then a CZ chain)
  - measure ZZ on the first two qubits
  - start from sqrt(H) on every qubit. NOT a plain Hadamard! I nearly got
    this wrong. Plain H gives <Z> = <Y> = 0 per qubit, which quietly kills
    the identity-block half of the figure later on.
  - angles uniform in [0, 2pi), 200 circuits per point, n = 2 to 12
  - watch the gradient of the very first rotation, params[0, 0]

Speed note: we only ever need ONE gradient component, so there's no reason
to call autograd. Two shifted circuit evaluations and the parameter-shift
rule give it exactly. Runs in minutes instead of forever.

Two things the paper never actually says, so I picked and wrote down:
the CZ chain doesn't wrap around, and ZZ sits on qubits (0, 1).

  python grant_fig1a_random.py            full sweep
  python grant_fig1a_random.py --quick    tiny version, for checking it runs
"""

import sys
import time

import numpy as np
import pennylane as qml

SEED = 190305076  # arXiv number of Grant et al.; change to redraw ensembles

# --- sqrt(H), the input gate. H squares to identity, so this is easy in closed
# --- form and I don't need scipy just for one matrix square root. ----------
# H is an involution, so exp(-i a H) = cos(a) I - i sin(a) H, and
# sqrt(H) = e^{i pi/4} exp(-i (pi/4) H)  (principal branch, no scipy needed)
_H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
SQRT_H = np.exp(1j * np.pi / 4) * (
    np.cos(np.pi / 4) * np.eye(2) - 1j * np.sin(np.pi / 4) * _H
)
assert np.allclose(SQRT_H @ SQRT_H, _H), "sqrt(H) construction failed"
assert np.allclose(SQRT_H.conj().T @ SQRT_H, np.eye(2)), "sqrt(H) not unitary"

_ROT = (qml.RX, qml.RY, qml.RZ)


def _make_qnode(dev, n_qubits, n_layers, axes):
    """One circuit. Axes are baked in, angles stay free."""

    @qml.qnode(dev)
    def circuit(params):
        for q in range(n_qubits):
            qml.QubitUnitary(SQRT_H, wires=q)
        for l in range(n_layers):
            for q in range(n_qubits):
                _ROT[axes[l, q]](params[l, q], wires=q)
            for q in range(n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return circuit


def _device(n_qubits):
    try:
        return qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        return qml.device("default.qubit", wires=n_qubits)


def watched_gradients(n_qubits, n_layers=120, n_samples=200, rng=None):
    """Grab the watched gradient once per circuit.

    Hands back the raw samples, not the variance -- future me will want to
    bootstrap these, and re-running the sweep to get them back would be daft.
    """
    rng = np.random.default_rng(rng)
    dev = _device(n_qubits)
    grads = np.empty(n_samples)
    for s in range(n_samples):
        axes = rng.integers(0, 3, size=(n_layers, n_qubits))
        params = rng.uniform(0.0, 2.0 * np.pi, size=(n_layers, n_qubits))
        circuit = _make_qnode(dev, n_qubits, n_layers, axes)
        # parameter-shift: shift one angle by +-pi/2, difference over 2. Exact.
        plus = params.copy()
        minus = params.copy()
        plus[0, 0] += np.pi / 2.0
        minus[0, 0] -= np.pi / 2.0
        grads[s] = 0.5 * (float(circuit(plus)) - float(circuit(minus)))
    return grads


def main():
    quick = "--quick" in sys.argv
    n_layers = 120
    n_samples = 50 if quick else 200
    qubit_range = range(2, 9, 2) if quick else range(2, 13)

    # one seed child per system size, so any single n can be re-run alone
    children = np.random.SeedSequence(SEED).spawn(len(qubit_range))

    print("Grant et al. Fig. 1(a), random-init half")
    print(f"{n_layers} layers, {n_samples} circuits/point, seed={SEED}\n")

    ns, variances, all_grads = [], [], {}
    for n, child in zip(qubit_range, children):
        t0 = time.time()
        g = watched_gradients(n, n_layers, n_samples, rng=child)
        var = float(np.var(g))
        ns.append(n)
        variances.append(var)
        all_grads[f"n{n}"] = g
        rel_se = np.sqrt(2.0 / (n_samples - 1))  # rough error bar on a variance
        print(
            f"n={n:2d}  Var={var:.4e}  (+-{100 * rel_se:.0f}%)  "
            f"log10={np.log10(var):+.3f}   [{time.time() - t0:.1f}s]"
        )

    ns_arr = np.array(ns, dtype=float)
    log_v = np.log(np.array(variances))
    slope, intercept = np.polyfit(ns_arr, log_v, 1)
    print(f"\nexponential fit: Var ~ exp({slope:.3f} * n)")
    print(f"decay factor per qubit: {np.exp(slope):.3f}")

    out = "grant_fig1a_random.npz"
    np.savez(
        out,
        n=ns_arr,
        variance=np.array(variances),
        slope=slope,
        intercept=intercept,
        seed=SEED,
        n_layers=n_layers,
        n_samples=n_samples,
        **all_grads,
    )
    print(f"raw gradients + summary saved to {out}")

    try:  # plot if matplotlib is around, shrug if not
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.semilogy(ns_arr, variances, "o-", label="random init")
        ax.semilogy(
            ns_arr,
            np.exp(intercept + slope * ns_arr),
            "--",
            color="gray",
            label=f"fit: exp({slope:.2f} n)",
        )
        ax.set_xlabel("n qubits")
        ax.set_ylabel(r"Var[$\partial_{\theta_{1,1,1}} E$]")
        ax.set_title("Grant Fig. 1(a) reproduction (random init)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("grant_fig1a_random.png", dpi=150)
        print("plot saved to grant_fig1a_random.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()