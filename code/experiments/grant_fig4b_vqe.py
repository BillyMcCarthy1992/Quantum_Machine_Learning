"""
Grant's Figure 4: train a 7-qubit Heisenberg VQE and watch the gradient
variance die off as the energy converges.

This is the second reproduction, and the important one, because it's the
exact figure the whole project is arguing with. Grant looked at this and
said "see, the variance only drops because we're converging". Maybe! But
there's no control here that can tell convergence apart from the landscape
going flat. Reproducing it faithfully is step one; the controls come later
in e1_trajectory_vs_patch.py.

Their setup, copied:
  - 7 qubits, Heisenberg on a ring, J = h = 1
  - input is B|0> with B = 7 layers of random rotations + CZ, never trained
    (identity init can't entangle, and this ground state very much is)
  - M = 2 identity blocks, L = 33, so 132 layers total
  - Adam at 0.001, 200 trials, 300 steps

While it runs it logs way more than the figure needs -- energy, distance
from the start, gradients at a handful of watched positions every step,
plus full gradients and per-block drift at ~25 checkpoints. That's
deliberate: e1 reuses this exact logging machinery.

Speed: it tries a few differentiation routes at full size, times one step,
and tells you what it picked and how long the run will take BEFORE
committing. v1 didn't do that and I lost a night to a run that was
secretly 50x slower than it should have been.

  python grant_fig4b_vqe.py --quick               8 trials, quick look
  python grant_fig4b_vqe.py --workers 6           the real thing
  python grant_fig4b_vqe.py --analyze fig4b.npz   just redraw the plot
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

SEED_VQE = 190305078  # third seed branch (random=...76, identity=...77)

_ROT = (qml.RX, qml.RY, qml.RZ)


# --------------------------------------------- the Hamiltonian we minimise
def heisenberg(n, J=1.0, h=1.0):
    coeffs, ops = [], []
    for i in range(n):  # periodic boundary
        j = (i + 1) % n
        for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
            coeffs.append(J)
            ops.append(P(i) @ P(j))
    for i in range(n):
        coeffs.append(h)
        ops.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeffs, ops)


def exact_ground_energy(n, J=1.0, h=1.0):
    return float(np.min(np.linalg.eigvalsh(qml.matrix(heisenberg(n, J, h)))))


# ------------------------------------------------- circuit building blocks
def _entangler_B(b_axes, b_params, n):
    for l in range(b_axes.shape[0]):
        for q in range(n):
            _ROT[b_axes[l, q]](b_params[l, q], wires=q)
        for q in range(n - 1):
            qml.CZ(wires=[q, q + 1])


def _identity_blocks(params, axes, n, M, L):
    """The identity-block sandwich: L rounds of (rot, CZ), then the same in
    reverse with CZ first. Start the second half at minus the first and the
    whole thing collapses to the identity."""
    for m in range(M):
        for l in range(L):
            for q in range(n):
                _ROT[axes[m, l, q]](params[m, 0, l, q], wires=q)
            for q in range(n - 1):
                qml.CZ(wires=[q, q + 1])
        for l in reversed(range(L)):
            for q in range(n - 1):
                qml.CZ(wires=[q, q + 1])
            for q in range(n):
                _ROT[axes[m, l, q]](params[m, 1, l, q], wires=q)


def _device(n):
    try:
        return qml.device("lightning.qubit", wires=n)
    except Exception:
        return qml.device("default.qubit", wires=n)


def _observable(H, n, mode):
    if mode == "hamiltonian":
        return H
    return qml.Hermitian(qml.matrix(H), wires=list(range(n)))


def make_energy_qnode(dev, H, b_axes, b_params, axes, n, M, L, dm, mode):
    obs = _observable(H, n, mode)

    def _tape(params):
        _entangler_B(b_axes, b_params, n)
        _identity_blocks(params, axes, n, M, L)
        return qml.expval(obs)

    if dm is None:
        return qml.QNode(_tape, dev)
    return qml.QNode(_tape, dev, diff_method=dm)


def choose_diff_path(n, M, L, b_layers):
    """Race the differentiation options at full size and keep the winner.

    Doing this at production size matters -- a method can look fine on a toy
    circuit and then crawl on the real one.
    """
    rng = np.random.default_rng(12345)
    H = heisenberg(n)
    b_axes = rng.integers(0, 3, size=(b_layers, n))
    b_params = rng.uniform(0, 2 * np.pi, size=(b_layers, n))
    axes = rng.integers(0, 3, size=(M, L, n))
    p = np.zeros((M, 2, L, n))
    p[:, 0] = rng.uniform(0, 2 * np.pi, size=(M, L, n))
    p[:, 1] = -p[:, 0]

    for dm, mode in (
        ("adjoint", "hamiltonian"),
        ("adjoint", "hermitian"),
        (None, "hamiltonian"),
    ):
        try:
            dev = _device(n)
            qn = make_energy_qnode(
                dev, H, b_axes, b_params, axes, n, M, L, dm, mode
            )
            gf = qml.grad(qn)
            t0 = time.time()
            _ = float(qn(pnp.array(p, requires_grad=True)))
            g = np.asarray(gf(pnp.array(p, requires_grad=True)))
            dt = time.time() - t0
            assert np.isfinite(g).all()
            return dm, mode, dt
        except Exception:
            continue
    raise RuntimeError("no working differentiation path found")


# ------- how scrambled is each block? pure numpy, because qml.matrix crawls
def _np_rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _np_ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def _np_rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])


_NP_ROT = (_np_rx, _np_ry, _np_rz)


def _np_apply_1q(T, U, q):
    T = np.tensordot(U, T, axes=([1], [q]))
    return np.moveaxis(T, 0, q)


def _np_cz_chain(T, n):
    for q in range(n - 1):
        idx = [slice(None)] * n
        idx[q] = 1
        idx[q + 1] = 1
        T[tuple(idx)] *= -1.0
    return T


def block_unitary_np(p_m, axes_m, n, L):
    """Build the block's matrix by shoving every basis state through at once."""
    dim = 2**n
    T = np.eye(dim, dtype=complex).reshape((2,) * n + (dim,))
    for l in range(L):
        for q in range(n):
            T = _np_apply_1q(T, _NP_ROT[axes_m[l, q]](p_m[0, l, q]), q)
        T = _np_cz_chain(T, n)
    for l in reversed(range(L)):
        T = _np_cz_chain(T, n)
        for q in range(n):
            T = _np_apply_1q(T, _NP_ROT[axes_m[l, q]](p_m[1, l, q]), q)
    return T.reshape(dim, dim)


def block_probe(p_m, axes_m, n, L):
    U = block_unitary_np(p_m, axes_m, n, L)
    return 1.0 - abs(np.trace(U)) / 2**n


# --------------------------------------------------- a single training run
def run_trial(args):
    (seed_seq, n, M, L, b_layers, steps, lr, checkpoints, watch, dm, mode) = args
    rng = np.random.default_rng(seed_seq)
    H = heisenberg(n)
    dev = _device(n)

    b_axes = rng.integers(0, 3, size=(b_layers, n))
    b_params = rng.uniform(0, 2 * np.pi, size=(b_layers, n))
    axes = rng.integers(0, 3, size=(M, L, n))

    p = np.zeros((M, 2, L, n))
    p[:, 0] = rng.uniform(0, 2 * np.pi, size=(M, L, n))
    p[:, 1] = -p[:, 0]
    p0 = p.copy()

    energy = make_energy_qnode(
        dev, H, b_axes, b_params, axes, n, M, L, dm, mode
    )
    grad_fn = qml.grad(energy)

    # Adam by hand -- I wanted the moment estimates visible, not hidden
    m_t = np.zeros_like(p)
    v_t = np.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8

    energies = np.empty(steps)
    dists = np.empty(steps)
    g_watch = np.empty((steps, len(watch)), dtype=np.float32)
    g_full = np.empty((len(checkpoints), p.size), dtype=np.float32)
    probes = np.empty((len(checkpoints), M), dtype=np.float32)
    ck = {int(c): i for i, c in enumerate(checkpoints)}

    for t in range(steps):
        pt = pnp.array(p, requires_grad=True)
        energies[t] = float(energy(pt))
        g = np.asarray(grad_fn(pt))

        dists[t] = float(np.linalg.norm(p - p0))
        g_flat = g.ravel()
        g_watch[t] = g_flat[watch]
        if t in ck:
            g_full[ck[t]] = g_flat
            for m in range(M):
                probes[ck[t], m] = block_probe(p[m], axes[m], n, L)

        m_t = b1 * m_t + (1 - b1) * g
        v_t = b2 * v_t + (1 - b2) * g * g
        mhat = m_t / (1 - b1 ** (t + 1))
        vhat = v_t / (1 - b2 ** (t + 1))
        p = p - lr * mhat / (np.sqrt(vhat) + eps)

    return energies, dists, g_watch, g_full, probes


# ------------------------------ paranoia checks, run before anything real
def self_test(dm, mode):
    """Small-circuit sanity checks. If these fail, stop and fix it."""
    n, M, L, b_layers = 4, 1, 3, 2
    rng = np.random.default_rng(0)
    H = heisenberg(n)
    b_axes = rng.integers(0, 3, size=(b_layers, n))
    b_params = rng.uniform(0, 2 * np.pi, size=(b_layers, n))
    axes = rng.integers(0, 3, size=(M, L, n))
    p = np.zeros((M, 2, L, n))
    p[:, 0] = rng.uniform(0, 2 * np.pi, size=(M, L, n))
    p[:, 1] = -p[:, 0]

    dv = qml.device("default.qubit", wires=n)

    @qml.qnode(dv)
    def with_blocks(pp):
        _entangler_B(b_axes, b_params, n)
        _identity_blocks(pp, axes, n, M, L)
        return qml.state()

    @qml.qnode(dv)
    def b_only():
        _entangler_B(b_axes, b_params, n)
        return qml.state()

    d = np.max(np.abs(np.asarray(with_blocks(p)) - np.asarray(b_only())))
    assert d < 1e-9, f"blocks not identity at init (diff {d:.2e})"

    pr = block_probe(p[0], axes[0], n, L)
    assert pr < 1e-9, f"block probe not zero at init ({pr:.2e})"

    dev = _device(n)
    energy = make_energy_qnode(dev, H, b_axes, b_params, axes, n, M, L, dm, mode)

    psi = np.asarray(b_only())
    e_direct = float(np.real(psi.conj() @ qml.matrix(H) @ psi))
    e_qnode = float(energy(pnp.array(p, requires_grad=True)))
    assert abs(e_qnode - e_direct) < 1e-8, "init energy mismatch"

    g = np.asarray(qml.grad(energy)(pnp.array(p, requires_grad=True))).ravel()
    for k in (0, 7, p.size - 1):
        pp, pm = p.copy().ravel(), p.copy().ravel()
        pp[k] += np.pi / 2
        pm[k] -= np.pi / 2
        gs = 0.5 * (
            float(energy(pnp.array(pp.reshape(p.shape), requires_grad=True)))
            - float(energy(pnp.array(pm.reshape(p.shape), requires_grad=True)))
        )
        assert abs(g[k] - gs) < 1e-7, f"grad mismatch at {k}: {g[k]} vs {gs}"

    print("self-tests passed: identity at init, probe=0, init energy "
          "verified, gradients match shift rule\n")


# ------------------------------------------------------------ the driver
def default_watch(n, M, L):
    idx = []
    shape = (M, 2, L, n)
    for m in range(M):
        for half in range(2):
            for l in (0, L // 2, L - 1):
                for q in range(n):
                    idx.append(np.ravel_multi_index((m, half, l, q), shape))
    return np.array(sorted(set(idx)))


def analyze(path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    d = np.load(path, allow_pickle=True)
    cfg = json.loads(str(d["config"]))
    energies, g_watch = d["energies"], d["g_watch"]
    e_min = cfg["e_min"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    mean, std = energies.mean(axis=0), energies.std(axis=0)
    a1.plot(mean, label="ID blocks (mean)")
    a1.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    a1.axhline(e_min, ls="--", color="green", label=f"E_min = {e_min:.3f}")
    a1.set_xlabel("iteration")
    a1.set_ylabel("energy")
    a1.legend()

    var = g_watch.astype(np.float64).var(axis=0)
    for j in range(var.shape[1]):
        a2.plot(var[:, j], color="C0", alpha=0.15, lw=0.7)
    a2.plot(np.median(var, axis=1), color="C1", lw=2, label="median (watch set)")
    a2.set_xlabel("iteration")
    a2.set_ylabel(r"Var[$\partial_\theta E$] across trials")
    a2.legend()

    fig.tight_layout()
    out = os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=150)
    print(f"plot saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--L", type=int, default=33)
    ap.add_argument("--b-layers", type=int, default=7)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze", type=str, default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return
    if args.quick:
        args.trials, args.steps = 8, 60

    dm, mode, dt = choose_diff_path(args.n, args.M, args.L, args.b_layers)
    dm_label = dm if dm is not None else "pennylane-default"
    per_trial = dt * args.steps
    total = per_trial * args.trials / max(args.workers, 1)
    print(f"differentiation path: diff_method={dm_label}, observable={mode}")
    print(f"timed one production-size step: {dt:.3f}s  ->  "
          f"~{per_trial:.0f}s/trial, ~{total / 60:.0f} min total "
          f"({args.workers} worker[s])")
    if dt > 1.0:
        print("WARNING: >1s/step means the fast paths failed; this will be "
              "very slow.\nConsider upgrading pennylane + pennylane-lightning "
              "before running the full ensemble.")

    self_test(dm, mode)

    e_min = exact_ground_energy(args.n)
    print(f"exact ground energy (n={args.n}): {e_min:.6f}")

    ckpts = np.unique(
        np.concatenate(
            [
                [0],
                np.round(np.logspace(0, np.log10(max(args.steps - 1, 2)), 24)),
                [args.steps - 1],
            ]
        ).astype(int)
    )
    watch = default_watch(args.n, args.M, args.L)
    seeds = np.random.SeedSequence(SEED_VQE).spawn(args.trials)
    jobs = [
        (s, args.n, args.M, args.L, args.b_layers,
         args.steps, args.lr, ckpts, watch, dm, mode)
        for s in seeds
    ]

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(run_trial, jobs))
    else:
        results = []
        for i, j in enumerate(jobs):
            results.append(run_trial(j))
            if (i + 1) % 5 == 0 or i == 0:
                el = time.time() - t0
                print(
                    f"  trial {i + 1}/{args.trials}  "
                    f"[{el:.0f}s, ~{el / (i + 1) * args.trials:.0f}s total]",
                    flush=True,
                )

    energies = np.stack([r[0] for r in results])
    dists = np.stack([r[1] for r in results])
    g_watch = np.stack([r[2] for r in results])
    g_full = np.stack([r[3] for r in results])
    probes = np.stack([r[4] for r in results])

    cfg = dict(vars(args))
    cfg.pop("analyze")
    cfg["e_min"] = e_min
    cfg["seed"] = SEED_VQE
    cfg["diff_method"] = dm_label
    cfg["observable_mode"] = mode
    out = "fig4b_quick.npz" if args.quick else "fig4b.npz"
    np.savez_compressed(
        out,
        energies=energies,
        dists=dists,
        g_watch=g_watch,
        g_full=g_full,
        probes=probes,
        checkpoints=ckpts,
        watch=watch,
        config=json.dumps(cfg),
    )
    print(
        f"\n{args.trials} trials x {args.steps} steps in "
        f"{time.time() - t0:.0f}s; saved to {out}"
    )
    print(
        f"final energy: mean {energies[:, -1].mean():+.4f}, "
        f"best {energies[:, -1].min():+.4f}, E_min {e_min:+.4f}"
    )
    analyze(out)


if __name__ == "__main__":
    main()