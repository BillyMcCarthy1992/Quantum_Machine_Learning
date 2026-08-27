"""
End-to-end dry run of the E1 and E2 designs on a circuit small enough to do
everything the slow, obvious way.

The point isn't the physics (n=4 is far too small for any of the
interesting effects) -- it's to prove the machinery does what it says:
training runs, checkpoints land where expected, the null points really are
at the matched distance, and the cross-measurement logs both costs. Bugs in
plumbing are much cheaper to find here than after an overnight run.

Everything uses full parameter-shift gradients, no autograd, so it's slow
but there's nothing clever to get wrong.
"""

import time

import numpy as np

import verify_numpy_vqe as V

N, M, L, B = 4, 1, 3, 2
DIM = 2**N


def draw(rng):
    b_axes = rng.integers(0, 3, size=(B, N))
    b_params = rng.uniform(0, 2 * np.pi, size=(B, N))
    axes = rng.integers(0, 3, size=(M, L, N))
    p = np.zeros((M, 2, L, N))
    p[:, 0] = rng.uniform(0, 2 * np.pi, size=(M, L, N))
    p[:, 1] = -p[:, 0]
    return b_axes, b_params, axes, p


def state(p, axes, b_axes, b_params):
    return V.run_state(p, axes, b_axes, b_params, N, M, L).ravel()


H4 = V.heisenberg_matrix(N)
HL = sum(V.kron_term({i: V.Z}, N) for i in range(N)) / (2 * N)


def cost_heis(p, inst):
    psi = state(p, *inst)
    return float(np.real(psi.conj() @ H4 @ psi))


def cost_global(p, inst):
    psi = state(p, *inst)
    return 1.0 - abs(psi[0]) ** 2


def cost_local(p, inst):
    psi = state(p, *inst)
    return 0.5 - float(np.real(psi.conj() @ HL @ psi))


def full_grad(cost, p, inst):
    g = np.zeros_like(p)
    f = p.ravel()
    for k in range(f.size):
        pp, pm = f.copy(), f.copy()
        pp[k] += np.pi / 2
        pm[k] -= np.pi / 2
        g.ravel()[k] = 0.5 * (
            cost(pp.reshape(p.shape), inst) - cost(pm.reshape(p.shape), inst)
        )
    return g


def adam():
    s = {"m": 0, "v": 0, "t": 0}

    def step(p, g, lr=0.05):
        s["t"] += 1
        s["m"] = 0.9 * s["m"] + 0.1 * g
        s["v"] = 0.999 * s["v"] + 0.001 * g * g
        mh = s["m"] / (1 - 0.9 ** s["t"])
        vh = s["v"] / (1 - 0.999 ** s["t"])
        return p - lr * mh / (np.sqrt(vh) + 1e-8)

    return step


def null_point(rng, p0, r):
    u = rng.standard_normal(p0.shape)
    u /= np.linalg.norm(u)
    return p0 + r * u


def shift_vs_fd(cost, p, inst):
    k, h = 1, 1e-5
    f = p.ravel()
    pp, pm = f.copy(), f.copy()
    pp[k] += h
    pm[k] -= h
    fd = (cost(pp.reshape(p.shape), inst) - cost(pm.reshape(p.shape), inst)) / (2 * h)
    return abs(fd - full_grad(cost, p, inst).ravel()[k])


def mini_e1(K=6, steps=40, ckpts=(0, 4, 9, 19, 29, 39), dirs=2):
    print("--- mini-E1 (heisenberg, trajectory vs matched null) ---")
    t0 = time.time()
    gf = np.empty((K, len(ckpts), M * 2 * L * N))
    ngf = np.empty((K, len(ckpts), dirs, M * 2 * L * N))
    pr = np.empty((K, len(ckpts)))
    npr = np.empty((K, len(ckpts), dirs))
    costs = np.empty((K, steps))
    root = np.random.SeedSequence(7)
    for s, ss in enumerate(root.spawn(K)):
        tr, nl = ss.spawn(2)
        rng, nrng = np.random.default_rng(tr), np.random.default_rng(nl)
        inst = draw(rng)
        b_axes, b_params, axes, p = inst
        inst = (axes, b_axes, b_params)
        p0 = p.copy()
        assert 1 - abs(np.trace(V.block_unitary(p[0], axes[0], N, L))) / DIM < 1e-10
        if s == 0:
            assert shift_vs_fd(cost_heis, p, inst) < 1e-8
        step = adam()
        ci = {c: i for i, c in enumerate(ckpts)}
        for t in range(steps):
            costs[s, t] = cost_heis(p, inst)
            g = full_grad(cost_heis, p, inst)
            if t in ci:
                i = ci[t]
                gf[s, i] = g.ravel()
                pr[s, i] = 1 - abs(np.trace(V.block_unitary(p[0], axes[0], N, L))) / DIM
                r = float(np.linalg.norm(p - p0))
                for d in range(dirs):
                    pn = null_point(nrng, p0, r)
                    assert abs(np.linalg.norm(pn - p0) - r) < 1e-10
                    ngf[s, i, d] = full_grad(cost_heis, pn, inst).ravel()
                    npr[s, i, d] = 1 - abs(
                        np.trace(V.block_unitary(pn[0], axes[0], N, L))
                    ) / DIM
            p = step(p, g)
    assert (costs[:, -1] < costs[:, 0] - 0.5).all(), "training failed"
    vt = np.median(gf.var(axis=0), axis=-1)
    vn = np.median(
        ngf.transpose(0, 2, 1, 3).reshape(K * dirs, len(ckpts), -1).var(axis=0),
        axis=-1,
    )
    print(f"  costs: {costs[:, 0].mean():+.3f} -> {costs[:, -1].mean():+.3f} "
          f"(E_min {np.linalg.eigvalsh(H4)[0]:+.3f})")
    print("  ckpt  Var(traj)   Var(null)   probe(traj)  probe(null)")
    for i, c in enumerate(ckpts):
        print(f"  {c:4d}  {vt[i]:.4e}  {vn[i]:.4e}   {pr[:, i].mean():.3f}"
              f"        {npr[:, i].mean():.3f}")
    print(f"  mechanics OK [{time.time() - t0:.0f}s]\n")


def mini_e2(K=5, steps=40, ckpts=(0, 19, 39), dirs=2):
    print("--- mini-E2 (global vs local cost, cross-measured) ---")
    t0 = time.time()
    for train_kind, cost_tr, cost_ot, name_ot in (
        ("global", cost_global, cost_local, "local"),
        ("local", cost_local, cost_global, "global"),
    ):
        ctr = np.empty((K, steps))
        cot = np.empty((K, steps))
        gtr = np.empty((K, len(ckpts), M * 2 * L * N))
        got = np.empty((K, len(ckpts), M * 2 * L * N))
        root = np.random.SeedSequence(8)
        for s, ss in enumerate(root.spawn(K)):
            rng = np.random.default_rng(ss)
            inst = draw(rng)
            b_axes, b_params, axes, p = inst
            inst = (axes, b_axes, b_params)
            if s == 0:
                assert shift_vs_fd(cost_tr, p, inst) < 1e-8
            step = adam()
            ci = {c: i for i, c in enumerate(ckpts)}
            for t in range(steps):
                ctr[s, t] = cost_tr(p, inst)
                cot[s, t] = cost_ot(p, inst)
                g = full_grad(cost_tr, p, inst)
                if t in ci:
                    gtr[s, ci[t]] = g.ravel()
                    got[s, ci[t]] = full_grad(cost_ot, p, inst).ravel()
                p = step(p, g)
        assert (ctr[:, -1] < ctr[:, 0]).all(), f"{train_kind} training failed"
        vt = np.median(gtr.var(axis=0), axis=-1)
        vo = np.median(got.var(axis=0), axis=-1)
        print(f"  train={train_kind:6s}: C_train {ctr[:, 0].mean():.3f} -> "
              f"{ctr[:, -1].mean():.3f};  C_{name_ot} {cot[:, 0].mean():.3f} "
              f"-> {cot[:, -1].mean():.3f}")
        print(f"    Var(train-cost grads) at ckpts: "
              + "  ".join(f"{v:.2e}" for v in vt))
        print(f"    Var({name_ot}-cost grads) at ckpts:  "
              + "  ".join(f"{v:.2e}" for v in vo))
    print(f"  mechanics OK [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    mini_e1()
    mini_e2()