"""
E2. Does it matter what you measure?

Cerezo's result says global cost functions are hopeless (gradients vanish
even for shallow circuits) while local ones are fine up to log depth. That's
a statement about random points in the landscape. This asks the training
version of it.

The task: take the entangled state B|0> and squash it back down to |0...0>.
Two ways to score that, and crucially they bottom out at the same state:

  C_G = 1 - |<0...0|psi>|^2        global: did you get ALL of it right
  C_L = 1 - average <P0_i>         local: how many qubits are roughly right

Pick which one to train with using --train. Then, along every trajectory,
log the gradients of BOTH. That cross-measurement is the whole design: it
separates "this cost has a nicer landscape" from "this cost steers you
somewhere nicer", which you can't untangle by training each one alone.

Things I expected going in (some right, some not):
  - at the very start the circuit is effectively shallow, so C_L should have
    healthy gradients while C_G is already squashed
  - the local cost should have a wider fertile region (Mhiri's frequency
    argument)
  - training on C_L should wake C_G's gradients up as fidelity climbs, i.e.
    falling into the narrow gorge

  python e2_cost_locality.py --train local  --workers 6
  python e2_cost_locality.py --train global --workers 6
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# --- so this runs from wherever, not just from one magic folder ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.dirname(_HERE), _HERE):   # code/ , code/experiments/
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import idblock_lib as lib

SEED_E2 = 190305080


def run_seed(job):
    (train_ss, null_ss, n, M, L, b_layers, steps, lr, opt_name, ckpts,
     watch, dirs, train_kind, paths) = job
    rng = np.random.default_rng(train_ss)
    nrng = np.random.default_rng(null_ss)

    b_axes, b_params, axes, p = lib.draw_instance(rng, n, M, L, b_layers)
    p0 = p.copy()
    dev = lib.device(n)

    costs, grads = {}, {}
    for kind in ("global", "local"):
        off, sc, factory = lib.cost_def(kind, n)
        dm, mode = paths[kind]
        costs[kind] = lib.make_cost(dev, factory, off, sc, b_axes, b_params,
                                    axes, n, M, L, dm, mode)
        grads[kind] = qml.grad(costs[kind])
    other = "local" if train_kind == "global" else "global"
    opt = lib.make_optimizer(opt_name, p.shape, lr)

    C, P = len(ckpts), p.size
    c_train = np.empty(steps)
    c_other = np.empty(steps)
    dists = np.empty(steps)
    gw_train = np.empty((steps, len(watch)), dtype=np.float32)
    gw_other = np.empty((steps, len(watch)), dtype=np.float32)
    gf_train = np.empty((C, P), dtype=np.float32)
    gf_other = np.empty((C, P), dtype=np.float32)
    probes = np.empty((C, M), dtype=np.float32)
    nc_train = np.empty((C, dirs), dtype=np.float32)
    nc_other = np.empty((C, dirs), dtype=np.float32)
    ngf_train = np.empty((C, dirs, P), dtype=np.float32)
    ngf_other = np.empty((C, dirs, P), dtype=np.float32)
    n_probes = np.empty((C, dirs, M), dtype=np.float32)
    ck = {int(c): i for i, c in enumerate(ckpts)}

    for t in range(steps):
        pt = pnp.array(p, requires_grad=True)
        c_train[t] = float(costs[train_kind](pt))
        c_other[t] = float(costs[other](pt))
        g_tr = np.asarray(grads[train_kind](pt))
        g_ot = np.asarray(grads[other](pt))
        dists[t] = float(np.linalg.norm(p - p0))
        gw_train[t] = g_tr.ravel()[watch]
        gw_other[t] = g_ot.ravel()[watch]

        if t in ck:
            i = ck[t]
            gf_train[i] = g_tr.ravel()
            gf_other[i] = g_ot.ravel()
            probes[i] = lib.all_probes(p, axes, n, M, L)
            r = dists[t]
            for d in range(dirs):
                pn = lib.null_point(nrng, p0, r)
                pnt = pnp.array(pn, requires_grad=True)
                nc_train[i, d] = float(costs[train_kind](pnt))
                nc_other[i, d] = float(costs[other](pnt))
                ngf_train[i, d] = np.asarray(grads[train_kind](pnt)).ravel()
                ngf_other[i, d] = np.asarray(grads[other](pnt)).ravel()
                n_probes[i, d] = lib.all_probes(pn, axes, n, M, L)

        p = opt.step(p, g_tr)

    return (c_train, c_other, dists, gw_train, gw_other, gf_train, gf_other,
            probes, nc_train, nc_other, ngf_train, ngf_other, n_probes)


def self_test(paths):
    n, M, L, b = 4, 1, 3, 2
    rng = np.random.default_rng(0)
    b_axes, b_params, axes, p = lib.draw_instance(rng, n, M, L, b)
    for kind in ("global", "local"):
        off, sc, factory = lib.cost_def(kind, n)
        dm, mode = paths[kind]
        cost = lib.make_cost(lib.device(n), factory, off, sc, b_axes,
                             b_params, axes, n, M, L, dm, mode)
        c0 = float(cost(pnp.array(p, requires_grad=True)))
        assert 0.0 <= c0 <= 1.0 + 1e-9, f"{kind} cost out of range: {c0}"
        g = np.asarray(
            qml.grad(cost)(pnp.array(p, requires_grad=True))
        ).ravel()
        k = 0
        pp, pm = p.copy().ravel(), p.copy().ravel()
        pp[k] += np.pi / 2
        pm[k] -= np.pi / 2
        gs = 0.5 * (
            float(cost(pnp.array(pp.reshape(p.shape), requires_grad=True)))
            - float(cost(pnp.array(pm.reshape(p.shape), requires_grad=True)))
        )
        assert abs(g[k] - gs) < 1e-7, f"{kind} grad mismatch"
    print("self-tests passed (both costs in [0,1] at init; shift rule)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=("global", "local"), required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--L", type=int, default=33)
    ap.add_argument("--b-layers", type=int, default=None)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--optimizer", choices=("adam", "gd"), default="adam")
    ap.add_argument("--dirs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.b_layers is None:
        args.b_layers = args.n
    if args.quick:
        args.trials, args.steps, args.dirs = 6, 40, 2

    paths, dts = {}, {}
    for kind in ("global", "local"):
        dm, mode, dt = lib.choose_diff_path(
            kind, args.n, args.M, args.L, args.b_layers
        )
        paths[kind] = (dm, mode)
        dts[kind] = dt
        print(f"{kind:6s} cost path: {dm or 'default'}/{mode}  "
              f"({dt:.3f}s per cost+grad)")
    step_cost = dts["global"] + dts["local"]
    ckpts = lib.checkpoint_steps(args.steps)
    per_trial = step_cost * (args.steps + len(ckpts) * args.dirs)
    print(f"-> ~{per_trial:.0f}s/trial, "
          f"~{per_trial * args.trials / max(args.workers, 1) / 60:.0f} min "
          f"total\n")
    self_test(paths)

    watch = lib.default_watch(args.n, args.M, args.L)
    root = np.random.SeedSequence(SEED_E2)
    jobs = []
    for s in root.spawn(args.trials):
        tr, nl = s.spawn(2)
        jobs.append((tr, nl, args.n, args.M, args.L, args.b_layers,
                     args.steps, args.lr, args.optimizer, ckpts, watch,
                     args.dirs, args.train, paths))

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(run_seed, jobs))
    else:
        results = []
        for i, j in enumerate(jobs):
            results.append(run_seed(j))
            if (i + 1) % 5 == 0 or i == 0:
                el = time.time() - t0
                print(f"  trial {i + 1}/{args.trials}  [{el:.0f}s, "
                      f"~{el / (i + 1) * args.trials:.0f}s total]", flush=True)

    stack = [np.stack([r[k] for r in results]) for k in range(13)]
    cfg = dict(vars(args))
    cfg.update(seed=SEED_E2, kind="e2",
               paths={k: (v[0] or "default", v[1]) for k, v in paths.items()})
    out = args.out or (
        f"e2_train{args.train}_n{args.n}_M{args.M}_L{args.L}"
        + ("_quick" if args.quick else "") + ".npz"
    )
    np.savez_compressed(
        out, c_train=stack[0], c_other=stack[1], dists=stack[2],
        gw_train=stack[3], gw_other=stack[4], gf_train=stack[5],
        gf_other=stack[6], probes=stack[7], nc_train=stack[8],
        nc_other=stack[9], ngf_train=stack[10], ngf_other=stack[11],
        null_probes=stack[12], checkpoints=ckpts, watch=watch,
        config=json.dumps(cfg),
    )
    print(f"\nsaved {out}  [{time.time() - t0:.0f}s]")
    print(f"final {args.train} cost: mean {stack[0][:, -1].mean():.4f}  "
          f"(other cost: {stack[1][:, -1].mean():.4f})")


if __name__ == "__main__":
    main()