"""
E1. The main event.

The question: when gradients fade during training, is it because we've
wandered too far from the identity start (geometry), because the optimizer
found a nice path (trajectory), or just because we're converging?

The trick to telling them apart is stupidly simple in hindsight. At each
checkpoint, note how far training has moved, r = ||theta - theta_0||. Then
jump to a few RANDOM points at exactly that same distance from the start
and measure gradients there too. Same displacement, no optimizer involved.
If the trajectory sits above those random points, the optimizer is doing
something clever. If it sits on them, distance is all that matters. And
because we log the cost as well, we can throw out converged runs and see
what's left.

Version 2 of this file exists because version 1 saved nothing until every
trial finished, and I lost about fifteen hours to an overnight Windows
reboot. Now every trial gets written to <out>.parts/ the moment it's done,
and re-running the same command just picks up the leftovers. Also: the
per-block drift probe costs 4^n and is switched off above n = 10, since the
scrambling story was already settled at smaller sizes.

Seeds are nested so the null directions come from their own stream --
asking for more null directions doesn't budge the trajectory at all.

Optimizer collapse test = run this a few times with different settings:
  python e1_trajectory_vs_patch.py --optimizer gd   --lr 0.01
  python e1_trajectory_vs_patch.py --optimizer adam --lr 0.001
then throw them at analyze_experiments.py --overlay.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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

SEED_E1 = 190305079
PROBE_MAX_N = 10  # the block probe costs 4^n, so above this it's just not worth it


def run_seed(job):
    (train_ss, null_ss, n, M, L, b_layers, steps, lr, opt_name,
     ckpts, watch, dirs, dm, mode) = job
    rng = np.random.default_rng(train_ss)
    nrng = np.random.default_rng(null_ss)

    b_axes, b_params, axes, p = lib.draw_instance(rng, n, M, L, b_layers)
    p0 = p.copy()
    off, sc, factory = lib.cost_def("heisenberg", n)
    cost = lib.make_cost(lib.device(n), factory, off, sc, b_axes, b_params,
                         axes, n, M, L, dm, mode)
    grad_fn = qml.grad(cost)
    opt = lib.make_optimizer(opt_name, p.shape, lr)
    do_probes = n <= PROBE_MAX_N

    C, P = len(ckpts), p.size
    costs = np.empty(steps)
    dists = np.empty(steps)
    g_watch = np.empty((steps, len(watch)), dtype=np.float32)
    g_full = np.empty((C, P), dtype=np.float32)
    probes = np.full((C, M), np.nan, dtype=np.float32)
    n_costs = np.empty((C, dirs), dtype=np.float32)
    n_g_full = np.empty((C, dirs, P), dtype=np.float32)
    n_probes = np.full((C, dirs, M), np.nan, dtype=np.float32)
    ck = {int(c): i for i, c in enumerate(ckpts)}

    for t in range(steps):
        pt = pnp.array(p, requires_grad=True)
        costs[t] = float(cost(pt))
        g = np.asarray(grad_fn(pt))
        dists[t] = float(np.linalg.norm(p - p0))
        g_watch[t] = g.ravel()[watch]

        if t in ck:
            i = ck[t]
            g_full[i] = g.ravel()
            if do_probes:
                probes[i] = lib.all_probes(p, axes, n, M, L)
            r = dists[t]
            for d in range(dirs):
                pn = lib.null_point(nrng, p0, r)
                pnt = pnp.array(pn, requires_grad=True)
                n_costs[i, d] = float(cost(pnt))
                n_g_full[i, d] = np.asarray(grad_fn(pnt)).ravel()
                if do_probes:
                    n_probes[i, d] = lib.all_probes(pn, axes, n, M, L)

        p = opt.step(p, g)

    return costs, dists, g_watch, g_full, probes, n_costs, n_g_full, n_probes

_KEYS = ("costs", "dists", "g_watch", "g_full", "probes",
         "null_costs", "null_g_full", "null_probes")


def _run_part(args):
    idx, part_path, job = args
    res = run_seed(job)
    np.savez_compressed(part_path, **dict(zip(_KEYS, res)))
    return idx


def self_test(dm, mode):
    n, M, L, b = 4, 1, 3, 2
    rng = np.random.default_rng(0)
    b_axes, b_params, axes, p = lib.draw_instance(rng, n, M, L, b)
    assert lib.all_probes(p, axes, n, M, L).max() < 1e-9, "probe != 0 at init"
    off, sc, factory = lib.cost_def("heisenberg", n)
    cost = lib.make_cost(lib.device(n), factory, off, sc, b_axes, b_params,
                         axes, n, M, L, dm, mode)
    g = np.asarray(qml.grad(cost)(pnp.array(p, requires_grad=True))).ravel()
    for k in (0, p.size - 1):
        pp, pm = p.copy().ravel(), p.copy().ravel()
        pp[k] += np.pi / 2
        pm[k] -= np.pi / 2
        gs = 0.5 * (
            float(cost(pnp.array(pp.reshape(p.shape), requires_grad=True)))
            - float(cost(pnp.array(pm.reshape(p.shape), requires_grad=True)))
        )
        assert abs(g[k] - gs) < 1e-7, f"grad mismatch at {k}"
    pn = lib.null_point(np.random.default_rng(1), p, 0.37)
    assert abs(np.linalg.norm(pn - p) - 0.37) < 1e-12, "null norm mismatch"
    print("self-tests passed (identity init, shift rule, null norms)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--L", type=int, default=33)
    ap.add_argument("--b-layers", type=int, default=None)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--optimizer", choices=("adam", "gd"), default="adam")
    ap.add_argument("--dirs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.b_layers is None:
        args.b_layers = args.n
    if args.quick:
        args.trials, args.steps, args.dirs = 6, 40, 2

    out = args.out or (
        f"e1_n{args.n}_M{args.M}_L{args.L}_{args.optimizer}_lr{args.lr}"
        + ("_quick" if args.quick else "") + ".npz"
    )
    parts_dir = out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    dm, mode, dt = lib.choose_diff_path(
        "heisenberg", args.n, args.M, args.L, args.b_layers
    )
    ckpts = lib.checkpoint_steps(args.steps)
    per_trial = dt * (args.steps + len(ckpts) * args.dirs)
    print(f"diff path: {dm or 'default'}/{mode}; {dt:.3f}s/step -> "
          f"~{per_trial:.0f}s/trial before overheads")
    if args.n > PROBE_MAX_N:
        print(f"n={args.n} > {PROBE_MAX_N}: block probes disabled (NaN)")
    self_test(dm, mode)

    e_min = lib.exact_ground_energy(args.n)
    print(f"exact ground energy (n={args.n}): {e_min:.6f}")

    watch = lib.default_watch(args.n, args.M, args.L)
    root = np.random.SeedSequence(SEED_E1)
    jobs = []
    for idx, s in enumerate(root.spawn(args.trials)):
        tr, nl = s.spawn(2)
        job = (tr, nl, args.n, args.M, args.L, args.b_layers, args.steps,
               args.lr, args.optimizer, ckpts, watch, args.dirs, dm, mode)
        jobs.append((idx, os.path.join(parts_dir, f"trial_{idx:03d}.npz"),
                     job))

    todo = [j for j in jobs if not os.path.exists(j[1])]
    done = args.trials - len(todo)
    print(f"{done}/{args.trials} trials already on disk; running {len(todo)}")

    t0 = time.time()
    if todo:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_run_part, j) for j in todo]
                for k, f in enumerate(as_completed(futs)):
                    f.result()
                    el = time.time() - t0
                    rem = el / (k + 1) * (len(todo) - k - 1)
                    print(f"  {done + k + 1}/{args.trials} done "
                          f"[{el:.0f}s elapsed, ~{rem / 60:.0f} min left]",
                          flush=True)
        else:
            for k, j in enumerate(todo):
                _run_part(j)
                el = time.time() - t0
                print(f"  {done + k + 1}/{args.trials} done [{el:.0f}s]",
                      flush=True)

    parts = [np.load(j[1]) for j in jobs]
    stack = {k: np.stack([p[k] for p in parts]) for k in _KEYS}
    cfg = dict(vars(args))
    cfg.update(e_min=e_min, seed=SEED_E1, diff_method=dm or "default",
               observable_mode=mode, kind="heisenberg")
    np.savez_compressed(
        out, checkpoints=ckpts, watch=watch, config=json.dumps(cfg), **stack
    )
    print(f"\nsaved {out}  [{time.time() - t0:.0f}s this session]")
    print(f"final cost: mean {stack['costs'][:, -1].mean():+.4f}, "
          f"E_min {e_min:+.4f}")
    print(f"parts kept in {parts_dir} (delete after verifying the npz)")
    print(f"analyze with: python analyze_experiments.py {out}")


if __name__ == "__main__":
    main()