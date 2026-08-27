"""
E3. The grid-fill job.

Same measurement as E1, smaller ensembles, run over a bunch of (n, L, M)
combinations so we can see what depends on size versus shape. The useful
bit is the depth-matched trio: three different ways of splitting 60 layers
of total depth, so if results differ it's the SPLIT doing it, not the depth.

Every config writes its own E1-format npz and gets skipped if the file is
already sitting there, which means you can run this in dribs and drabs
whenever the laptop is otherwise idle and never lose progress.

  python e3_scaling_sweep.py --list          what is done, what isn't
  python e3_scaling_sweep.py --only 3        just that one
  python e3_scaling_sweep.py --workers 6     chew through the rest
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# --- so this runs from wherever, not just from one magic folder ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_os.path.dirname(_HERE), _HERE):   # code/ , code/experiments/
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import idblock_lib as lib
from e1_trajectory_vs_patch import run_seed

SEED_E3 = 190305081

CONFIGS = [
    dict(n=n, L=L, M=M)
    for n in (4, 6, 8, 10, 12)
    for (L, M) in ((15, 2), (30, 1), (10, 3))
] + [dict(n=8, L=15, M=1), dict(n=8, L=15, M=4)]


def out_name(c):
    return f"e3_n{c['n']}_M{c['M']}_L{c['L']}.npz"


def run_config(idx, cfg, trials, steps, dirs, lr, workers):
    n, M, L = cfg["n"], cfg["M"], cfg["L"]
    b_layers = n
    dm, mode, dt = lib.choose_diff_path("heisenberg", n, M, L, b_layers)
    ckpts = lib.checkpoint_steps(steps)
    per_trial = dt * (steps + len(ckpts) * dirs)
    print(f"[{idx}] n={n} M={M} L={L}: path {dm or 'default'}/{mode}, "
          f"~{per_trial * trials / max(workers, 1) / 60:.0f} min")

    e_min = lib.exact_ground_energy(n)
    watch = lib.default_watch(n, M, L)
    root = np.random.SeedSequence(SEED_E3).spawn(len(CONFIGS))[idx]
    jobs = []
    for s in root.spawn(trials):
        tr, nl = s.spawn(2)
        jobs.append((tr, nl, n, M, L, b_layers, steps, lr, "adam",
                     ckpts, watch, dirs, dm, mode))

    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(run_seed, jobs))
    else:
        results = [run_seed(j) for j in jobs]

    stack = [np.stack([r[k] for r in results]) for k in range(8)]
    meta = dict(cfg)
    meta.update(trials=trials, steps=steps, dirs=dirs, lr=lr,
                optimizer="adam", b_layers=b_layers, e_min=e_min,
                seed=SEED_E3, config_index=idx, kind="heisenberg",
                diff_method=dm or "default", observable_mode=mode)
    np.savez_compressed(
        out_name(cfg), costs=stack[0], dists=stack[1], g_watch=stack[2],
        g_full=stack[3], probes=stack[4], null_costs=stack[5],
        null_g_full=stack[6], null_probes=stack[7], checkpoints=ckpts,
        watch=watch, config=json.dumps(meta),
    )
    print(f"[{idx}] saved {out_name(cfg)}  [{time.time() - t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dirs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for i, c in enumerate(CONFIGS):
            done = "done" if os.path.exists(out_name(c)) else "todo"
            print(f"[{i:2d}] n={c['n']:2d} M={c['M']} L={c['L']:2d}  {done}")
        return

    todo = (
        [args.only]
        if args.only is not None
        else [i for i, c in enumerate(CONFIGS)
              if not os.path.exists(out_name(c))]
    )
    print(f"{len(todo)} config(s) to run\n")
    for i in todo:
        run_config(i, CONFIGS[i], args.trials, args.steps, args.dirs,
                   args.lr, args.workers)


if __name__ == "__main__":
    main()