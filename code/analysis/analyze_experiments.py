"""
Turns a results file into the pictures.

Point it at an E1/E3 npz and you get four panels:
  (a) the cost coming down, with the exact ground energy for reference
  (b) gradient variance against step -- the view that looks convincing and
      tells you almost nothing, since it depends on your learning rate
  (c) THE ONE THAT MATTERS: variance against distance from the start, with
      the trajectory, the matched random null, and the trajectory with
      converged runs thrown out, all on the same axes with error bands
  (d) how far the blocks have drifted from the identity, path vs null

An E2 file gets the cost curves plus the cross-measured 2x2 instead.

--overlay takes several files and stacks them, which is how the optimizer
collapse plot gets made.

Heads up on --delta: it's the "count this run as still training" threshold.
For the Heisenberg runs it's an energy above E_min, so 1.0 is sensible. For
E2 the costs live in [0, 1], so pass something like 0.1 or the conditioned
curve comes out empty.

  python analyze_experiments.py e1_n8_M2_L33_adam_lr0.001.npz
  python analyze_experiments.py e2_trainlocal_n8_M2_L33.npz --delta 0.1
  python analyze_experiments.py --overlay e1_*adam* e1_*gd*
"""

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    d = np.load(path, allow_pickle=True)
    cfg = json.loads(str(d["config"]))
    return d, cfg


def med_var_ci(g, boot=200, seed=0):
    """g: (K, C, P) or (K, D, C, P)-style with samples on axis 0 (pool any
    extra sample axes into axis 0 first). Returns (median-over-P variance,
    lo, hi) per C via bootstrap over samples."""
    rng = np.random.default_rng(seed)
    K = g.shape[0]
    point = np.median(g.var(axis=0), axis=-1)
    if K < 4:
        return point, point, point
    bs = np.empty((boot,) + point.shape)
    for b in range(boot):
        idx = rng.integers(0, K, K)
        bs[b] = np.median(g[idx].var(axis=0), axis=-1)
    return point, np.percentile(bs, 16, axis=0), np.percentile(bs, 84, axis=0)


def traj_null_panel(ax, d, cost_key, gf_key, ngf_key, ckpts, delta, e_ref,
                    label_prefix=""):
    gf = d[gf_key].astype(np.float64)          # (K, C, P)
    ngf = d[ngf_key].astype(np.float64)        # (K, C, D, P)
    K, C, D, P = ngf.shape
    x = d["dists"][:, ckpts].mean(axis=0)      # mean distance per checkpoint

    v, lo, hi = med_var_ci(gf)
    ax.plot(x, v, "o-", color="C0", label=f"{label_prefix}trajectory")
    ax.fill_between(x, lo, hi, color="C0", alpha=0.25)

    nflat = ngf.transpose(0, 2, 1, 3).reshape(K * D, C, P)
    v2, lo2, hi2 = med_var_ci(nflat)
    ax.plot(x, v2, "s-", color="C3", label=f"{label_prefix}matched null")
    ax.fill_between(x, lo2, hi2, color="C3", alpha=0.25)

    # cost-conditioned trajectory (exclude converged runs)
    if delta is not None:
        costs = d[cost_key][:, ckpts]          # (K, C)
        thr = (e_ref + delta) if e_ref is not None else delta
        vc = np.full(C, np.nan)
        for c in range(C):
            mask = costs[:, c] > thr
            if mask.sum() >= 5:
                vc[c] = np.median(gf[mask, c].var(axis=0), axis=-1)
        ax.plot(x, vc, "^--", color="C2",
                label=f"{label_prefix}traj (unconverged only)")

    ax.set_yscale("log")
    ax.set_xlabel(r"$\|\theta-\theta_0\|_2$")
    ax.set_ylabel(r"median$_p$ Var[$\partial_p C$]")
    ax.legend(fontsize=8)


def analyze_e1(path, delta):
    d, cfg = load(path)
    ckpts = d["checkpoints"]
    e_min = cfg.get("e_min")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    (a, b), (c, e) = ax

    costs = d["costs"]
    a.plot(costs.mean(axis=0), color="C0")
    a.fill_between(range(costs.shape[1]),
                   costs.mean(0) - costs.std(0),
                   costs.mean(0) + costs.std(0), alpha=0.3)
    if e_min is not None:
        a.axhline(e_min, ls="--", color="green", label=f"E_min={e_min:.3f}")
        a.legend()
    a.set_xlabel("step"); a.set_ylabel("cost"); a.set_title("(a) training")

    vw = d["g_watch"].astype(np.float64).var(axis=0)
    b.semilogy(np.median(vw, axis=1), color="C0")
    b.set_xlabel("step"); b.set_ylabel("median watch-set Var")
    b.set_title("(b) variance vs step")

    traj_null_panel(c, d, "costs", "g_full", "null_g_full", ckpts, delta,
                    e_min)
    c.set_title("(c) trajectory vs patch null")

    x = d["dists"][:, ckpts].mean(axis=0)
    pr = d["probes"].astype(np.float64)        # (K, C, M)
    npr = d["null_probes"].astype(np.float64)  # (K, C, D, M)
    e.plot(x, pr.mean(axis=(0, 2)), "o-", color="C0", label="trajectory")
    e.plot(x, npr.mean(axis=(0, 2, 3)), "s-", color="C3", label="null")
    e.set_xlabel(r"$\|\theta-\theta_0\|_2$")
    e.set_ylabel(r"block deviation $1-|\mathrm{Tr}\,U_m|/2^n$")
    e.set_title("(d) scrambling: trajectory vs null"); e.legend()

    fig.suptitle(os.path.basename(path))
    fig.tight_layout()
    out = os.path.splitext(path)[0] + "_analysis.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def analyze_e2(path, delta):
    d, cfg = load(path)
    ckpts = d["checkpoints"]
    train = cfg["train"]
    other = "local" if train == "global" else "global"

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    a, b, c = ax
    for key, lab, col in (("c_train", f"C_{train[0].upper()} (trained)",
                           "C0"), ("c_other", f"C_{other[0].upper()}", "C3")):
        m = d[key].mean(axis=0)
        a.plot(m, color=col, label=lab)
        a.fill_between(range(len(m)), m - d[key].std(0), m + d[key].std(0),
                       color=col, alpha=0.25)
    a.set_xlabel("step"); a.set_ylabel("cost"); a.legend()
    a.set_title(f"(a) training under {train} cost")

    traj_null_panel(b, d, "c_train", "gf_train", "ngf_train", ckpts,
                    delta, None)
    b.set_title(f"(b) {train}-cost gradients")
    traj_null_panel(c, d, "c_train", "gf_other", "ngf_other", ckpts,
                    delta, None)
    c.set_title(f"(c) {other}-cost gradients (cross-measured)")

    fig.suptitle(os.path.basename(path))
    fig.tight_layout()
    out = os.path.splitext(path)[0] + "_analysis.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def overlay(paths, delta):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for i, path in enumerate(paths):
        d, cfg = load(path)
        lab = f"{cfg.get('optimizer', '?')} lr={cfg.get('lr', '?')}"
        vw = np.median(d["g_watch"].astype(np.float64).var(axis=0), axis=1)
        a.semilogy(vw, color=f"C{i}", label=lab)
        xd = d["dists"].mean(axis=0)
        b.semilogy(xd, vw, color=f"C{i}", label=lab)
    a.set_xlabel("step"); a.set_ylabel("median watch-set Var")
    a.set_title("vs step"); a.legend(fontsize=8)
    b.set_xlabel(r"mean $\|\theta-\theta_0\|_2$")
    b.set_title("vs distance (collapse test)"); b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("overlay_collapse.png", dpi=150)
    print("saved overlay_collapse.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--delta", type=float, default=1.0,
                    help="unconverged threshold: cost > E_min + delta "
                         "(heisenberg) or cost > delta (E2)")
    args = ap.parse_args()

    if args.overlay:
        overlay(args.files, args.delta)
        return
    for path in args.files:
        _, cfg = load(path)
        if cfg.get("kind") == "e2":
            analyze_e2(path, args.delta)
        else:
            analyze_e1(path, args.delta)


if __name__ == "__main__":
    main()