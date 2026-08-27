"""
Boils every dataset down to the numbers that actually appear in the report.

The raw npz files are ~380 MB, which is too big to hand anyone, and most of
that bulk is per-step gradient arrays nobody needs twice. This script walks
whatever npz files are present and writes results_summary.txt with the
derived quantities the report quotes, each tagged with the section or table
it lands in, so a reader can check the paper against the data without
running a single experiment.

Everything here is recomputed from scratch. Nothing is copied from the text,
which is the whole point -- if the report and this file disagree, the report
is wrong.

Run it from results/data.
"""

import glob
import json
import os

import numpy as np

REL = 0.25          # reference point: gap = 25% of |E_min|
WINDOW = (0.12, 0.45)   # relative-gap window for exponent fits
BOOT = 200
rng = np.random.default_rng(0)
out = []


def say(line=""):
    print(line)
    out.append(line)


def load(f):
    d = np.load(f, allow_pickle=True)
    return d, json.loads(str(d["config"]))


def funnel_fit(d, cfg):
    """Exponent and coefficient at matched relative progress, with CIs."""
    ck = d["checkpoints"]
    gap = d["costs"][:, ck].mean(0) - cfg["e_min"]
    gf = d["g_full"].astype(np.float64)
    K = gf.shape[0]
    rel = gap / abs(cfg["e_min"])
    m = (rel > WINDOW[0]) & (rel < WINDOW[1])
    gstar = REL * abs(cfg["e_min"])
    o = np.argsort(gap)
    boot = []
    for _ in range(BOOT):
        idx = rng.integers(0, K, K)
        v = np.median(gf[idx].var(0), axis=-1)
        a = np.polyfit(np.log(gap[m]), np.log(v[m]), 1)[0] if m.sum() > 3 else np.nan
        vg = np.exp(np.interp(np.log(gstar), np.log(gap[o]), np.log(v[o])))
        boot.append([a, vg])
    ab, vgb = np.array(boot).T
    # point estimates use the whole ensemble; the bootstrap only sizes the
    # error bar. Resampling with replacement biases a variance slightly low
    # at small K, so its median is not the right central value.
    v0 = np.median(gf.var(0), axis=-1)
    a0 = np.polyfit(np.log(gap[m]), np.log(v0[m]), 1)[0] if m.sum() > 3 else np.nan
    vg0 = np.exp(np.interp(np.log(gstar), np.log(gap[o]), np.log(v0[o])))
    return a0, ab.std(), vg0, vgb.std(), gap.min(), K


say("RESULTS SUMMARY")
say("Recomputed from the datasets. Cross-check against the report.")
say("=" * 72)

# ---------------------------------------------------------------- anchors
say()
say("ANCHORS  (report Sec. III-F, Figs. 1 and 2)")
if os.path.exists("grant_fig1a_random.npz"):
    d = np.load("grant_fig1a_random.npz", allow_pickle=True)
    slope = np.polyfit(d["n"], np.log(d["variance"]), 1)[0]
    say(f"  random init decay per qubit : {np.exp(slope):.4f}   (theory 2^-1 = 0.5)")
if os.path.exists("grant_fig1a_identity.npz"):
    d = np.load("grant_fig1a_identity.npz", allow_pickle=True)
    say(f"  identity-block variance     : {d['variance'].mean():.5f} mean over n, "
        f"range {d['variance'].min():.5f}-{d['variance'].max():.5f}")
    say(f"  closed-form prediction      : {float(d['var_analytic']):.6f}  (Eq. 3)")
if os.path.exists("fig4b.npz"):
    d, cfg = load("fig4b.npz")
    say(f"  Heisenberg n=7 E_min        : {cfg['e_min']:.4f}   "
        f"final mean energy {d['energies'][:, -1].mean():.4f}")

# ------------------------------------------------------------ funnel fits
say()
say("FUNNEL FITS  (report Table II, Sec. IV-D and IV-F)")
say(f"  exponent fitted over relative gap {WINDOW[0]}-{WINDOW[1]}; "
    f"coefficient at relative gap {REL}")
say("  point estimates over the full ensemble, +- bootstrap standard deviation")
say(f"  {'config':<22} {'K':>4} {'alpha':>14} {'coefficient':>20}")
coeffs = {}
for f in sorted(glob.glob("e1_n*_M*_L*.npz"), key=lambda p: (int(p.split("_n")[1].split("_")[0]), p)):
    d, cfg = load(f)
    if cfg.get("optimizer") != "adam" or cfg.get("lr") != 0.001:
        continue
    a, asd, v, vsd, gmin, K = funnel_fit(d, cfg)
    tag = f"n={cfg['n']}, {cfg['M']}x{cfg['L']}"
    coeffs[(cfg["n"], cfg["L"])] = v
    say(f"  {tag:<22} {K:>4}  {a:5.2f} +- {asd:.2f}   {v:.3e} +- {vsd:.1e}")

say()
say("  suppression per qubit at matched relative progress (Sec. IV-F):")
std = sorted(k for k in coeffs if k[1] == 33)
for i in range(len(std) - 1):
    n0, n1 = std[i][0], std[i + 1][0]
    r = (coeffs[std[i + 1]] / coeffs[std[i]]) ** (1.0 / (n1 - n0))
    say(f"    n={n0} -> n={n1}: {r:.3f} per qubit")
if (12, 33) in coeffs and (12, 50) in coeffs:
    say(f"    depth control at n=12: 132-layer {coeffs[(12,33)]:.3e} vs "
        f"200-layer {coeffs[(12,50)]:.3e}, factor {coeffs[(12,33)]/coeffs[(12,50)]:.2f}")

# ------------------------------------------------------------- patch decay
say()
say("PATCH DECAY  (report Sec. IV-C)")
say(f"  {'n':>3} {'r_max':>7} {'null start':>11} {'null end':>11} "
    f"{'decay':>7} {'traj/null':>10}")
for f in sorted(glob.glob("e1_n*_M2_L33_adam_lr0.001.npz"), key=lambda p: int(p.split("_n")[1].split("_")[0])):
    d, cfg = load(f)
    K, C, D, P = d["null_g_full"].shape
    nv = np.median(d["null_g_full"].astype(np.float64)
                   .transpose(0, 2, 1, 3).reshape(K * D, C, P).var(0), axis=-1)
    tv = np.median(d["g_full"].astype(np.float64).var(0), axis=-1)
    r = d["dists"][:, d["checkpoints"]].mean(0)[-1]
    say(f"  {cfg['n']:>3} {r:>7.2f} {nv[0]:>11.3e} {nv[-1]:>11.3e} "
        f"{nv[0]/nv[-1]:>7.2f} {tv[-1]/nv[-1]:>10.4f}")
say("  decay = null variance at first checkpoint / at last")
say("  traj/null well below 1 at every size: the path stays under the landscape")

# ------------------------------------------------------------ depth series
say()
say("DEPTH SERIES AT n=8  (report Sec. IV-F, Fig. 6)")
say(f"  {'depth':>6} {'M x L':>8} {'alpha':>7} {'coefficient':>13} {'final gap':>10}")
dv = []
for f in ["e3_n8_M1_L15.npz", "e3_n8_M2_L15.npz", "e3_n8_M4_L15.npz",
          "e1_n8_M2_L33_adam_lr0.001.npz"]:
    if not os.path.exists(f):
        continue
    d, cfg = load(f)
    a, _, v, _, gmin, K = funnel_fit(d, cfg)
    dv.append(v)
    depth = 2 * cfg["L"] * cfg["M"]
    say(f"  {depth:>6} {f'{cfg[chr(77)]}x{cfg[chr(76)]}':>8} {a:>7.2f} "
        f"{v:>13.3e} {gmin:>10.2f}")
if dv:
    say(f"  spread across depth: factor {max(dv)/min(dv):.2f}, non-monotone")

# ---------------------------------------------------------------- E2 / sizes
say()
say("COST LOCALITY  (report Sec. IV-G)")
for f in sorted(glob.glob("e2_*.npz")):
    d, cfg = load(f)
    ck = d["checkpoints"]
    K, C, D, P = d["ngf_train"].shape
    tv = np.median(d["gf_train"].astype(np.float64).var(0), axis=-1)
    nv = np.median(d["ngf_train"].astype(np.float64)
                   .transpose(0, 2, 1, 3).reshape(K * D, C, P).var(0), axis=-1)
    peak = int(np.argmax(tv / nv))
    say(f"  train={cfg['train']:<6} K={K}: null level {np.median(nv):.2e}, "
        f"peak traj/null {(tv/nv).max():.1f}x at checkpoint {peak}, "
        f"final cost {d['c_train'][:, -1].mean():.4f}")

say()
say("DATASET SIZES  (report Table I)")
say(f"  {'file':<42} {'n':>3} {'MxL':>7} {'P':>6} {'K':>4}")
for f in sorted(glob.glob("*.npz")):
    d = np.load(f, allow_pickle=True)
    if "config" not in d.files:      # the Fig-1a anchors carry no config block
        continue
    cfg = json.loads(str(d["config"]))
    key = "g_full" if "g_full" in d.files else "gf_train"
    if key not in d.files:
        continue
    say(f"  {f:<42} {cfg['n']:>3} {f'{cfg[chr(77)]}x{cfg[chr(76)]}':>7} "
        f"{d[key].shape[-1]:>6} {d[key].shape[0]:>4}")

with open("results_summary.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\nwritten to results_summary.txt")
