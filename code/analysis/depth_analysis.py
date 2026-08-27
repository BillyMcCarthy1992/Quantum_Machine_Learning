"""
Does making the circuit deeper change the funnel? (At n=8: no.)

Reads the depth series at fixed width, pulls the funnel level at matched
progress out of each, and plots level against total depth. Four depths
spanning 4.4x, and the answer wobbles by about 20% with no clear direction,
which is nothing next to what changing n does.
"""

import numpy as np, json, matplotlib.pyplot as plt

files = ["e3_n8_M1_L15.npz",              # depth  30
         "e3_n8_M2_L15.npz",              # depth  60
         "e3_n8_M4_L15.npz",              # depth 120
         "e1_n8_M2_L33_adam_lr0.001.npz"] # depth 132 (headline)
REL = 0.25
rng = np.random.default_rng(0)

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
depths, vstars, los, his = [], [], [], []
for f in files:
    d = np.load(f, allow_pickle=True); cfg = json.loads(str(d["config"]))
    depth = 2 * cfg["L"] * cfg["M"]
    ck = d["checkpoints"]; K = d["g_full"].shape[0]
    gap = d["costs"][:, ck].mean(0) - cfg["e_min"]
    gf = d["g_full"].astype(np.float64)
    v = np.median(gf.var(0), axis=-1)
    a.loglog(gap, v, "o-", label=f"depth {depth} (M={cfg['M']}, L={cfg['L']})")

    gstar = REL * abs(cfg["e_min"]); o = np.argsort(gap)
    rel = gap / abs(cfg["e_min"]); m = (rel > 0.12) & (rel < 0.45)
    alpha = np.polyfit(np.log(gap[m]), np.log(v[m]), 1)[0]
    boot = []
    for _ in range(400):
        idx = rng.integers(0, K, K)
        vb = np.median(gf[idx].var(0), axis=-1)
        boot.append(np.exp(np.interp(np.log(gstar), np.log(gap[o]),
                                     np.log(vb[o]))))
    vg = np.median(boot)
    depths.append(depth); vstars.append(vg)
    los.append(np.percentile(boot, 16)); his.append(np.percentile(boot, 84))
    print(f"depth {depth:3d} (M={cfg['M']}, L={cfg['L']}, K={K}): "
          f"alpha={alpha:.2f}  Var(gstar)={vg:.3e} "
          f"[{np.percentile(boot,16):.3e},{np.percentile(boot,84):.3e}]  "
          f"final gap={gap.min():.2f}")

a.axvline(REL * abs(cfg["e_min"]), ls=":", color="gray")
a.set_xlabel("E - E_min"); a.set_ylabel("median Var"); a.legend(fontsize=8)
a.set_title("funnels at n=8, four depths")
o = np.argsort(depths)
b.errorbar(np.array(depths)[o], np.array(vstars)[o],
           yerr=[np.array(vstars)[o] - np.array(los)[o],
                 np.array(his)[o] - np.array(vstars)[o]], fmt="o-")
b.set_xlabel("total depth 2LM"); b.set_ylabel(f"Var at rel. gap {REL}")
b.set_yscale("log")
b.set_title("does the funnel coefficient depend on depth?")
fig.tight_layout(); fig.savefig("depth_dependence.png", dpi=150)
print("saved depth_dependence.png")