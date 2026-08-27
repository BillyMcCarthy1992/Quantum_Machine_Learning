"""
The funnel plot, and the number the whole scaling story rests on.

Plots gradient variance against how much energy is left to go rather than
against steps. Everything collapses onto a straight line when you do that,
which is the point: variance isn't doing its own thing, it's just tracking
how far from done you are.

Then it reads off that line at the SAME RELATIVE progress for each size
(25% of |E_min|, since bigger systems have bigger energies -- comparing at
a fixed absolute gap quietly flatters the small ones) and fits how the
level drops per qubit. That per-qubit factor is the punchline.
"""

import numpy as np, json, matplotlib.pyplot as plt

files = ["e1_n6_M2_L33_adam_lr0.001.npz",
         "e1_n8_M2_L33_adam_lr0.001.npz",
         "e1_n10_M2_L33_adam_lr0.001.npz",
         "e1_n12_M2_L33_adam_lr0.001.npz"]
REL = 0.25  # measure Var where E - E_min = 25% of |E_min| for each size

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
ns, vstar = [], []
for f in files:
    d = np.load(f, allow_pickle=True)
    cfg = json.loads(str(d["config"]))
    ck = d["checkpoints"]
    gap = d["costs"][:, ck].mean(0) - cfg["e_min"]
    v = np.median(d["g_full"].astype(np.float64).var(0), axis=-1)
    a.loglog(gap, v, "o-", label=f"n={cfg['n']}")

    gstar = REL * abs(cfg["e_min"])          # per-size reference gap
    o = np.argsort(gap)
    vg = np.exp(np.interp(np.log(gstar), np.log(gap[o]), np.log(v[o])))
    a.axvline(gstar, ls=":", color=a.lines[-1].get_color(), alpha=0.5)

    m = (gap < 0.5 * gap.max()) & (gap > 0)
    s, _ = np.polyfit(np.log(gap[m]), np.log(v[m]), 1)
    ns.append(cfg["n"]); vstar.append(vg)
    print(f"n={cfg['n']}: alpha={s:.2f}, gstar={gstar:.2f}, "
          f"Var(rel gap {REL})={vg:.3e}")

a.set_xlabel("E - E_min"); a.set_ylabel("median Var"); a.legend()
b.semilogy(ns, vstar, "o-")
sl = np.polyfit(ns, np.log(vstar), 1)[0]
b.set_title(f"Var at rel. gap {REL}: factor {np.exp(sl):.3f} per qubit")
b.set_xlabel("n"); b.set_ylabel("Var at matched relative gap")
fig.tight_layout(); fig.savefig("funnel_prefactor_rel.png", dpi=150)
print(f"\nsuppression per qubit at matched RELATIVE gap: {np.exp(sl):.3f}")