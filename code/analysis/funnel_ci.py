"""
Same idea as funnel_rel.py but with error bars, and with the exponent
fitted over the same relative window for every size so the comparison is
actually fair.

Bootstraps over trials, 500 resamples, prints 16-84% intervals. This is the
script that settled whether the n=12 miss was real or just noise. It was
real.
"""

import numpy as np, json
rng = np.random.default_rng(0)
for f in ["e1_n6_M2_L33_adam_lr0.001.npz", "e1_n8_M2_L33_adam_lr0.001.npz",
          "e1_n10_M2_L33_adam_lr0.001.npz", "e1_n12_M2_L33_adam_lr0.001.npz"]:
    d = np.load(f, allow_pickle=True); cfg = json.loads(str(d["config"]))
    ck = d["checkpoints"]; K = d["g_full"].shape[0]
    gap = d["costs"][:, ck].mean(0) - cfg["e_min"]
    gf = d["g_full"].astype(np.float64)
    rel = gap / abs(cfg["e_min"])
    m = (rel > 0.12) & (rel < 0.45)
    gstar = 0.25 * abs(cfg["e_min"]); o = np.argsort(gap)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, K, K)
        v = np.median(gf[idx].var(0), axis=-1)
        boot.append([np.polyfit(np.log(gap[m]), np.log(v[m]), 1)[0],
                     np.exp(np.interp(np.log(gstar), np.log(gap[o]),
                                      np.log(v[o])))])
    a, vg = np.array(boot).T
    print(f"n={cfg['n']} (K={K}): alpha={np.median(a):.2f} "
          f"[{np.percentile(a,16):.2f},{np.percentile(a,84):.2f}]  "
          f"Var(gstar)={np.median(vg):.3e} "
          f"[{np.percentile(vg,16):.3e},{np.percentile(vg,84):.3e}]")