"""
The same depth question, but at n=12 where the scaling law went weird.

Compares the standard 132-layer run against the deeper 200-layer one at
matched relative progress. Unlike n=8, this one moves -- a lot -- which is
what makes the "we were comparing circuits that hadn't scrambled yet"
explanation the likely one.
"""

import numpy as np, json
rng = np.random.default_rng(0)
for f in ["e1_n12_M2_L33_adam_lr0.001.npz",   # depth 132
          "e1_n12_M2_L50_adam_lr0.001.npz"]:  # depth 200
    d = np.load(f, allow_pickle=True); cfg = json.loads(str(d["config"]))
    depth = 2 * cfg["L"] * cfg["M"]; K = d["g_full"].shape[0]
    ck = d["checkpoints"]
    gap = d["costs"][:, ck].mean(0) - cfg["e_min"]
    gf = d["g_full"].astype(np.float64)
    gstar = 0.25 * abs(cfg["e_min"]); o = np.argsort(gap)
    boot = []
    for _ in range(400):
        idx = rng.integers(0, K, K)
        v = np.median(gf[idx].var(0), axis=-1)
        boot.append(np.exp(np.interp(np.log(gstar), np.log(gap[o]),
                                     np.log(v[o]))))
    print(f"depth {depth} (L={cfg['L']}, K={K}): Var(gstar)={np.median(boot):.3e} "
          f"[{np.percentile(boot,16):.3e},{np.percentile(boot,84):.3e}]")