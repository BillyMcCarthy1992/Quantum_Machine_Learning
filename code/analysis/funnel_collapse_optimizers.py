"""
Four training runs -- two optimizers, two learning rates each -- drawn
against remaining energy instead of steps.

They land on top of each other. That's the evidence that the per-step decay
rate everyone quotes is really just an artefact of how fast your optimizer
happens to descend.
"""

import numpy as np, json, matplotlib.pyplot as plt
files = ["e1_n8_M2_L33_adam_lr0.001.npz", "e1_n8_M2_L33_gd_lr0.01.npz",
         "e1_n8_M2_L33_gd_lr0.003.npz", "e1_n8_M2_L33_adam_lr0.0003.npz"]
for f in files:
    d = np.load(f, allow_pickle=True); cfg = json.loads(str(d["config"]))
    ck = d["checkpoints"]
    gap = d["costs"][:, ck].mean(0) - cfg["e_min"]
    v = np.median(d["g_full"].astype(np.float64).var(0), axis=-1)
    plt.loglog(gap, v, "o-", label=f"{cfg['optimizer']} lr={cfg['lr']}")
plt.xlabel("E - E_min"); plt.ylabel("median Var"); plt.legend()
plt.savefig("funnel_collapse.png", dpi=150)