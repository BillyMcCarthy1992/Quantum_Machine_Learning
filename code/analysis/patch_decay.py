"""
How much does the landscape around the starting point darken as we go?

The random-null points aren't just a control -- they ARE the landscape at a
given distance from the start, which is exactly the thing the warm-start
theory papers put bounds on. So this measures how much the null fades from
the first checkpoint to the last, for each system size.

At n=6 it barely fades at all. By n=12 it drops like a stone. That's the
fertile patch shrinking as the problem grows, which is precisely what the
theory says should happen -- nice to see it in actual numbers.

Also prints how the trajectory compares to the null at the end, which backs
up the claim that training stays below the surrounding landscape at every
size we looked at.

Run it from results/data.
"""

import json

import numpy as np

FILES = [
    "e1_n6_M2_L33_adam_lr0.001.npz",
    "e1_n8_M2_L33_adam_lr0.001.npz",
    "e1_n10_M2_L33_adam_lr0.001.npz",
    "e1_n12_M2_L33_adam_lr0.001.npz",
]

print(f"{'n':>3} {'r_max':>6} {'null(0)':>10} {'null(end)':>10} "
      f"{'decay':>7} {'traj/null(end)':>15}")
for f in FILES:
    try:
        d = np.load(f, allow_pickle=True)
    except FileNotFoundError:
        print(f"  missing: {f}")
        continue
    cfg = json.loads(str(d["config"]))
    K, C, D, P = d["null_g_full"].shape
    ngf = d["null_g_full"].astype(np.float64)
    gf = d["g_full"].astype(np.float64)

    null_v = np.median(
        ngf.transpose(0, 2, 1, 3).reshape(K * D, C, P).var(axis=0), axis=-1
    )
    traj_v = np.median(gf.var(axis=0), axis=-1)
    r_max = d["dists"][:, d["checkpoints"]].mean(axis=0)[-1]

    print(f"{cfg['n']:>3} {r_max:>6.2f} {null_v[0]:>10.3e} {null_v[-1]:>10.3e} "
          f"{null_v[0] / null_v[-1]:>7.2f} {traj_v[-1] / null_v[-1]:>15.4f}")

print("\ndecay = null variance at first checkpoint / at last checkpoint")
print("traj/null(end) < 1 means the trajectory sits below the landscape null")
