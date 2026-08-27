# Convergence Funnels and Fertile Valleys

Trainability of identity-block-initialized quantum neural networks: does
initialization-time protection against vanishing gradients survive training?

Machine Learning research project — Billy McCarthy, 2026.

## Layout

```
code/
  idblock_lib.py              shared library: circuits, costs, probes,
                              optimizers, seeding, diff-path selection
  experiments/
    grant_fig1a_random.py     E0 anchor (i), random-init half
    grant_fig1a_identity.py   E0 anchor (i), identity-block half
    grant_fig4b_vqe.py        E0 anchor (ii), 7-qubit Heisenberg VQE
    e1_trajectory_vs_patch.py E1 headline: trajectory vs matched-norm null
    e2_cost_locality.py       E2: global vs local cost, cross-measured
    e3_scaling_sweep.py       E3: (n, L, M) grid, resumable
  analysis/
    analyze_experiments.py    4-panel per-run figures; --overlay mode
    funnel_rel.py             funnel law + coefficient vs n (relative gap)
    funnel_ci.py              bootstrap CIs on exponents and coefficients
    funnel_collapse_optimizers.py  optimizer collapse in funnel coordinates
    depth_analysis.py         depth control at n = 8
    depth_n12.py              depth control at n = 12
  verification/
    verify_numpy_*.py         independent pure-numpy twin simulators used to
                              validate every construction before production
  analysis/
    patch_decay.py            how the fertile region shrinks with n
results/
  data/                       final .npz datasets (one per configuration)
  figures/                    figures used in the report
report/
  report.tex + figures        the paper, ready to drop into Overleaf
docs/
  prereg_n12_predictions.md   pre-registered n = 12 predictions (recorded
                              before the run; the miss is analysed in the report)
```

## Key results (see the report for detail)

- Both Grant et al. anchors reproduced quantitatively: random-init plateau at
  0.503/qubit vs. theory 2^(1-n); identity-block line flat at the analytically
  derived 0.02203.
- No plateau reemergence: against a matched-displacement null and
  cost-conditioning, gradient variance decays only as the loss gap closes.
- Convergence funnel: Var ≈ c(n)·(E − E_min)^α, optimizer-independent.
- c(n) falls ≈0.48/qubit through n = 10 (the barren-plateau factor); the
  pre-registered n = 12 prediction missed, traced to a finite-depth confound.
- Cost locality inverts the trajectory/landscape relation: local costs give
  funnels, global costs give fertile valleys (~40× above a barren null).

## Reproducing

Everything is deterministic from a single integer seed per experiment.
Scripts resolve their own imports, so they can be launched from anywhere; they
read and write data files in the **working directory**, so run them from
`results/data`.

PowerShell:

```powershell
cd results\data

# experiments (each prints a runtime projection before starting)
python3.11 ..\..\code\experiments\e1_trajectory_vs_patch.py --n 8 --trials 100 --workers 6
python3.11 ..\..\code\experiments\e2_cost_locality.py --train local --trials 100 --workers 6
python3.11 ..\..\code\experiments\e3_scaling_sweep.py --list

# analysis
python3.11 ..\..\code\analysis\analyze_experiments.py e1_n8_M2_L33_adam_lr0.001.npz
python3.11 ..\..\code\analysis\funnel_rel.py
```

Output `.png` files land in the working directory (`results/data`); move them
to `results/figures` afterwards. Long runs checkpoint per trial into
`<name>.npz.parts/` and resume by re-running the identical command.

Validation (no PennyLane required, pure numpy):

```powershell
cd code\verification
python3.11 verify_numpy_random.py 2 4 6
python3.11 verify_numpy_identity.py 2 4 6
python3.11 verify_numpy_vqe.py
python3.11 verify_numpy_e1e2.py
```

## Notes

- `code/experiments/grant_fig4b_vqe.py` picks its differentiation path at
  production size and prints a runtime projection before committing.
- Block-deviation probes cost O(4^n) and are disabled automatically above
  n = 10; those fields are NaN in the n = 12 datasets.
- Requires: Python 3.11, pennylane, pennylane-lightning, numpy, matplotlib.
