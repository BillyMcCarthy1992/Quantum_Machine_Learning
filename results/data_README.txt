The raw .npz datasets (about 380 MB) are not in this archive.

results_summary.txt holds every derived number the report quotes, recomputed
straight from those datasets, tagged with the report section it belongs to.
That covers checking the paper's arithmetic without running anything.

If you want the datasets themselves, they regenerate exactly from their
seeds. See the README for commands. Each experiment script prints a runtime
estimate before it commits to anything, and long runs checkpoint per trial
so they can be interrupted and resumed. The analysis scripts in
code/analysis expect the .npz files to live in this folder (results/data);
create it and run the experiments into it to reproduce the figures directly.

Figures made from those datasets are in results/figures, and the subset used
in the paper is duplicated in report/ alongside report.tex.
