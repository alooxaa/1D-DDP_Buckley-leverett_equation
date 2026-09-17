# Weighted-loss series: Adam only on 2 GPUs

The earlier series of the experiment. It uses the same network, the same
collocation points, the same optimizer settings and the same metrics as the
unweighted series in the repository root, with a single difference: the three
loss terms are combined with fixed weights,

```
loss = 3.0 * loss_pde + 5.0 * loss_ic + 1.0 * loss_bc
```

chosen beforehand by a grid search. The script is `train_ddp_weighted.py`; the
root-level `train_ddp.py` is the same code with the plain sum instead.

## Why this series is kept

Its practical result is the cheapest configuration in the whole project that
still gives a competitive error. Mini-batch Adam on two GPUs, with no L-BFGS
stage at all, reaches L2 about 0.06 to 0.07 in roughly 20 s at 5000 updates
and 60 s at 15000 updates. The sequential two-stage scheme of the same series
needed 1414 s and 1029 s for L2 0.0655 and 0.0705. For a quick solution of
acceptable accuracy this is the mode to use.

## Contents

```
train_ddp_weighted.py     the sweep script (English, weighted loss)
5000/                     sweep with 5000 Adam updates
15000/                    sweep with 15000 Adam updates
```

Each sweep directory holds one subdirectory per scaling rule with one
`bs<N>/model_result.json` per batch size, plus `results_all.csv` and
`time_summary.csv` across the whole sweep and `baseline_reference.json` for
the single-GPU full-batch run at the same number of updates. Trained weights
and per-configuration plots are not tracked; the sweep is reproducible with
the script above.

## Scaling rules

With `L0 = 1e-3` and `K = 2` GPUs:

| Rule | Formula |
|---|---|
| `baseline` | `lr = L0` |
| `linearK` | `lr = L0 * K` |
| `sqrtK` | `lr = L0 * sqrt(K)` |

Batch sizes per GPU: 32 to 4096, and additionally 8192 in the 5000-update
sweep.

## Metrics

`l2_error` is the mean relative L2 over the four time levels 0.25, 0.5, 0.75
and 1.0, as measured during training. `l2_mean` is the same error averaged over
100 time levels from 0.01 to 1, the headline figure used everywhere in the
repository. `l_inf` and `mass_balance_rel_mean` are defined in the root README.
