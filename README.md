# Parallel PINN for the 1D Buckley-Leverett problem

A physics-informed neural network (PINN) that solves the one-dimensional
Buckley-Leverett equation of two-phase flow in porous media, trained in two
stages (Adam, then L-BFGS) and parallelised across GPUs with PyTorch
DistributedDataParallel.

The repository answers one practical question: **does data parallelism help a
small PINN, and where exactly does the benefit come from?** The short answer is
that the Adam stage does not speed up at all on two cards, yet the full
two-stage pipeline still finishes faster than the sequential one, because the
mini-batch regime leaves L-BFGS with an easier starting point.

## The problem

Water displaces oil in a homogeneous 1D reservoir. In dimensionless form the
saturation `s(x, t)` obeys a scalar conservation law with a non-convex flux,

```
ds/dt + df/ds * ds/dx - eps * d2s/dx2 = 0,   x in [0, 1], t in [0, 1]
f(s) = s^2 / (s^2 + (1 - s)^2 / M),          M = 1
s(x, 0) = 0,   s(0, t) = 1
```

Because `f` is non-convex the solution develops a shock, and a network trained
on the pointwise residual of the inviscid equation converges to a non-entropic
solution or fails to converge at all. A small artificial diffusion term,
`eps = 0.0025`, regularises the problem; it also sets a floor on the achievable
error against the inviscid analytic reference, which is worth keeping in mind
when reading the results.

The reference solution is built by the method of characteristics with the Welge
tangent construction and is tabulated in `data/`.

## Method

The network maps `(x, t) -> s` and is deliberately small: `2 -> 20 -> 7x(20) -> 1`
with `tanh`, 3021 parameters. Training uses 10000 collocation points placed by a
Latin hypercube, plus 150 points each for the initial and boundary conditions.
Every derivative in the residual, including `d2s/dx2`, comes from reverse-mode
automatic differentiation.

Training runs in two stages:

**Stage 1, Adam** in float32, with a fixed number of optimizer updates (not
epochs), so configurations with different batch sizes do the same amount of
optimization work. Learning rate decays by `StepLR(1000, 0.7)`.

**Stage 2, L-BFGS** in float64, full batch, `strong_wolfe` line search. Both
non-obvious settings here are forced by the problem. float64 is mandatory
because the residual contains a second derivative, and in float32 the gradient
noise exceeds the per-step improvement, so the line search returns a zero step.
`lr = 1.0` is used because the line search picks the real step length, and a
small initial value makes the trial step so short that `tolerance_change` fires
on the second iteration.

### Parallelisation

The Adam stage is data-parallel over `K = 2` GPUs. Collocation points are
sharded by `[rank::world_size]`; the small IC and BC sets are replicated on
every rank so that after gradient averaging their contribution matches the
single-GPU case exactly. `DistributedDataParallel` averages gradients during
backward.

The L-BFGS stage is **not** parallelised, and this is a property of the
algorithm rather than a shortcut. L-BFGS accumulates a curvature history from
secant pairs and its line search evaluates the objective several times per
iteration, both of which require a deterministic objective, that is, the full
batch. Rank 0 runs it while the other rank waits at a barrier.

One consequence is worth stating up front: since the second stage is serial, the
speedup of the whole pipeline is bounded by Amdahl's law, and with
`T_lbfgs >> T_adam` that bound sits close to 1. Any observed gain therefore has
to come from somewhere other than raw parallel speed.

### Learning rate scaling laws

With `L0 = 1e-3` and `K` GPUs:

| Law | Formula | Origin |
|---|---|---|
| `baseline` | `lr = L0` | no scaling, control |
| `linearK` | `lr = L0 * K` | linear scaling rule, Goyal et al. 2017 |
| `sqrtK` | `lr = L0 * sqrt(K)` | square root rule, Krizhevsky 2014; Hoffer et al. 2017 |

The linear rule was derived and validated for SGD with momentum; for adaptive
optimizers such as Adam the square root rule is the better-motivated choice.

## Metrics

All three are evaluated on an inference grid that was never used in training:
`t = linspace(0.01, 1, 100)` by `x = linspace(0, 1, 200)`.

**`l2_mean`**, the primary metric, is the mean relative L2 error over the 100
time slices. **`l_inf`** is the maximum absolute deviation over the grid, which
is sensitive to a misplaced front.

**`mass_balance_rel_mean`** needs no reference solution at all. Integrating the
conservation law over the domain turns the divergence term into boundary fluxes:

```
M(t)   = int_0^1 s(x, t) dx
F(x,t) = f(s) - eps * s_x
E(t)   = [M(t) - M(0)] - int_0^t [F(0,tau) - F(1,tau)] dtau
```

`E(t)` is zero for the exact solution. The reported figure is the mean of
`|E(t)| / t` over `t >= 0.1`, where `t` is the injected pore volume implied by
the boundary condition `s(0,t) = 1`. Normalising by `t` rather than by the
influx the network itself predicts matters: a model that violates the boundary
condition underestimates its own influx, which would inflate the ratio by orders
of magnitude for reasons unrelated to conservation.

This metric separates two failure modes that L2 conflates, namely being close to
the reference and actually satisfying the equation.

## Experiments

Six sweeps: 3 scaling laws x 8 mini-batch sizes (32 to 4096 per GPU) x 2 step
counts (5000 and 15000) x 2 modes (Adam only, Adam + L-BFGS), which is 96
trained models, plus sequential runs and a single-GPU full-batch reference at
each step count.

```
./run_all.sh            # everything, roughly 9 hours on 2x RTX 5090
./run_all.sh 5000       # only the 5000-step sweeps
./run_all.sh 15000      # only the 15000-step sweeps
./run_all.sh report     # only recompute metrics and rebuild the tables
```

The driver stops on the first failure so that later stages are never built from
partial data. Each step writes its own log into `logs/`.

## Results

Best model of each mode and sweep, on 2 GPUs:

| Mode | Steps | Adam, s | L-BFGS, s | Total, s | L2 |
|---|---|---|---|---|---|
| Adam only, `lr(sqrtK)_bs4096` | 5000 | 21.4 | — | 21.4 | 0.0895 |
| Adam + L-BFGS, `lr(baseline)_bs512` | 5000 | 20.6 | 528.9 | 549.5 | **0.0571** |
| Adam only, `lr(sqrtK)_bs1024` | 15000 | 60.0 | — | 60.0 | 0.0670 |
| Adam + L-BFGS, `lr(sqrtK)_bs512` | 15000 | 60.0 | 480.9 | 540.9 | **0.0644** |

Sequential reference on one GPU: L2 0.0731 in 386 s at 5000 steps, L2 0.0768 in
368 s at 15000 steps.

Three findings stand out.

**The Adam stage does not speed up on two cards.** With 3021 parameters a single
step does not saturate one GPU, so the bottleneck is kernel launch rather than
arithmetic, and gradient synchronisation costs about as much as the step itself.
For the same reason the batch size barely affects the Adam time: growing it by a
factor of 128 changes the time by single-digit percent. Speedup from data
parallelism should be expected on networks orders of magnitude larger, or with
far more collocation points.

**L-BFGS improves the mass balance far more than it improves L2.** The second
stage mostly fixes how well the equation is satisfied rather than how closely
the profile matches the reference. The best model by L2 is frequently not the
best by mass balance, which is exactly why the third metric earns its place.

**More Adam steps buy very little.** Tripling the step count moves the best L2
by a few percent, and sometimes in the wrong direction. The ceiling here is set
by the artificial diffusion `eps`, not by the optimizer: a lower loss means a
more accurate solution of the *viscous* equation, which differs from the
inviscid analytic solution with a shock.

Full tables, including every batch size for every law, are in
[report/tables.md](report/tables.md). Comparison plots are
`report/time_comparison.png` and `report/l2_comparison.png`.

## Layout

| Path | Contents |
|---|---|
| `train_bl.py` | sequential version, Adam then L-BFGS on one GPU |
| `train_ddp.py` | DDP sweep, Adam only, plus `--baseline-reference` mode |
| `train_ddp_lbfgs.py` | DDP sweep, Adam on all GPUs then L-BFGS on rank 0 |
| `mass_balance.py` | mass balance metric shared by the DDP scripts |
| `eval_metrics.py` | recomputes all metrics for every trained model |
| `make_report.py` | builds `report/tables.md` and the comparison plots |
| `run_all.sh` | driver for the whole series |
| `data/` | tabulated analytic solution |
| `sequential/`, `ddp_adam/`, `ddp_adam_lbfgs/` | results per sweep |
| `report/` | tables and plots |
| `logs/` | one log per sweep |

Trained weights, per-configuration plots and run logs are not tracked in git;
`run_all.sh` regenerates them.

## Requirements

PyTorch with CUDA, NumPy, SciPy and Matplotlib. Two GPUs are needed for the DDP
sweeps; `train_bl.py`, `eval_metrics.py` and `make_report.py` run on one GPU or
on CPU. The results above were produced on 2x NVIDIA RTX 5090.

```
pip install torch numpy scipy matplotlib
```

## References

1. Buckley S. E., Leverett M. C. Mechanism of Fluid Displacement in Sands //
   Transactions of the AIME. 1942. Vol. 146. P. 107-116.
2. Welge H. J. A Simplified Method for Computing Oil Recovery by Gas or Water
   Drive // Journal of Petroleum Technology. 1952. Vol. 4, No. 4. P. 91-98.
3. Raissi M., Perdikaris P., Karniadakis G. E. Physics-Informed Neural
   Networks // Journal of Computational Physics. 2019. Vol. 378. P. 686-707.
4. Fuks O., Tchelepi H. A. Limitations of Physics Informed Machine Learning for
   Nonlinear Two-Phase Transport in Porous Media // Journal of Machine Learning
   for Modeling and Computing. 2020. Vol. 1, No. 1. P. 19-37.
5. Liu D. C., Nocedal J. On the Limited Memory BFGS Method for Large Scale
   Optimization // Mathematical Programming. 1989. Vol. 45. P. 503-528.
6. Kingma D. P., Ba J. Adam: A Method for Stochastic Optimization // ICLR. 2015.
7. Li S. et al. PyTorch Distributed: Experiences on Accelerating Data Parallel
   Training // Proceedings of the VLDB Endowment. 2020. Vol. 13, No. 12.
8. Goyal P. et al. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour //
   arXiv:1706.02677. 2017.
9. Krizhevsky A. One Weird Trick for Parallelizing Convolutional Neural
   Networks // arXiv:1404.5997. 2014.
10. Hoffer E., Hubara I., Soudry D. Train Longer, Generalize Better // NeurIPS.
    2017.

## License

MIT, see [LICENSE](LICENSE).
