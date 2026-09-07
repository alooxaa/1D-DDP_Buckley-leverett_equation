"""
Hybrid PINN for 1D Buckley-Leverett: Adam on two GPUs, then L-BFGS on one.

The idea behind the experiment. Adam parallelises well: the mini-batch is split
across the cards and the gradients are averaged through DistributedDataParallel.
L-BFGS cannot be parallelised, since it needs a deterministic full batch and a
curvature history, so the second stage runs on a single card. The question the
experiment answers is whether this combination reaches the accuracy of the
sequential version in less time.

The physics, the model, the metrics and the first stage are taken from
train_ddp.py unchanged.

Matrix: 3 lr scaling laws x 8 mini-batch sizes = 24 configurations. Only laws
that depend on the number of GPUs are used.

    L0 = 1e-3            base lr
    K  = number of GPUs  from dist.get_world_size()

    baseline : lr = L0
    linearK  : lr = L0 * K
    sqrtK    : lr = L0 * sqrt(K)

Usage:
    torchrun --nproc_per_node=2 train_ddp_lbfgs.py --adam-steps 50 \\
        --lbfgs-max-iter 50 --out /tmp/smoke          # quick check
    torchrun --nproc_per_node=2 train_ddp_lbfgs.py --adam-steps 5000 \\
        --out ddp_adam_lbfgs/5000
    torchrun --nproc_per_node=2 train_ddp_lbfgs.py --adam-steps 15000 \\
        --out ddp_adam_lbfgs/15000
"""

import argparse
import csv
import json
import math
import os
import time
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from scipy.stats import qmc
from torch.nn.parallel import DistributedDataParallel

import mass_balance as mb

# ─────────────────────────────── settings ───────────────────────────────────
# Taken from train_bl.py / train_ddp.py unchanged.
N_F = 10000           # number of collocation points
N_U = 300             # initial and boundary condition points (split evenly)

M = 1.0               # viscosity ratio
EPS = 0.0025          # artificial diffusion coefficient
SEED = 42

# ── experiment parameters ──
BASE_LR = 1e-3
ADAM_STEPS = 5000
LBFGS_MAX_ITER = 20000
BATCH_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]   # per GPU
LR_LAWS = ["baseline", "linearK", "sqrtK"]

# ── inference grid used by the metrics ──
N_T_EVAL = 100
T_GRID = np.linspace(0.01, 1.0, N_T_EVAL)   # t = 0 excluded: the exact solution is a step there
X_GRID = np.linspace(0, 1, 200)
T_PLOT = [0.25, 0.5, 0.75, 1.0]             # time levels for the solution plot

# metrics that end up in results_all.csv and summary.json
L2_KEYS = ("l2_norm", "l2_mean", "l_inf")
MB_KEYS = ("mass_balance_rel_mean", "mass_balance_rel_final",
           "mass_balance_mean", "mass_balance_max", "mass_t0")
METRIC_KEYS = L2_KEYS + MB_KEYS

DATA_DIR = "data"

# Assigned in setup_ddp(): every rank works on its own card.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────── model ─────────────────────────────────────
class PINN_BL(nn.Module):
    """Fully connected net (x, t) -> Sw. Two inputs, one output, 7 hidden layers of 20."""

    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(2, 20)
        self.hiddens = nn.ModuleList([nn.Linear(20, 20) for _ in range(7)])
        self.linear_out = nn.Linear(20, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.linear_in(x))
        for layer in self.hiddens:
            x = self.activation(layer(x))
        return self.linear_out(x)


def derivative(dy, x, order=1):
    """Derivative of dy with respect to x of the requested order, via autograd."""
    for _ in range(order):
        dy = torch.autograd.grad(
            dy, x,
            grad_outputs=torch.ones_like(dy),
            create_graph=True,
            retain_graph=True,
        )[0]
    return dy


def u_function(model, x, t):
    """Predicted Sw at the points (x, t)."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    return model(torch.cat([x, t], dim=1))


def f(model, x, t):
    """PDE residual: s_t + df/ds * s_x - eps * s_xx."""
    s = u_function(model, x, t)

    # f(s) = s^2 / (s^2 + (1 - s)^2 / M); df/ds is differentiated analytically
    den = s**2 + (1 - s)**2 / M
    df_ds = (2 * s * den - s**2 * (2 * s - 2 * (1 - s) / M)) / den**2

    s_t = derivative(s, t, order=1)
    s_x = derivative(s, x, order=1)
    s_xx = derivative(s_x, x, order=1)

    return s_t + df_ds * s_x - EPS * s_xx


def loss_function(model, x_f, t_f, x_ic, t_ic, x_bc, t_bc, u_ic, u_bc):
    """Sum of three MSE terms: PDE residual, initial condition, boundary condition."""
    mse = nn.MSELoss()

    residual = f(model, x_f, t_f)
    loss_pde = mse(residual, torch.zeros_like(residual))
    loss_ic = mse(u_function(model, x_ic, t_ic), u_ic)
    loss_bc = mse(u_function(model, x_bc, t_bc), u_bc)

    total = loss_pde + loss_ic + loss_bc
    return total, loss_pde, loss_ic, loss_bc


# ─────────────────────────── training points ────────────────────────────────
def make_data(dtype=torch.float32):
    """Training points: collocation points in the domain plus IC and BC points."""
    n_ic = N_U // 2
    n_bc = N_U // 2

    # collocation points over the domain (Latin hypercube)
    engine = qmc.LatinHypercube(d=2, seed=41)
    points = engine.random(n=N_F)
    x_f = points[:, 1:2]
    t_f = points[:, 0:1]

    # initial condition: t = 0, arbitrary x, Sw = 0
    engine_1d = qmc.LatinHypercube(d=1, seed=41)
    t_bc = engine_1d.random(n=n_bc)
    x_ic = engine_1d.random(n=n_ic)
    t_ic = np.zeros((n_ic, 1))

    # boundary condition: x = 0, arbitrary t, Sw = 1
    x_bc = np.zeros((n_bc, 1))

    u_ic = np.zeros((n_ic, 1))     # Sw(x, 0) = 0
    u_bc = np.ones((n_bc, 1))      # Sw(0, t) = 1

    def tensor(a, grad=False):
        out = torch.tensor(a, dtype=dtype, device=device)
        return out.requires_grad_(True) if grad else out

    return {
        "x_f": tensor(x_f, grad=True), "t_f": tensor(t_f, grad=True),
        "x_ic": tensor(x_ic), "t_ic": tensor(t_ic),
        "x_bc": tensor(x_bc), "t_bc": tensor(t_bc),
        "u_ic": tensor(u_ic), "u_bc": tensor(u_bc),
    }


def shard_collocation(data, rank, world_size):
    """Splits collocation points across processes: the slice [rank::world_size].

    IC and BC points (150 + 150) are not split. They are small and stay whole on
    every rank, so that after gradient averaging their contribution matches the
    single-GPU case exactly.
    """
    x_local = data["x_f"].detach()[rank::world_size]
    t_local = data["t_f"].detach()[rank::world_size]
    return x_local, t_local


class BatchSampler:
    """Cyclic pass over the local points, reshuffled on every cycle."""

    def __init__(self, n_points, batch_size, generator):
        self.n_points = n_points
        self.batch_size = batch_size
        self.generator = generator
        self.order = torch.randperm(n_points, generator=generator)
        self.pos = 0

    def next_indices(self):
        idx = []
        need = self.batch_size
        while need > 0:
            if self.pos >= self.n_points:
                self.order = torch.randperm(self.n_points, generator=self.generator)
                self.pos = 0
            take = min(need, self.n_points - self.pos)
            idx.append(self.order[self.pos:self.pos + take])
            self.pos += take
            need -= take
        return torch.cat(idx)


# ─────────────────────── stage 1: Adam on every GPU ─────────────────────────
def train_adam_ddp(model, data, x_local, t_local, steps, lr, batch_size,
                   rank, world_size, log_every=1000):
    """Mini-batch Adam. DDP averages the gradients during backward."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)
    history = []

    generator = torch.Generator().manual_seed(SEED + rank)
    sampler = BatchSampler(x_local.shape[0], batch_size, generator)

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()
    start = time.perf_counter()

    for step in range(steps):
        idx = sampler.next_indices().to(device)
        batch = dict(data)
        batch["x_f"] = x_local[idx].clone().requires_grad_(True)
        batch["t_f"] = t_local[idx].clone().requires_grad_(True)

        optimizer.zero_grad()
        total, loss_pde, loss_ic, loss_bc = loss_function(model, **batch)
        total.backward()          # DDP averages gradients across cards here
        optimizer.step()
        scheduler.step()
        history.append(total.item())

        if rank == 0 and log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    [adam]  step {step:>6d} | loss={total.item():.4e} "
                  f"pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} "
                  f"bc={loss_bc.item():.3e}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()
    return history, time.perf_counter() - start


# ────────────────────── stage 2: L-BFGS on a single GPU ─────────────────────
def train_lbfgs_single(model, max_iter, log_every=2000):
    """L-BFGS on a single card, full batch, float64.

    This stage cannot be parallelised: L-BFGS builds a curvature history and
    needs a deterministic objective, that is, the full batch on every call.
    Only rank 0 runs it while the other ranks wait at a barrier.

    lr is 1.0 rather than something small because strong_wolfe picks the step
    length. With a small lr the trial step becomes so short that the optimizer
    immediately hits tolerance_change and stops on the second iteration.

    float64 is mandatory: in float32 the gradient noise (the residual contains
    the second derivative s_xx) exceeds the real improvement, strong_wolfe
    returns a zero step and the stage ends before it starts.
    """
    model = model.double()
    data = make_data(torch.float64)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        history_size=100,
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
        line_search_fn="strong_wolfe",
    )
    history = []

    def closure():
        optimizer.zero_grad()
        total, loss_pde, loss_ic, loss_bc = loss_function(model, **data)
        total.backward()
        history.append(total.item())

        if log_every and len(history) % log_every == 0:
            print(f"    [lbfgs] call {len(history):>6d} | loss={total.item():.4e} "
                  f"pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} "
                  f"bc={loss_bc.item():.3e}", flush=True)
        return total

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    optimizer.step(closure)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # one L-BFGS iteration makes several closure calls while searching the step
    state = list(optimizer.state.values())
    n_iter = int(state[0].get("n_iter", 0)) if state else 0
    return model, history, elapsed, n_iter


# ───────────────────────────── analytic solution ────────────────────────────
def load_exact():
    """Analytic solution from CSV (method of characteristics, Welge construction)."""
    sol = np.loadtxt(os.path.join(DATA_DIR, "matrix_only.csv"), delimiter=",")
    xs = np.loadtxt(os.path.join(DATA_DIR, "matrix2_only.csv"), delimiter=",")
    return xs, sol


def exact_at(t_val, x_plot, exact_x, exact_sol):
    """Profile Sw(x) at time t_val sampled on the grid x_plot."""
    row = int(round(t_val * (exact_sol.shape[0] - 1)))
    row = min(max(row, 0), exact_sol.shape[0] - 1)
    xs = exact_x[row, :]
    ss = exact_sol[row, :]

    # After breakthrough (t ~ 0.8284) the front leaves the domain and the tail
    # of the row stops increasing, but np.interp needs increasing x, so trim it.
    decreasing = np.where(np.diff(xs) < 0)[0]
    if decreasing.size:
        xs, ss = xs[:decreasing[0] + 1], ss[:decreasing[0] + 1]
    inside = xs <= 1.0
    if inside.sum() >= 2:
        xs, ss = xs[inside], ss[inside]

    return np.interp(x_plot, xs, ss)


def predict(model, t_val, x_plot):
    """Model prediction on the grid x_plot at time t_val."""
    model.eval()
    dtype = next(model.parameters()).dtype
    x = torch.tensor(x_plot, dtype=dtype, device=device)
    t = torch.full_like(x, float(t_val))
    with torch.no_grad():
        return u_function(model, x, t).cpu().numpy().flatten()


def compute_metrics(model, exact_x, exact_sol):
    """Error against the reference solution plus the mass balance residual.

    The first three metrics are computed on a grid of 100 time slices by 200
    points in x against the analytic solution. The mass balance
    (mass_balance.py) uses no reference at all: it checks whether the network
    itself conserves mass.
    """
    pred = np.empty((N_T_EVAL, X_GRID.size))
    ref = np.empty((N_T_EVAL, X_GRID.size))
    for i, t_val in enumerate(T_GRID):
        pred[i] = predict(model, t_val, X_GRID)
        ref[i] = exact_at(t_val, X_GRID, exact_x, exact_sol)

    diff = pred - ref
    per_time = np.linalg.norm(diff, axis=1) / np.linalg.norm(ref, axis=1)

    return {
        "l2_norm": float(np.linalg.norm(diff) / np.linalg.norm(ref)),
        "l2_mean": float(per_time.mean()),
        "l_inf": float(np.abs(diff).max()),
        "l2_std": float(per_time.std()),
        "l2_max_slice": float(per_time.max()),
        **mb.mass_balance(model),
    }


# ──────────────────────────────── plots ─────────────────────────────────────
def plot_solution(model, exact_x, exact_sol, path, title=""):
    """PINN against the analytic solution at four time levels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, t_val in zip(axes.flat, T_PLOT):
        ref = exact_at(t_val, X_GRID, exact_x, exact_sol)
        pred = predict(model, t_val, X_GRID)
        l2 = np.linalg.norm(pred - ref) / np.linalg.norm(ref)

        ax.plot(X_GRID, ref, "b-", lw=2, label="Exact")
        ax.plot(X_GRID, pred, "r--", lw=2, label="PINN")
        ax.set_title(f"t = {t_val}   |   L2 = {l2:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("Sw")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(adam_history, lbfgs_history, path, title=""):
    """Two panels: Adam on two GPUs and L-BFGS on one."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].semilogy(adam_history, "b", lw=1.0)
    axes[0].set_title(f"Adam on 2 GPUs, {len(adam_history)} steps")
    axes[0].set_xlabel("optimizer step")

    axes[1].semilogy(lbfgs_history, "r", lw=1.0)
    axes[1].set_title(f"L-BFGS on 1 GPU, {len(lbfgs_history)} closure calls")
    axes[1].set_xlabel("closure call")

    for ax in axes:
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ───────────────────────────── scaling laws ─────────────────────────────────
def compute_lr(law, world_size):
    """lr for the chosen law. L0 = BASE_LR, K = world_size."""
    if law == "baseline":
        return BASE_LR
    if law == "linearK":
        return BASE_LR * world_size
    if law == "sqrtK":
        return BASE_LR * math.sqrt(world_size)
    raise ValueError(f"unknown lr law: {law}")


def model_name(law, lr, batch_size, world_size):
    return f"{law}_lr{lr:.3e}_bs{batch_size}_gpu{world_size}"


# ──────────────────────────────── DDP ───────────────────────────────────────
def setup_ddp():
    """Process group setup. Returns (rank, world_size, distributed)."""
    global device

    if "RANK" not in os.environ:          # plain python launch, no torchrun
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, False

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # The default timeout is 30 minutes, and rank 1 idles at the barrier for the
    # whole L-BFGS stage, which took up to 23 minutes in the sequential version.
    # With a 20000 iteration cap some configurations would cross 30 minutes and
    # the barrier would abort the run.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, True


# ────────────────────────── a single experiment ─────────────────────────────
def run_config(law, batch_size, args, rank, world_size, exact_x, exact_sol,
               out_root):
    """Adam on every GPU, then L-BFGS on rank 0."""
    lr = compute_lr(law, world_size)
    name = model_name(law, lr, batch_size, world_size)

    set_seed(args.seed)
    raw_model = PINN_BL().to(device)
    model = (DistributedDataParallel(raw_model, device_ids=[device.index])
             if world_size > 1 else raw_model)

    data = make_data()
    x_local, t_local = shard_collocation(data, rank, world_size)

    # ── stage 1: Adam on every card ──
    adam_history, time_adam = train_adam_ddp(
        model, data, x_local, t_local, args.adam_steps, lr, batch_size,
        rank, world_size, log_every=args.log_every,
    )
    if rank == 0:
        print(f"    Adam on {world_size} GPUs: {time_adam:.2f} s, "
              f"loss {adam_history[-1]:.4e}", flush=True)

    metrics_adam = compute_metrics(raw_model, exact_x, exact_sol) if rank == 0 else None

    # ── stage 2: L-BFGS on rank 0 only ──
    if rank == 0:
        # unwrap DDP: otherwise the autograd hooks would all-reduce onto a rank
        # that is sitting at the barrier at this moment
        model_lbfgs, lbfgs_history, time_lbfgs, lbfgs_iters = train_lbfgs_single(
            raw_model, args.lbfgs_max_iter, log_every=args.log_every_lbfgs
        )
        print(f"    L-BFGS on 1 GPU: {time_lbfgs:.2f} s, {lbfgs_iters} iterations, "
              f"loss {lbfgs_history[-1]:.4e}", flush=True)

    if world_size > 1:
        dist.barrier()      # the other ranks wait here, timeout 3 hours

    if rank != 0:
        return None

    metrics_lbfgs = compute_metrics(model_lbfgs, exact_x, exact_sol)

    run_dir = os.path.join(out_root, law, f"bs{batch_size}")
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model_lbfgs.state_dict(), os.path.join(run_dir, "model.pth"))

    result = {
        "model_name": name,
        "lr_law": law,
        "batch_size_per_gpu": batch_size,
        "global_batch_size": batch_size * world_size,
        "n_f": N_F,
        "num_gpus": world_size,
        "learning_rate": lr,
        "adam_steps": len(adam_history),
        "lbfgs_iterations": lbfgs_iters,
        "lbfgs_closure_calls": len(lbfgs_history),
        "time_adam": time_adam,
        "time_lbfgs": time_lbfgs,
        "time_total": time_adam + time_lbfgs,
        # metrics after each stage, so the contribution of L-BFGS is visible
        **{key: metrics_lbfgs[key] for key in METRIC_KEYS},
        "after_adam": metrics_adam,
        "after_lbfgs": metrics_lbfgs,
        "loss_after_adam": adam_history[-1],
        "loss_after_lbfgs": lbfgs_history[-1],
        "n_parameters": sum(p.numel() for p in raw_model.parameters()),
        "seed": args.seed,
        "adam_dtype": "float32",
        "lbfgs_dtype": "float64",
        "local_points": x_local.shape[0],
        "batch_exceeds_local_data": batch_size > x_local.shape[0],
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    with open(os.path.join(run_dir, "model_result.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    title = (f"{law}  lr={lr:.3e}  bs={batch_size}  |  "
             f"L2 {metrics_adam['l2_mean']:.4f} -> {metrics_lbfgs['l2_mean']:.4f}")
    plot_loss(adam_history, lbfgs_history, os.path.join(run_dir, "loss.png"),
              title=title)
    plot_solution(model_lbfgs, exact_x, exact_sol,
                  os.path.join(run_dir, "solution.png"), title=title)

    print(f"    L2 mean: {metrics_adam['l2_mean']:.4e} (after Adam) -> "
          f"{metrics_lbfgs['l2_mean']:.4e} (after L-BFGS) | "
          f"Linf {metrics_lbfgs['l_inf']:.4e}", flush=True)
    print(f"    mass balance: {metrics_adam['mass_balance_rel_mean']:.4e} -> "
          f"{metrics_lbfgs['mass_balance_rel_mean']:.4e} (rel, mean over t) | "
          f"{metrics_lbfgs['mass_balance_rel_final']:.4e} at t=1", flush=True)
    return result


# ──────────────────────────────── tables ────────────────────────────────────
CSV_FIELDS = [
    "model_name", "lr_law", "batch_size_per_gpu", "global_batch_size", "n_f",
    "num_gpus", "learning_rate", "adam_steps", "lbfgs_iterations",
    "lbfgs_closure_calls", "time_adam", "time_lbfgs", "time_total",
    *METRIC_KEYS, "loss_after_adam", "loss_after_lbfgs",
]


def flatten(row):
    out = {k: row[k] for k in CSV_FIELDS}
    for stage in ("after_adam", "after_lbfgs"):
        for key in METRIC_KEYS:
            out[f"{stage}_{key}"] = row[stage][key]
    return out


def write_tables(rows, out_dir):
    """results_all.csv plus time_summary.csv sorted by training time."""
    flat = [flatten(r) for r in rows]
    fields = list(flat[0].keys())

    with open(os.path.join(out_dir, "results_all.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)

    ranked = [{"rank": i, **r} for i, r in
              enumerate(sorted(flat, key=lambda r: r["time_total"]), 1)]
    with open(os.path.join(out_dir, "time_summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank"] + fields)
        writer.writeheader()
        writer.writerows(ranked)
    return ranked


def write_summary(rows, sweep_time, out_dir):
    """Mean metrics across all configurations and per scaling law."""
    def stats(subset):
        return {
            key: {
                "mean": float(np.mean([r[key] for r in subset])),
                "min": float(np.min([r[key] for r in subset])),
                "max": float(np.max([r[key] for r in subset])),
            }
            for key in METRIC_KEYS
        }

    best = min(rows, key=lambda r: r["l2_mean"])
    summary = {
        "n_experiments": len(rows),
        "adam_steps": rows[0]["adam_steps"],
        "lbfgs_max_iter": LBFGS_MAX_ITER,
        "num_gpus": rows[0]["num_gpus"],
        "batch_sizes": sorted({r["batch_size_per_gpu"] for r in rows}),
        "lr_laws": LR_LAWS,
        "metrics_all_experiments": stats(rows),
        "metrics_after_adam": {
            key: float(np.mean([r["after_adam"][key] for r in rows]))
            for key in METRIC_KEYS
        },
        "time": {
            "adam_mean_sec": float(np.mean([r["time_adam"] for r in rows])),
            "lbfgs_mean_sec": float(np.mean([r["time_lbfgs"] for r in rows])),
            "total_mean_sec": float(np.mean([r["time_total"] for r in rows])),
            "sweep_total_sec": sweep_time,
        },
        "best_model": best["model_name"],
        "best_metrics": {k: best[k] for k in METRIC_KEYS},
        "best_model_by_mass_balance":
            min(rows, key=lambda r: r["mass_balance_rel_mean"])["model_name"],
        "per_law": {law: stats([r for r in rows if r["lr_law"] == law])
                    for law in LR_LAWS if any(r["lr_law"] == law for r in rows)},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ───────────────────────────────── main ─────────────────────────────────────
def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(
        description="Hybrid PINN: Adam on N GPUs, then L-BFGS on one")
    parser.add_argument("--adam-steps", type=int, default=ADAM_STEPS)
    parser.add_argument("--lbfgs-max-iter", type=int, default=LBFGS_MAX_ITER)
    parser.add_argument("--out", default="ddp_adam_lbfgs/5000")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--log-every-lbfgs", type=int, default=2000)
    args = parser.parse_args()
    DATA_DIR = args.data_dir

    rank, world_size, distributed = setup_ddp()
    exact_x, exact_sol = load_exact()

    out_root = args.out
    if rank == 0:
        os.makedirs(out_root, exist_ok=True)
        print("=" * 70)
        print(f"GPU count       : {world_size}")
        print(f"N_F             : {N_F}")
        print(f"Adam steps      : {args.adam_steps}  (on {world_size} GPUs, mini-batch)")
        print(f"L-BFGS max_iter : {args.lbfgs_max_iter}  (on 1 GPU, full batch, float64)")
        print(f"lr laws         : {', '.join(LR_LAWS)}")
        print(f"batch sizes     : {BATCH_SIZES}")
        print(f"configurations  : {len(LR_LAWS) * len(BATCH_SIZES)}")
        if device.type == "cuda":
            print(f"device          : {torch.cuda.get_device_name(0)}")
        print(f"results         : {out_root}")
        print("=" * 70, flush=True)

    rows, failed = [], []
    total_configs = len(LR_LAWS) * len(BATCH_SIZES)
    sweep_start = time.perf_counter()

    for i, law in enumerate(LR_LAWS):
        for j, batch_size in enumerate(BATCH_SIZES):
            n = i * len(BATCH_SIZES) + j + 1
            if rank == 0:
                print(f"\n[{n}/{total_configs}] {law}  bs={batch_size}  "
                      f"lr={compute_lr(law, world_size):.6e}", flush=True)
            try:
                result = run_config(law, batch_size, args, rank, world_size,
                                    exact_x, exact_sol, out_root)
                if result is not None:
                    rows.append(result)
                    write_tables(rows, out_root)   # intermediate dump
            except Exception as exc:                       # noqa: BLE001
                if rank == 0:
                    print(f"  ERROR: {exc!r}", flush=True)
                    failed.append((law, batch_size, repr(exc)))
            if distributed:
                dist.barrier()

    sweep_time = time.perf_counter() - sweep_start

    if rank == 0 and rows:
        ranked = write_tables(rows, out_root)
        summary = write_summary(rows, sweep_time, out_root)

        print("\n" + "=" * 70)
        print(f"done: {len(rows)}/{total_configs}"
              f"{f', failed: {len(failed)}' if failed else ''}")
        print(f"total time: {sweep_time / 3600:.2f} h")
        print("=" * 70)

        header = (f"{'law':<10}{'bs':>6}{'Adam,s':>9}{'LBFGS,s':>10}"
                  f"{'L2 norm':>11}{'L2 mean':>11}{'Linf':>11}")
        print("\nbest by L2 mean:")
        print(header)
        print("-" * len(header))
        for r in sorted(rows, key=lambda x: x["l2_mean"])[:5]:
            print(f"{r['lr_law']:<10}{r['batch_size_per_gpu']:>6}"
                  f"{r['time_adam']:>9.1f}{r['time_lbfgs']:>10.1f}"
                  f"{r['l2_norm']:>11.4e}{r['l2_mean']:>11.4e}{r['l_inf']:>11.4e}")

        m = summary["metrics_all_experiments"]
        print(f"\nmean over {len(rows)} configurations:")
        print(f"  L2 norm {m['l2_norm']['mean']:.4e} | "
              f"L2 mean {m['l2_mean']['mean']:.4e} | "
              f"Linf {m['l_inf']['mean']:.4e}")
        print(f"  after Adam (before L-BFGS): "
              f"L2 mean {summary['metrics_after_adam']['l2_mean']:.4e}")
        print(f"  time: Adam {summary['time']['adam_mean_sec']:.1f} s + "
              f"L-BFGS {summary['time']['lbfgs_mean_sec']:.1f} s = "
              f"{summary['time']['total_mean_sec']:.1f} s on average")

        print(f"\nmodels  : {out_root}/<law>/bs<N>/model.pth")
        print(f"tables  : {out_root}/results_all.csv, {out_root}/time_summary.csv")
        print(f"summary : {out_root}/summary.json", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
