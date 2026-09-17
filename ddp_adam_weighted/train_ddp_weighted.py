"""
Parallel PINN (DDP) for the 1D Buckley-Leverett equation, Adam only, with a
weighted loss.

This is the earlier series of the experiment. It differs from train_ddp.py in
the root of the repository in exactly one respect: the three loss terms are
combined with fixed weights,

    loss = LAMBDA_F * loss_pde + LAMBDA_IC * loss_ic + LAMBDA_BC * loss_bc,

with (3, 5, 1) chosen by a grid search beforehand. The later series in the
root drops the weights and uses the plain sum. Everything else, the physics,
the network, the collocation points, the optimizer and the metrics, is the
same, so the two series are directly comparable.

The practical point of this series is that mini-batch Adam on two GPUs, with
no second stage at all, already reaches a competitive L2 error in about 20 s
at 5000 updates and about 60 s at 15000, that is, an order of magnitude faster
than the two-stage sequential scheme.

Experiment: 5 learning rate scaling laws x 8 mini-batch sizes = 40
configurations per sweep (the 5000-update sweep also includes batch 8192, so
45). Each one performs a fixed number of optimizer updates (not epochs), 5000
or 15000 depending on --adam-steps.

Three laws scale the learning rate with the number of GPUs, two scale it with
the mini-batch size relative to the full collocation set:

    L0 = 1e-3            base lr, the same value used in train_bl.py
    K  = number of GPUs  from dist.get_world_size()
    G  = mini-batch size per GPU
    G0 = 10000           number of collocation points, i.e. the full batch

    baseline : lr = L0
    linearK  : lr = L0 * K
    sqrtK    : lr = L0 * sqrt(K)
    linearG  : lr = L0 * (G / G0)
    sqrtG    : lr = L0 * sqrt(G / G0)

The batch-dependent laws are included for completeness: at small batches they
shrink the learning rate by two to three orders of magnitude (linearG gives
3.2e-6 at G = 32), so the network barely trains. This is the expected
behaviour of the formula, not a failure of the run, and it is why the later
unweighted series keeps only the three GPU-count laws.

Usage (from the repository root, so that data/ resolves):
    torchrun --nproc_per_node=2 ddp_adam_weighted/train_ddp_weighted.py --verify
    torchrun --nproc_per_node=2 ddp_adam_weighted/train_ddp_weighted.py \
        --adam-steps 5000 --out ddp_adam_weighted/5000
    python ddp_adam_weighted/train_ddp_weighted.py --baseline-reference \
        --adam-steps 5000 --out ddp_adam_weighted/5000
"""

import argparse
import csv
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from scipy.stats import qmc
from torch.nn.parallel import DistributedDataParallel

# ─────────────────────────────── settings ───────────────────────────────────
# Taken from train_bl.py unchanged.
N_F = 10000           # number of collocation points
N_U = 300             # initial and boundary condition points (split evenly)

# Loss weights. The only thing that distinguishes this series from the
# unweighted one in the repository root.
LAMBDA_F = 3.0        # PDE residual
LAMBDA_IC = 5.0       # initial condition
LAMBDA_BC = 1.0       # boundary condition

M = 1.0               # viscosity ratio
EPS = 0.0025          # artificial diffusion coefficient
SEED = 42

T_EVAL = [0.25, 0.5, 0.75, 1.0]   # time levels used for the L2 report and plots

# ── experiment parameters ──
BASE_LR = 1e-3        # L0: the same lr as in train_bl.py
ADAM_STEPS = 15000    # exactly this many optimizer updates per configuration
BATCH_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]         # per GPU
LR_LAWS = ["baseline", "linearK", "sqrtK", "linearG", "sqrtG"]
G0 = 10000            # denominator of the batch-dependent laws: the full batch

EXPERIMENTS_DIR = "ddp_adam_weighted/15000"
DATA_DIR = "data"

# Assigned in setup_ddp(): every rank works on its own card. The rest of the
# code (make_data, predict) just reads this global, exactly as train_bl.py does.
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
    """Weighted sum of three MSE terms: PDE residual, initial condition, boundary condition."""
    mse = nn.MSELoss()

    residual = f(model, x_f, t_f)
    loss_pde = mse(residual, torch.zeros_like(residual))
    loss_ic = mse(u_function(model, x_ic, t_ic), u_ic)
    loss_bc = mse(u_function(model, x_bc, t_bc), u_bc)

    total = LAMBDA_F * loss_pde + LAMBDA_IC * loss_ic + LAMBDA_BC * loss_bc
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
    """Cyclic pass over the local points, reshuffled on every cycle.

    If the batch is larger than the local shard the cycle wraps around and the
    batch contains repeats, which is unavoidable with N_F = 10000 split across
    the cards and batches of up to 4096 per GPU.
    """

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


# ──────────────────────────────── training ──────────────────────────────────
def train_adam_ddp(model, data, x_local, t_local, steps, lr, batch_size,
                   rank, world_size, grad_sync, log_every=1000):
    """Mini-batch Adam with gradient synchronisation across processes.

    Matches train_adam from train_bl.py: the same Adam, the same
    StepLR(1000, 0.7), the same zero_grad -> backward -> step order. The only
    difference is that each step draws a mini-batch of collocation points and
    the gradients are averaged across GPUs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)
    history = []

    generator = torch.Generator().manual_seed(SEED + rank)
    sampler = BatchSampler(x_local.shape[0], batch_size, generator)

    # Parameters for manual averaging (used only when grad_sync="manual")
    params = [p for p in model.parameters() if p.requires_grad]

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
        total.backward()

        # Average the gradients exactly once. With grad_sync="ddp" the autograd
        # hooks of DistributedDataParallel already did it, so a second average
        # would be wrong.
        if grad_sync == "manual" and world_size > 1:
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        optimizer.step()
        scheduler.step()
        history.append(total.item())

        if rank == 0 and log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    step {step:>6d} | loss={total.item():.4e} "
                  f"pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} "
                  f"bc={loss_bc.item():.3e}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()
    return history, time.perf_counter() - start


# ───────────────────────────── analytic solution ────────────────────────────
def load_exact():
    """Analytic solution loaded from CSV.

    matrix_only.csv holds the saturation profile (identical in every row,
    decreasing from 1 to 0), matrix2_only.csv holds the x positions of those
    values. Row i corresponds to t = i / (number of rows - 1).
    """
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


def compute_l2(model, exact_x, exact_sol):
    """Relative L2 error at each time level in T_EVAL."""
    x_plot = np.linspace(0, 1, 200)
    errors = {}
    for t_val in T_EVAL:
        pred = predict(model, t_val, x_plot)
        ref = exact_at(t_val, x_plot, exact_x, exact_sol)
        errors[t_val] = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
    return errors


# ──────────────────────────────── plots ─────────────────────────────────────
def plot_solution(model, exact_x, exact_sol, l2, path, title=""):
    """PINN against the analytic solution at four time levels."""
    x_plot = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, t_val in zip(axes.flat, T_EVAL):
        ax.plot(x_plot, exact_at(t_val, x_plot, exact_x, exact_sol),
                "b-", lw=2, label="Exact")
        ax.plot(x_plot, predict(model, t_val, x_plot), "r--", lw=2, label="PINN")
        ax.set_title(f"t = {t_val}   |   L2 = {l2[t_val]:.4e}")
        ax.set_xlabel("x")
        ax.set_ylabel("Sw")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(title or f"mean L2 = {np.mean(list(l2.values())):.4e}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(history, path, title=""):
    """Adam loss decay, logarithmic scale."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history, "b", lw=1.0)
    ax.set_title(title or f"Adam, {len(history)} steps")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(rows, out_dir):
    """Summary plots: L2 and training time versus batch size for each law."""
    for values, fname, ylabel, title, logy in (
        ("l2_error", "l2_vs_batch.png", "mean L2",
         "L2 error versus mini-batch size", True),
        ("training_time_sec", "time_vs_batch.png", "training time, s",
         "Adam training time versus mini-batch size", False),
    ):
        fig, ax = plt.subplots(figsize=(9, 6))
        for law in LR_LAWS:
            points = sorted(
                [(r["batch_size_per_gpu"], r[values]) for r in rows if r["lr_law"] == law]
            )
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points],
                        marker="o", lw=1.8, label=law)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("mini-batch size per GPU")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(title="lr law")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)


# ───────────────────────────── scaling laws ─────────────────────────────────
def compute_lr(law, batch_size, world_size):
    """lr for the chosen law. L0 = BASE_LR, K = world_size, G0 = 10000."""
    if law == "baseline":
        return BASE_LR
    if law == "linearK":
        return BASE_LR * world_size
    if law == "sqrtK":
        return BASE_LR * math.sqrt(world_size)
    if law == "linearG":
        return BASE_LR * (batch_size / G0)
    if law == "sqrtG":
        return BASE_LR * math.sqrt(batch_size / G0)
    raise ValueError(f"unknown lr law: {law}")


def model_name(law, lr, batch_size, world_size):
    """A name that is safe to use as a directory name."""
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

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, True


def detect_grad_sync(world_size, rank):
    """Detects whether stock DDP can handle our loss.

    loss_function calls the model three times per backward pass, and the
    residual is built with create_graph=True (a double backward).
    DistributedDataParallel expects a single forward and usually fails on this.
    We try it, and if it fails we fall back to an explicit all_reduce of the
    gradients (either way the averaging happens exactly once).
    """
    if world_size == 1:
        return "single"

    set_seed()
    probe = DistributedDataParallel(PINN_BL().to(device),
                                    device_ids=[device.index])
    data = make_data()
    ok = torch.tensor([1], device=device)
    try:
        total, _, _, _ = loss_function(probe, **data)
        total.backward()
        if all(p.grad is None for p in probe.parameters()):
            ok = torch.tensor([0], device=device)
    except RuntimeError as exc:
        if rank == 0:
            print(f"  stock DDP failed: {exc}", flush=True)
        ok = torch.tensor([0], device=device)

    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    del probe
    return "ddp" if ok.item() == 1 else "manual"


def build_model(world_size, grad_sync):
    """Model with identical initial weights on every process."""
    set_seed()
    model = PINN_BL().to(device)
    if world_size == 1:
        return model, model

    # DDP is useful even in "manual" mode: it broadcasts the initial weights.
    ddp = DistributedDataParallel(model, device_ids=[device.index])
    return (ddp, model) if grad_sync == "ddp" else (model, model)


def check_grad_sync(model, data, x_local, t_local, world_size, rank, grad_sync):
    """Checks that gradients agree across ranks after a backward pass."""
    if world_size == 1:
        return None

    model.zero_grad()
    batch = dict(data)
    batch["x_f"] = x_local[:64].clone().requires_grad_(True)
    batch["t_f"] = t_local[:64].clone().requires_grad_(True)
    total, _, _, _ = loss_function(model, **batch)
    total.backward()

    if grad_sync == "manual":
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    norm = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters()
                          if p.grad is not None))
    gathered = [torch.zeros_like(norm) for _ in range(world_size)]
    dist.all_gather(gathered, norm)
    model.zero_grad()

    values = [g.item() for g in gathered]
    spread = max(values) - min(values)
    if rank == 0:
        print(f"  gradient norms per rank: "
              f"{', '.join(f'{v:.10e}' for v in values)}  (spread {spread:.2e})",
              flush=True)
    return spread


# ────────────────────────── a single experiment ─────────────────────────────
def run_config(law, batch_size, args, rank, world_size, grad_sync,
               exact_x, exact_sol, out_root):
    """Trains one configuration and saves its results."""
    lr = compute_lr(law, batch_size, world_size)
    name = model_name(law, lr, batch_size, world_size)

    model, raw_model = build_model(world_size, grad_sync)
    data = make_data()
    x_local, t_local = shard_collocation(data, rank, world_size)

    history, elapsed = train_adam_ddp(
        model, data, x_local, t_local, args.adam_steps, lr, batch_size,
        rank, world_size, grad_sync, log_every=args.log_every,
    )

    if rank != 0:
        return None

    l2 = compute_l2(raw_model, exact_x, exact_sol)
    l2_mean = float(np.mean(list(l2.values())))

    run_dir = os.path.join(out_root, law, f"bs{batch_size}")
    os.makedirs(run_dir, exist_ok=True)
    torch.save(raw_model.state_dict(), os.path.join(run_dir, "model.pth"))

    result = {
        "model_name": name,
        "lr_law": law,
        "batch_size_per_gpu": batch_size,
        "global_batch_size": batch_size * world_size,
        "n_f": N_F,
        "num_gpus": world_size,
        "learning_rate": lr,
        "adam_steps": len(history),
        "training_time_sec": elapsed,
        "l2_error": l2_mean,
        "l2_t0.25": l2[0.25],
        "l2_t0.5": l2[0.5],
        "l2_t0.75": l2[0.75],
        "l2_t1.0": l2[1.0],
        "final_loss": history[-1],
        "lambda_f": LAMBDA_F,
        "lambda_ic": LAMBDA_IC,
        "lambda_bc": LAMBDA_BC,
        "n_parameters": sum(p.numel() for p in raw_model.parameters()),
        "seed": SEED,
        "scheduler": "StepLR(step_size=1000, gamma=0.7)",
        "grad_sync": grad_sync,
        "local_points": x_local.shape[0],
        "batch_exceeds_local_data": batch_size > x_local.shape[0],
        "unique_points_per_batch": min(batch_size, x_local.shape[0]),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    with open(os.path.join(run_dir, "model_result.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    title = f"{law}  lr={lr:.3e}  bs={batch_size}  |  L2 = {l2_mean:.4e}"
    plot_loss(history, os.path.join(run_dir, "loss.png"), title=title)
    plot_solution(raw_model, exact_x, exact_sol, l2,
                  os.path.join(run_dir, "solution.png"), title=title)

    print(f"  done: {elapsed:7.2f} s | L2 = {l2_mean:.4e} | loss = {history[-1]:.4e}",
          flush=True)
    return result


# ──────────────────────────────── tables ────────────────────────────────────
def write_summary(rows, sweep_time, out_dir):
    """Mean L2 across the sweep, overall and per scaling law."""
    l2 = np.array([r["l2_error"] for r in rows])
    times = np.array([r["training_time_sec"] for r in rows])

    per_law = {}
    for law in LR_LAWS:
        sub = [r["l2_error"] for r in rows if r["lr_law"] == law]
        if sub:
            per_law[law] = {
                "n_experiments": len(sub),
                "l2_mean": float(np.mean(sub)),
                "l2_min": float(np.min(sub)),
                "l2_max": float(np.max(sub)),
            }

    best = min(rows, key=lambda r: r["l2_error"])
    summary = {
        "n_experiments": len(rows),
        "adam_steps": rows[0]["adam_steps"],
        "num_gpus": rows[0]["num_gpus"],
        "batch_sizes": sorted({r["batch_size_per_gpu"] for r in rows}),
        "l2_mean_all_experiments": float(l2.mean()),
        "l2_std_all_experiments": float(l2.std()),
        "l2_min_all_experiments": float(l2.min()),
        "l2_max_all_experiments": float(l2.max()),
        "l2_median_all_experiments": float(np.median(l2)),
        "time_mean_sec": float(times.mean()),
        "time_min_sec": float(times.min()),
        "time_max_sec": float(times.max()),
        "sweep_total_sec": sweep_time,
        "best_model": best["model_name"],
        "best_l2": best["l2_error"],
        "l2_mean_per_law": per_law,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def write_tables(rows, out_dir):
    """results_all.csv plus time_summary.csv sorted by training time."""
    fields = list(rows[0].keys())

    with open(os.path.join(out_dir, "results_all.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ranked = []
    for i, row in enumerate(sorted(rows, key=lambda r: r["training_time_sec"]), 1):
        ranked.append({"rank": i, **row})

    with open(os.path.join(out_dir, "time_summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank"] + fields)
        writer.writeheader()
        writer.writerows(ranked)

    return ranked


# ────────────────────── single-GPU baseline reference ───────────────────────
def run_baseline_reference(args):
    """Full-batch Adam on one GPU, the reference point for the whole sweep.

    In sequential/<steps>/results.txt the L2 error is measured after L-BFGS, so
    Adam-only runs cannot be compared against it. This run uses the same number
    of Adam steps, the full batch and lr = 1e-3.

    The output directory is created here because main returns from this branch
    before it reaches the makedirs of the full sweep.
    """
    os.makedirs(args.out, exist_ok=True)
    print(f"baseline: 1 GPU, full batch ({N_F} points), lr = {BASE_LR:g}, "
          f"{args.adam_steps} Adam steps", flush=True)

    set_seed()
    model = PINN_BL().to(device)
    data = make_data()
    x_local, t_local = data["x_f"].detach(), data["t_f"].detach()

    history, elapsed = train_adam_ddp(
        model, data, x_local, t_local, args.adam_steps, BASE_LR, N_F,
        rank=0, world_size=1, grad_sync="single", log_every=args.log_every,
    )

    exact_x, exact_sol = load_exact()
    l2 = compute_l2(model, exact_x, exact_sol)
    l2_mean = float(np.mean(list(l2.values())))

    result = {
        "model_name": f"baseline_reference_1gpu_fullbatch_lr{BASE_LR:.3e}",
        "lr_law": "reference",
        "batch_size_per_gpu": N_F,
        "global_batch_size": N_F,
        "n_f": N_F,
        "num_gpus": 1,
        "learning_rate": BASE_LR,
        "adam_steps": len(history),
        "training_time_sec": elapsed,
        "l2_error": l2_mean,
        "l2_t0.25": l2[0.25],
        "l2_t0.5": l2[0.5],
        "l2_t0.75": l2[0.75],
        "l2_t1.0": l2[1.0],
        "final_loss": history[-1],
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    os.makedirs(args.out, exist_ok=True)
    ref_path = os.path.join(args.out, "baseline_reference.json")
    with open(ref_path, "w") as fh:
        json.dump(result, fh, indent=2)

    torch.save(model.state_dict(),
               os.path.join(args.out, "baseline_reference_model.pth"))
    print(f"\ntime {elapsed:.2f} s | mean L2 = {l2_mean:.6e} | "
          f"loss {history[-1]:.4e}")
    print(f"saved to {args.out}/: baseline_reference.json, "
          f"baseline_reference_model.pth")


# ───────────────────────────────── main ─────────────────────────────────────
def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(
        description="Parallel PINN for the 1D Buckley-Leverett problem")
    parser.add_argument("--adam-steps", type=int, default=ADAM_STEPS,
                        help="optimizer updates per configuration")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out", default=EXPERIMENTS_DIR)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--verify", action="store_true",
                        help="run a single configuration and print its parameters")
    parser.add_argument("--verify-law", default="baseline", choices=LR_LAWS)
    parser.add_argument("--verify-bs", type=int, default=1024)
    parser.add_argument("--baseline-reference", action="store_true",
                        help="single full-batch run on 1 GPU")
    args = parser.parse_args()
    DATA_DIR = args.data_dir

    rank, world_size, distributed = setup_ddp()

    if args.baseline_reference:
        if rank == 0:
            run_baseline_reference(args)
        return

    grad_sync = detect_grad_sync(world_size, rank)

    if rank == 0:
        print("=" * 70)
        print(f"GPU count            : {world_size}")
        print(f"N_F                  : {N_F}")
        print(f"Adam steps           : {args.adam_steps}")
        print(f"base lr (L0)         : {BASE_LR:g}")
        print(f"gradient sync        : {grad_sync}")
        if device.type == "cuda":
            print(f"device               : {torch.cuda.get_device_name(0)}")
        print("=" * 70, flush=True)

    exact_x, exact_sol = load_exact()
    out_root = args.out
    if rank == 0:
        os.makedirs(out_root, exist_ok=True)

    # ── single configuration check ──
    if args.verify:
        law, batch_size = args.verify_law, args.verify_bs
        lr = compute_lr(law, batch_size, world_size)
        if rank == 0:
            print(f"\nsingle configuration check")
            print(f"  GPU count       : {world_size}")
            print(f"  N_F             : {N_F}")
            print(f"  batch size      : {batch_size} per GPU "
                  f"(global {batch_size * world_size})")
            print(f"  LR law          : {law}")
            print(f"  calculated LR   : {lr:.6e}")
            print(f"  Adam steps      : {args.adam_steps}\n", flush=True)

        model, raw_model = build_model(world_size, grad_sync)
        data = make_data()
        x_local, t_local = shard_collocation(data, rank, world_size)
        check_grad_sync(model, data, x_local, t_local, world_size, rank, grad_sync)

        run_config(law, batch_size, args, rank, world_size, grad_sync,
                   exact_x, exact_sol, out_root)
        if distributed:
            dist.destroy_process_group()
        return

    # ── full sweep ──
    total_configs = len(LR_LAWS) * len(BATCH_SIZES)
    rows = []
    failed = []
    sweep_start = time.perf_counter()

    for i, law in enumerate(LR_LAWS):
        for j, batch_size in enumerate(BATCH_SIZES):
            n = i * len(BATCH_SIZES) + j + 1
            lr = compute_lr(law, batch_size, world_size)
            if rank == 0:
                print(f"\n[{n}/{total_configs}] {law}  bs={batch_size}  "
                      f"lr={lr:.6e}", flush=True)
            try:
                result = run_config(law, batch_size, args, rank, world_size,
                                    grad_sync, exact_x, exact_sol, out_root)
                if result is not None:
                    rows.append(result)
            except Exception as exc:                       # noqa: BLE001
                if rank == 0:
                    print(f"  ERROR: {exc!r}", flush=True)
                    failed.append((law, batch_size, repr(exc)))
            if distributed:
                dist.barrier()

    sweep_time = time.perf_counter() - sweep_start

    if rank == 0 and rows:
        ranked = write_tables(rows, out_root)
        plot_sweep(rows, out_root)
        summary = write_summary(rows, sweep_time, out_root)

        print("\n" + "=" * 70)
        print(f"configurations done : {len(rows)}/{total_configs}"
              f"{f', failed: {len(failed)}' if failed else ''}")
        print(f"total sweep time    : {sweep_time:.1f} s ({sweep_time / 60:.1f} min)")
        print("=" * 70)

        print("\nfastest:")
        header = f"{'#':>3} {'law':<10}{'bs':>6}{'lr':>12}{'time, s':>11}{'L2':>12}"
        print(header)
        print("-" * len(header))
        for r in ranked[:5]:
            print(f"{r['rank']:>3} {r['lr_law']:<10}{r['batch_size_per_gpu']:>6}"
                  f"{r['learning_rate']:>12.3e}{r['training_time_sec']:>11.2f}"
                  f"{r['l2_error']:>12.4e}")

        print("\nbest by L2:")
        print(header)
        print("-" * len(header))
        for r in sorted(rows, key=lambda x: x["l2_error"])[:5]:
            print(f"{'':>3} {r['lr_law']:<10}{r['batch_size_per_gpu']:>6}"
                  f"{r['learning_rate']:>12.3e}{r['training_time_sec']:>11.2f}"
                  f"{r['l2_error']:>12.4e}")

        print(f"\nmean L2 over all {summary['n_experiments']} experiments: "
              f"{summary['l2_mean_all_experiments']:.6e}")
        print(f"  median {summary['l2_median_all_experiments']:.4e} | "
              f"range {summary['l2_min_all_experiments']:.4e} ... "
              f"{summary['l2_max_all_experiments']:.4e}")
        print("\n  mean L2 per law:")
        for law, st in summary["l2_mean_per_law"].items():
            print(f"    {law:<10} {st['l2_mean']:.4e}   "
                  f"({st['n_experiments']} experiments, "
                  f"best {st['l2_min']:.4e})")

        print(f"\nmodels  : {out_root}/<law>/bs<N>/model.pth")
        print(f"tables  : {out_root}/results_all.csv, {out_root}/time_summary.csv")
        print(f"summary : {out_root}/summary.json")
        print(f"plots   : {out_root}/l2_vs_batch.png, "
              f"{out_root}/time_vs_batch.png", flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
