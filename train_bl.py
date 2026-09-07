"""
PINN for the 1D Buckley-Leverett equation (sequential, single GPU).

Equation:
    ds/dt + df/ds * ds/dx - eps * d2s/dx2 = 0,   x in [0, 1],  t in [0, 1]
    f(s) = s^2 / (s^2 + (1 - s)^2 / M)           Buckley-Leverett flux, M = 1
    eps = 0.0025                                 artificial diffusion

    initial condition:   Sw(x, 0) = 0
    boundary condition:  Sw(0, t) = 1

Training runs in two stages, Adam followed by L-BFGS. The script measures the
time and the number of steps of each stage, computes the error against the
analytic solution, and writes the model, the plots and a report into --out.

Usage:
    python train_bl.py
    python train_bl.py --adam-epochs 200 --lbfgs-max-iter 200   # quick check
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

# ─────────────────────────────── settings ───────────────────────────────────
N_F = 10000           # number of collocation points
N_U = 300             # initial and boundary condition points (split evenly)

ADAM_EPOCHS = 5000
LBFGS_MAX_ITER = 20000

M = 1.0               # viscosity ratio
EPS = 0.0025          # artificial diffusion coefficient
SEED = 42

T_EVAL = [0.25, 0.5, 0.75, 1.0]   # time levels used for the L2 report and plots
N_T_MB = 201                      # time nodes for the mass balance metric
N_X_MB = 401                      # x nodes for the mass integral
T_REL_MIN = 0.1                   # below this t the injected volume is small and the ratio is noisy
RESULTS_DIR = "results"
DATA_DIR = "data"

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


# ──────────────────────────────── training ──────────────────────────────────
def train_adam(model, data, epochs):
    """Stage one: Adam. Returns the loss history and the elapsed time."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.7)
    history = []

    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()
        total, loss_pde, loss_ic, loss_bc = loss_function(model, **data)
        total.backward()
        optimizer.step()
        scheduler.step()
        history.append(total.item())

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"  Adam {epoch:>6d} | loss={total.item():.4e} "
                  f"pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} "
                  f"bc={loss_bc.item():.3e}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    return history, time.perf_counter() - start


def train_lbfgs(model, data, max_iter):
    """Stage two: L-BFGS. Returns the loss history, the time and the iteration count.

    lr is 1.0 rather than something small because strong_wolfe picks the actual
    step length. With a small lr the trial step becomes so short that the
    optimizer immediately hits tolerance_change and stops on the second
    iteration.
    """
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

        if len(history) % 1000 == 0:
            print(f"  L-BFGS {len(history):>6d} | loss={total.item():.4e} "
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
    return history, elapsed, n_iter


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


def mass_balance(model, n_t=N_T_MB, n_x=N_X_MB):
    """Mass balance error: a conservation check that needs no reference solution.

    Integrating s_t + d/dx [f(s) - eps * s_x] = 0 over x from 0 to 1 gives

        M(t)   = int_0^1 s(x, t) dx                mass inside the domain
        F(x,t) = f(s) - eps * s_x                  total flux at the boundary
        E(t)   = [M(t) - M(0)] - int_0^t [F(0,tau) - F(1,tau)] dtau

    E(t) is how much water is in the domain minus how much entered through
    x = 0 and left through x = 1. For the exact solution E(t) = 0. The relative
    form divides |E(t)| by the injected volume taken from the boundary
    condition: with s(0,t) = 1, exactly t pore volumes have been injected by
    time t. Dividing by the influx the network itself predicts does not work,
    because a poorly trained model underestimates it. Only t >= T_REL_MIN is
    used, since before that too little has been injected to divide by.

    M(0) is taken from the network, so the metric measures conservation alone
    and does not mix in the initial-condition error. The network's departure
    from s(x, 0) = 0 is reported separately in mass_t0.

    This is a copy of mass_balance.py: the physics is duplicated between the
    sequential and the DDP versions, just like f() and loss_function().
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    t_grid = np.linspace(0.0, 1.0, n_t)
    x_grid = np.linspace(0.0, 1.0, n_x)

    # M(t): predict on each time slice and integrate over x with trapezoids
    mass = np.array([np.trapezoid(predict(model, t_val, x_grid), x_grid)
                     for t_val in t_grid])

    def flux(x_val):
        """F(x_val, t) over the whole time grid; s_x comes from autograd."""
        x = torch.full((n_t, 1), float(x_val), dtype=dtype, device=device,
                       requires_grad=True)
        t = torch.tensor(t_grid.reshape(-1, 1), dtype=dtype, device=device)
        s = u_function(model, x, t)
        s_x = derivative(s, x, order=1)
        f_s = s**2 / (s**2 + (1 - s)**2 / M)      # same f(s) as in the residual
        return (f_s - EPS * s_x).detach().double().cpu().numpy().ravel()

    def cumtrapz(y):
        out = np.zeros_like(y)
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t_grid))
        return out

    injected = cumtrapz(flux(0.0))     # cumulative influx through x = 0
    produced = cumtrapz(flux(1.0))     # cumulative outflux through x = 1

    residual = (mass - mass[0]) - (injected - produced)
    abs_err = np.abs(residual)

    # divide by the injected volume implied by the BC (s(0,t) = 1 gives exactly
    # t pore volumes), not by the network's own influx: a bad model
    # underestimates the latter and that corrupts the ratio
    warm = t_grid >= T_REL_MIN
    rel_err = abs_err[warm] / t_grid[warm]

    return {
        "mass_balance_mean": float(abs_err[1:].mean()),
        "mass_balance_max": float(abs_err.max()),
        "mass_balance_final": float(abs_err[-1]),
        "mass_balance_rel_mean": float(rel_err.mean()),
        "mass_balance_rel_max": float(rel_err.max()),
        "mass_balance_rel_final": float(rel_err[-1]),
        "rel_t_from": float(T_REL_MIN),
        "mass_t0": float(mass[0]),                # ideally 0, since s(x, 0) = 0
        "mass_final": float(mass[-1]),
        "injected_final": float(injected[-1]),
        "produced_final": float(produced[-1]),
    }


# ──────────────────────────────── plots ─────────────────────────────────────
def plot_solution(model, exact_x, exact_sol, l2, path):
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

    fig.suptitle(f"1D Buckley-Leverett   |   mean L2 = {np.mean(list(l2.values())):.4e}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(adam_history, lbfgs_history, path):
    """Loss decay for each stage, logarithmic scale."""
    n_panels = 2 if lbfgs_history else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5), squeeze=False)
    axes = axes[0]

    axes[0].semilogy(adam_history, "b", lw=1.2)
    axes[0].set_title(f"Adam, {len(adam_history)} epochs")
    axes[0].set_xlabel("epoch")

    if lbfgs_history:
        axes[1].semilogy(lbfgs_history, "r", lw=1.2)
        axes[1].set_title(f"L-BFGS, {len(lbfgs_history)} closure calls")
        axes[1].set_xlabel("closure call")

    for ax in axes:
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ──────────────────────────────── report ────────────────────────────────────
def write_report(path, l2, mass, t_adam, t_lbfgs, adam_history, lbfgs_history,
                 lbfgs_iters, n_params):
    lines = [
        "1D Buckley-Leverett PINN (sequential)",
        "=" * 46,
        "",
        f"points          : {N_F} collocation, {N_U // 2} IC, {N_U // 2} BC",
        f"network         : 2-20-20x7-1, {n_params} parameters",
        f"device          : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}",
        "",
        "L2 error",
        "-" * 46,
    ]
    for t_val, err in l2.items():
        lines.append(f"  t = {t_val:<5} : {err:.6e}")
    lines += [
        f"  mean      : {np.mean(list(l2.values())):.6e}",
        "",
        "Mass balance",
        "-" * 46,
        f"  grid          : {N_T_MB} time levels x {N_X_MB} points in x",
        f"  relative from : t >= {T_REL_MIN:g}",
        f"  rel mean      : {mass['mass_balance_rel_mean']:.6e}",
        f"  rel at t = 1  : {mass['mass_balance_rel_final']:.6e}",
        f"  rel max       : {mass['mass_balance_rel_max']:.6e}",
        f"  abs mean      : {mass['mass_balance_mean']:.6e}   (pore volumes)",
        f"  abs max       : {mass['mass_balance_max']:.6e}",
        f"  mass at t = 0 : {mass['mass_t0']:.6e}   (ideally 0)",
        f"  mass at t = 1 : {mass['mass_final']:.6f}"
        f"   injected {mass['injected_final']:.6f}"
        f"   produced {mass['produced_final']:.6f}",
        "",
        "Time and steps",
        "-" * 46,
        f"  Adam      : {t_adam:8.2f} s   {len(adam_history)} epochs"
        f"   ({1000 * t_adam / len(adam_history):.2f} ms/epoch)",
        f"  L-BFGS    : {t_lbfgs:8.2f} s   {lbfgs_iters} iterations,"
        f" {len(lbfgs_history)} closure calls"
        f"   ({1000 * t_lbfgs / max(lbfgs_iters, 1):.2f} ms/iteration)"
        if lbfgs_history else "  L-BFGS    : skipped (--skip-lbfgs)",
        f"  total     : {t_adam + t_lbfgs:8.2f} s",
        "",
        "Loss",
        "-" * 46,
        f"  after Adam   : {adam_history[-1]:.6e}",
    ]
    if lbfgs_history:
        lines.append(f"  after L-BFGS : {lbfgs_history[-1]:.6e}")
    lines += [
        "",
    ]
    text = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(text)
    return text


# ───────────────────────────────── main ─────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PINN for the 1D Buckley-Leverett problem")
    parser.add_argument("--adam-epochs", type=int, default=ADAM_EPOCHS)
    parser.add_argument("--lbfgs-max-iter", type=int, default=LBFGS_MAX_ITER)
    parser.add_argument("--out", default=RESULTS_DIR, help="output directory")
    parser.add_argument("--skip-lbfgs", action="store_true",
                        help="stop after Adam and skip the second stage")
    args = parser.parse_args()

    results_dir = args.out
    os.makedirs(results_dir, exist_ok=True)
    print(f"device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

    set_seed()
    model = PINN_BL().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # ── stage 1: Adam in float32 ──
    print(f"stage 1: Adam, {args.adam_epochs} epochs")
    adam_history, t_adam = train_adam(model, make_data(torch.float32), args.adam_epochs)
    print(f"done in {t_adam:.2f} s\n")

    # ── stage 2: L-BFGS in float64 ──
    # In float32 the gradient noise (the residual contains the second derivative
    # s_xx) exceeds the real improvement: strong_wolfe returns a zero step and
    # the stage terminates immediately.
    if args.skip_lbfgs:
        print("stage 2: L-BFGS skipped (--skip-lbfgs)\n")
        lbfgs_history, t_lbfgs, lbfgs_iters = [], 0.0, 0
    else:
        print(f"stage 2: L-BFGS, up to {args.lbfgs_max_iter} iterations (float64)")
        model = model.double()
        lbfgs_history, t_lbfgs, lbfgs_iters = train_lbfgs(
            model, make_data(torch.float64), args.lbfgs_max_iter
        )
        print(f"done in {t_lbfgs:.2f} s, {lbfgs_iters} iterations\n")

    # ── results ──
    exact_x, exact_sol = load_exact()
    l2 = compute_l2(model, exact_x, exact_sol)
    mass = mass_balance(model)

    torch.save(model.state_dict(), os.path.join(results_dir, "model.pth"))
    plot_solution(model, exact_x, exact_sol, l2, os.path.join(results_dir, "solution.png"))
    plot_loss(adam_history, lbfgs_history, os.path.join(results_dir, "loss.png"))
    report = write_report(os.path.join(results_dir, "results.txt"), l2, mass,
                          t_adam, t_lbfgs, adam_history, lbfgs_history,
                          lbfgs_iters, n_params)

    print(report)
    print(f"saved to {results_dir}/: model.pth, solution.png, loss.png, results.txt")


if __name__ == "__main__":
    main()
