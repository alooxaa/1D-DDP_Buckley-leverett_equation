"""Mass balance error for the 1D Buckley-Leverett problem.

This metric never looks at the analytic solution. It checks whether the network
itself obeys the conservation law it was trained to solve. Integrating

    s_t + d/dx [ f(s) - eps * s_x ] = 0,   f(s) = s^2 / (s^2 + (1-s)^2 / M)

over x from 0 to 1 collapses the divergence term into boundary fluxes:

    M(t)   = int_0^1 s(x, t) dx                mass inside the domain
    F(x,t) = f(s) - eps * s_x                  total flux (advective + diffusive)
    E(t)   = [M(t) - M(0)] - int_0^t [F(0,tau) - F(1,tau)] dtau

E(t) is the balance residual: how much water is in the domain minus how much
entered through x = 0 and left through x = 1. For the exact solution E(t) = 0 at
every t. For a PINN it is nonzero, and its magnitude is a reference-free measure
of how physical the model is.

The relative form divides |E(t)| by the injected volume in pore volumes. That
volume comes from the boundary condition, not from the network: s(0,t) = 1 gives
f = 1, so exactly t pore volumes have been injected by time t. Dividing by the
influx the network itself predicts does not work, because a poorly trained model
underestimates its own influx and the ratio explodes for reasons unrelated to
conservation. The divisor is still small as t -> 0, so relative aggregates use
only t >= T_REL_MIN, while absolute ones cover the whole window. The network's
own influx is still reported in injected_final: for a trained model it is
close to 1.

M(0) is taken from the network rather than set to zero from the initial
condition, so that E(t) measures conservation alone and does not mix in the
initial-condition error. The network's departure from s(x, 0) = 0 is reported
separately in mass_t0 (ideally zero).

This module is shared by train_ddp_lbfgs.py and eval_metrics.py; a copy of the
same metric lives inside train_bl.py, like the rest of the physics that is
duplicated between the sequential and the DDP versions.
"""

import numpy as np
import torch

M = 1.0               # viscosity ratio, same as during training
EPS = 0.0025          # artificial diffusion coefficient, same as during training

N_T = 201             # time nodes on [0, 1] (t = 0 is needed for M(0))
N_X = 401             # x nodes for the mass integral
T_REL_MIN = 0.1       # below this t the injected volume is small and the ratio is noisy


def fractional_flow(s, m=M):
    """f(s), the same Buckley-Leverett flux function used in the PDE residual."""
    return s**2 / (s**2 + (1 - s)**2 / m)


def _cumtrapz(y, x):
    """Cumulative trapezoidal integral: out[i] = int_{x0}^{xi} y dx."""
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _mass(model, t_grid, x_grid, device, dtype):
    """M(t) = int_0^1 s dx over the whole grid in a single forward pass."""
    tt, xx = np.meshgrid(t_grid, x_grid, indexing="ij")
    points = np.stack([xx.ravel(), tt.ravel()], axis=1)   # input order: (x, t)
    with torch.no_grad():
        s = model(torch.tensor(points, dtype=dtype, device=device))
    s = s.reshape(t_grid.size, x_grid.size).double().cpu().numpy()
    return np.trapezoid(s, x_grid, axis=1)


def _boundary_flux(model, t_grid, x_val, device, dtype, m=M, eps=EPS):
    """F(x_val, t) = f(s) - eps * s_x along the line x = x_val.

    The derivative comes from autograd rather than a finite difference: the
    network provides it exactly, whereas a difference would add its own error
    near the front.
    """
    x = torch.full((t_grid.size, 1), float(x_val), dtype=dtype, device=device,
                   requires_grad=True)
    t = torch.tensor(t_grid.reshape(-1, 1), dtype=dtype, device=device)

    s = model(torch.cat([x, t], dim=1))
    s_x = torch.autograd.grad(s, x, grad_outputs=torch.ones_like(s))[0]
    flux = fractional_flow(s, m) - eps * s_x
    return flux.detach().double().cpu().numpy().ravel()


def mass_balance(model, n_t=N_T, n_x=N_X, m=M, eps=EPS):
    """Mass balance error on a grid of n_t time levels by n_x points in x.

    Returns absolute values (in pore volumes) and values relative to the volume
    injected according to the boundary condition, that is, relative to t.
    Relative aggregates start at t >= T_REL_MIN: before that too little has been
    injected to divide by.
    """
    model.eval()
    param = next(model.parameters())
    device, dtype = param.device, param.dtype

    t_grid = np.linspace(0.0, 1.0, n_t)
    x_grid = np.linspace(0.0, 1.0, n_x)

    mass = _mass(model, t_grid, x_grid, device, dtype)
    flux_in = _boundary_flux(model, t_grid, 0.0, device, dtype, m, eps)
    flux_out = _boundary_flux(model, t_grid, 1.0, device, dtype, m, eps)

    injected = _cumtrapz(flux_in, t_grid)      # cumulative influx through x = 0
    produced = _cumtrapz(flux_out, t_grid)     # cumulative outflux through x = 1

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
        "mass_t0": float(mass[0]),             # ideally 0, since s(x, 0) = 0
        "mass_final": float(mass[-1]),
        "injected_final": float(injected[-1]),
        "produced_final": float(produced[-1]),
    }


def fmt(m):
    """One-line summary for logs."""
    return (f"mass balance: rel {m['mass_balance_rel_mean']:.4e} (mean) / "
            f"{m['mass_balance_rel_final']:.4e} (t=1) | "
            f"abs {m['mass_balance_mean']:.4e}")
