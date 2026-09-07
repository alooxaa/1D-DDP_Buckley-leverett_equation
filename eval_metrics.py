"""Error and mass balance metrics for every trained model.

Everything is evaluated on one common inference grid and written back into the
json files of the finished experiments, so that all runs are compared using the
same numbers.

Metrics against the analytic solution:

    t = linspace(0.01, 1.0, 100)   t = 0 is excluded: there the exact solution
                                   degenerates into a step at a single point
    x = linspace(0, 1, 200)        inference points, unused during training

    l2_norm  ||pred - exact||2 / ||exact||2 over the whole grid at once
    l2_mean  mean relative L2 over the 100 time slices
    l_inf    max|pred - exact| over the whole grid

Mass balance (needs no reference, see mass_balance.py):

    t = linspace(0, 1, 201), x = linspace(0, 1, 401)

    mass_balance_rel_mean   |E(t)| / t (injected volume from the BC), t >= 0.1
    mass_balance_rel_final  the same at t = 1
    mass_balance_mean/max   |E(t)| in pore volumes, mean and maximum
    mass_t0                 mass in the domain at t = 0, ideally 0

What gets updated, for both sweeps (5000 and 15000 Adam steps) and both modes:

    ddp_adam/{5000,15000}/*/*/model_result.json         DDP, Adam only
    ddp_adam_lbfgs/{5000,15000}/*/*/model_result.json   DDP, Adam + L-BFGS
    ddp_adam/{5000,15000}/baseline_reference.json       reference points
    every results_all.csv                               new columns
    sequential/metrics.json                             sequential models
    metrics_summary.json                                summary

Usage (no GPU needed, inference is cheap):
    CUDA_VISIBLE_DEVICES="" python eval_metrics.py
"""

import csv
import glob
import json
import os

import numpy as np
import torch

import mass_balance as mb
import train_ddp as td

N_T = 100
T_GRID = np.linspace(0.01, 1.0, N_T)
X_GRID = np.linspace(0, 1, 200)

# Columns this script appends to results_all.csv and to model_result.json.
L2_KEYS = ("l2_norm", "l2_mean", "l_inf")
MB_KEYS = ("mass_balance_rel_mean", "mass_balance_rel_final",
           "mass_balance_mean", "mass_balance_max", "mass_t0")
CSV_KEYS = L2_KEYS + MB_KEYS

RUNS = {
    steps: {
        # DDP, Adam only
        "adam": {"csv": f"ddp_adam/{steps}/results_all.csv",
                 "dir": f"ddp_adam/{steps}",
                 "ref_json": f"ddp_adam/{steps}/baseline_reference.json",
                 "ref_model": f"ddp_adam/{steps}/baseline_reference_model.pth"},
        # DDP, Adam + L-BFGS
        "lbfgs": {"csv": f"ddp_adam_lbfgs/{steps}/results_all.csv",
                  "dir": f"ddp_adam_lbfgs/{steps}"},
        # sequential version, Adam + L-BFGS on 1 GPU
        "seq": f"sequential/{steps}/model.pth",
    }
    for steps in (5000, 15000)
}

SEQ_METRICS = "sequential/metrics.json"


def load_model(path):
    """Loads the weights. Models produced by L-BFGS are stored in float64."""
    state = torch.load(path, map_location=td.device, weights_only=True)
    dtype = next(iter(state.values())).dtype
    model = td.PINN_BL().to(td.device).to(dtype)
    model.load_state_dict(state)
    model.eval()
    return model


def compute_metrics(model, exact_x, exact_sol):
    """Errors against the reference plus the mass balance residual.

    First the prediction and the reference are assembled in full (100 x 200) and
    used to compute the global relative L2, the mean L2 over slices and the
    L-infinity norm. A separate pass then computes the mass balance residual,
    which needs no reference at all.
    """
    pred = np.empty((N_T, X_GRID.size))
    ref = np.empty((N_T, X_GRID.size))
    for i, t_val in enumerate(T_GRID):
        pred[i] = td.predict(model, t_val, X_GRID)
        ref[i] = td.exact_at(t_val, X_GRID, exact_x, exact_sol)

    diff = pred - ref
    # relative L2 at each time level
    per_time = np.linalg.norm(diff, axis=1) / np.linalg.norm(ref, axis=1)

    return {
        "l2_norm": float(np.linalg.norm(diff) / np.linalg.norm(ref)),
        "l2_mean": float(per_time.mean()),
        "l_inf": float(np.abs(diff).max()),
        "l2_std": float(per_time.std()),
        "l2_max_slice": float(per_time.max()),
        **mb.mass_balance(model),
    }


def fmt(m):
    return (f"L2 norm {m['l2_norm']:.4e} | L2 mean {m['l2_mean']:.4e} | "
            f"Linf {m['l_inf']:.4e} | balance {m['mass_balance_rel_mean']:.4e}")


def eval_sweep(paths, exact_x, exact_sol):
    """Recomputes every configuration of one sweep: model json files and the table.

    Returns (metrics keyed by model name, table rows). The rows are needed later
    for the summary and keep the same order as results_all.csv.
    """
    by_name = {}
    for res_path in sorted(glob.glob(os.path.join(paths["dir"], "*", "*",
                                                  "model_result.json"))):
        run_dir = os.path.dirname(res_path)
        metrics = compute_metrics(
            load_model(os.path.join(run_dir, "model.pth")), exact_x, exact_sol
        )
        result = json.load(open(res_path))
        result.update(metrics)
        with open(res_path, "w") as fh:
            json.dump(result, fh, indent=2)
        by_name[result["model_name"]] = metrics

    # ── new columns in the shared table ──
    rows = list(csv.DictReader(open(paths["csv"])))
    for row in rows:
        got = by_name[row["model_name"]]
        for key in CSV_KEYS:
            row[key] = f"{got[key]:.6e}"
    with open(paths["csv"], "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return by_name, rows


def sweep_stats(by_name, rows):
    """Mean, minimum and maximum over the sweep, plus the best model."""
    arr = {key: np.array([by_name[r["model_name"]][key] for r in rows])
           for key in CSV_KEYS}
    best_l2 = min(rows, key=lambda r: by_name[r["model_name"]]["l2_mean"])
    best_mb = min(rows,
                  key=lambda r: by_name[r["model_name"]]["mass_balance_rel_mean"])

    return {
        "n_experiments": len(rows),
        "mean_over_experiments": {k: float(v.mean()) for k, v in arr.items()},
        "min_over_experiments": {k: float(v.min()) for k, v in arr.items()},
        "max_over_experiments": {k: float(v.max()) for k, v in arr.items()},
        "best_model_by_l2_mean": best_l2["model_name"],
        "best_metrics": by_name[best_l2["model_name"]],
        "best_model_by_mass_balance": best_mb["model_name"],
        "best_mass_balance": by_name[best_mb["model_name"]]["mass_balance_rel_mean"],
    }


def print_sweep(title, by_name, rows, stats):
    mean = stats["mean_over_experiments"]
    print(f"  {title}: {len(rows)} configurations")
    print(f"    best by L2 ({stats['best_model_by_l2_mean']}):")
    print(f"      {fmt(stats['best_metrics'])}")
    print(f"    mean         : L2 norm {mean['l2_norm']:.4e} | "
          f"L2 mean {mean['l2_mean']:.4e} | Linf {mean['l_inf']:.4e}")
    print(f"    mass balance : rel mean {mean['mass_balance_rel_mean']:.4e} | "
          f"at t=1 {mean['mass_balance_rel_final']:.4e} | "
          f"abs {mean['mass_balance_mean']:.4e}")
    print(f"    best balance : {stats['best_mass_balance']:.4e} "
          f"({stats['best_model_by_mass_balance']})", flush=True)


def main():
    exact_x, exact_sol = td.load_exact()
    print(f"device: {td.device} | error grid {N_T} x {X_GRID.size} | "
          f"balance grid {mb.N_T} x {mb.N_X}\n", flush=True)

    summary = {
        "n_times": N_T,
        "n_x": int(X_GRID.size),
        "t_from": float(T_GRID[0]),
        "t_to": float(T_GRID[-1]),
        "mass_balance_grid": {"n_t": mb.N_T, "n_x": mb.N_X},
        "definitions": {
            "l2_norm": "||pred-exact||2 / ||exact||2 over the whole grid",
            "l2_mean": "mean relative L2 over the 100 time slices",
            "l_inf": "max|pred-exact| over the whole grid",
            "mass_balance_mean": "mean |E(t)| in pore volumes, "
                                 "E(t) = [M(t)-M(0)] - int_0^t [F(0)-F(1)] dtau",
            "mass_balance_rel_mean": "mean |E(t)| / t over t >= 0.1; "
                                     "t is the injected volume implied by s(0,t)=1",
            "mass_balance_rel_final": "|E(1)| / injected volume at t = 1",
            "mass_t0": "mass in the domain at t = 0, ideally 0",
        },
        "runs": {},
    }
    seq_metrics = {}

    for steps, paths in RUNS.items():
        print(f"=== sweep of {steps} Adam steps ===", flush=True)

        # ── DDP, Adam only ──
        adam_by_name, adam_rows = eval_sweep(paths["adam"], exact_x, exact_sol)
        adam_stats = sweep_stats(adam_by_name, adam_rows)

        # ── DDP, Adam + L-BFGS ──
        lbfgs_by_name, lbfgs_rows = eval_sweep(paths["lbfgs"], exact_x, exact_sol)
        lbfgs_stats = sweep_stats(lbfgs_by_name, lbfgs_rows)

        # ── reference point: 1 GPU, full batch, Adam only ──
        ref_metrics = compute_metrics(load_model(paths["adam"]["ref_model"]),
                                      exact_x, exact_sol)
        ref = json.load(open(paths["adam"]["ref_json"]))
        ref.update(ref_metrics)
        with open(paths["adam"]["ref_json"], "w") as fh:
            json.dump(ref, fh, indent=2)

        # ── sequential version: Adam + L-BFGS ──
        seq_metrics[str(steps)] = {
            "model": paths["seq"],
            "mode": "sequential Adam + L-BFGS (1 GPU)",
            **compute_metrics(load_model(paths["seq"]), exact_x, exact_sol),
        }

        summary["runs"][str(steps)] = {
            # top-level keys are kept as before: the Adam-only sweep
            **adam_stats,
            "baseline_reference": ref_metrics,
            "sequential_adam_lbfgs": seq_metrics[str(steps)],
            "ddp_adam_only": adam_stats,
            "ddp_adam_lbfgs": lbfgs_stats,
        }

        print_sweep("DDP, Adam only", adam_by_name, adam_rows, adam_stats)
        print_sweep("DDP, Adam + L-BFGS", lbfgs_by_name, lbfgs_rows, lbfgs_stats)
        print(f"  baseline (1 GPU Adam): {fmt(ref_metrics)}")
        print(f"  sequential Adam+LBFGS: {fmt(seq_metrics[str(steps)])}\n", flush=True)

    with open("metrics_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(SEQ_METRICS, "w") as fh:
        json.dump(seq_metrics, fh, indent=2)

    print("saved:")
    print("  metrics_summary.json                   summary of both sweeps")
    print("  sequential/metrics.json                sequential models")
    print("  l2_* / mass_balance_* fields           in every model_result.json,")
    print("                                         in baseline_reference.json")
    print("                                         and as columns in results_all.csv")


if __name__ == "__main__":
    main()
