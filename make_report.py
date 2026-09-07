"""Builds the result tables and the comparison plots.

Reads the four DDP sweeps, {5000, 15000} Adam steps x {Adam only,
Adam + L-BFGS}, plus the sequential runs, and writes everything into report/:

    tables.md             all result tables in markdown
    time_comparison.png   Adam stage time: 1 GPU against the fastest and the
                          slowest configuration on 2 GPUs
    l2_comparison.png     L2 after L-BFGS: 1 GPU against the best and the worst
                          configuration on 2 GPUs

All four sweeps share one grid of 3 scaling laws x 8 batch sizes = 24
configurations, so they can be compared directly.

Usage (run after eval_metrics.py, which fills the metric columns in the csv):
    python make_report.py
"""

import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = "report"
LAWS = ["baseline", "linearK", "sqrtK"]

# Every sweep in this series uses the same batch sizes 32...4096, so there is
# nothing to exclude; the set is kept as an extension point.
EXCLUDE_BS = set()

# ── where to read results from ──
# kind: "adam"  = Adam only (column training_time_sec, no L-BFGS stage),
#       "lbfgs" = Adam + L-BFGS (columns time_adam / time_lbfgs / time_total).
SWEEPS = {
    steps: {
        "adam": f"ddp_adam/{steps}/results_all.csv",
        "lbfgs": f"ddp_adam_lbfgs/{steps}/results_all.csv",
    }
    for steps in (5000, 15000)
}

REFS = {
    steps: {"ddp_ref": f"ddp_adam/{steps}/baseline_reference.json",
            "seq_txt": f"sequential/{steps}/results.txt"}
    for steps in (5000, 15000)
}

SEQ_METRICS = "sequential/metrics.json"


# ─────────────────────────────── loading ────────────────────────────────────
def load_ddp(path, kind):
    """Rows of one sweep in a common shape, whatever the csv layout is."""
    rows = []
    for r in csv.DictReader(open(path)):
        if int(r["batch_size_per_gpu"]) in EXCLUDE_BS or r["lr_law"] not in LAWS:
            continue
        if kind == "adam":
            t_adam, t_lbfgs = float(r["training_time_sec"]), None
            t_total = t_adam
        else:
            t_adam, t_lbfgs = float(r["time_adam"]), float(r["time_lbfgs"])
            t_total = float(r["time_total"])
        rows.append({
            "law": r["lr_law"],
            "bs": int(r["batch_size_per_gpu"]),
            "lr": float(r["learning_rate"]),
            "t_adam": t_adam,
            "t_lbfgs": t_lbfgs,
            "t_total": t_total,
            "l2": float(r["l2_mean"]),            # primary metric: 100 time slices
            # L2 of the same model before the second stage: a paired comparison
            # inside one run, so no reference point has to be chosen
            "l2_before": float(r["after_adam_l2_mean"]) if kind == "lbfgs" else None,
            "linf": float(r["l_inf"]),
            "mb": float(r["mass_balance_rel_mean"]),
        })
    return rows


def parse_seq_time(path):
    """Pulls the Adam, L-BFGS and total times out of results.txt."""
    text = open(path).read()
    grab = lambda pat: float(re.search(pat, text).group(1))
    return {"t_adam": grab(r"Adam\s+:\s+([\d.]+) s"),
            "t_lbfgs": grab(r"L-BFGS\s+:\s+([\d.]+) s"),
            "t_total": grab(r"total\s+:\s+([\d.]+) s")}


def as_row(metrics, times):
    """Metrics of a sequential run or a reference point, in the same shape."""
    return {**times,
            "l2": metrics["l2_mean"],
            "linf": metrics["l_inf"],
            "mb": metrics["mass_balance_rel_mean"]}


def median_row(rows):
    """Column-wise median over a sweep.

    A median rather than a mean: every sweep contains configurations where the
    network did not converge, and those drag the mean with them. Each column is
    reduced independently, so the median time and the median L2 may belong to
    different models.
    """
    out = {}
    for k in ("t_adam", "t_lbfgs", "t_total", "l2"):
        vals = [r[k] for r in rows if r[k] is not None]
        out[k] = float(np.median(vals)) if vals else None
    return out


def name(row):
    """Configuration name in the form lr(law)_bs<size>, as used in the report."""
    return f"lr({row['law']})_bs{row['bs']}"


# ──────────────────────────────── tables ────────────────────────────────────
SEC = lambda v: "—" if v is None else f"{v:.1f}"


def header(cols):
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)


def table_by_batch(rows, kind):
    """One law, one sweep: every batch size in order.

    The Adam-only mode has a single time column; Adam + L-BFGS has three, the
    total plus each stage separately.
    """
    lbfgs = kind == "lbfgs"
    cols = ["Batch"] + (["Total, s", "Adam, s", "L-BFGS, s"] if lbfgs
                        else ["Adam, s"])
    cols += ["L2 (100time)", "Linf", "Mass balance", "Model"]
    lines = [header(cols)]

    for r in sorted(rows, key=lambda r: r["bs"]):
        times = ([SEC(r["t_total"]), SEC(r["t_adam"]), SEC(r["t_lbfgs"])]
                 if lbfgs else [SEC(r["t_adam"])])
        lines.append("| " + " | ".join(
            [str(r["bs"])] + times
            + [f"{r['l2']:.4f}", f"{r['linf']:.4f}", f"{r['mb']:.2e}",
               f"`{name(r)}`"]) + " |")
    return "\n".join(lines)


def total_line(rows, kind):
    """SUMMARY line: the best batch by L2 and how it compares on time."""
    tkey = "t_total" if kind == "lbfgs" else "t_adam"
    best = min(rows, key=lambda r: r["l2"])
    fast = min(rows, key=lambda r: r[tkey])
    times = [r[tkey] for r in rows]
    spread = 100 * (max(times) / min(times) - 1)

    text = (f"**SUMMARY.** Best by L2: `bs{best['bs']}`, L2 {best['l2']:.4f}, "
            f"mass balance {best['mb']:.2e}, time {best[tkey]:.1f} s. ")
    if spread < 15:
        text += (f"Time across all batches stays within {min(times):.1f}...{max(times):.1f} s "
                 f"({spread:.0f} % spread), so it does not depend on the batch "
                 f"size and picking a batch by time makes no sense.")
    else:
        text += (f"Time across batches ranges over {min(times):.1f}...{max(times):.1f} s; "
                 f"the fastest one, `bs{fast['bs']}` at {fast[tkey]:.1f} s, gives "
                 f"L2 {fast['l2']:.4f}, so the time saving is paid for in "
                 f"accuracy.")
    return text


def table_best_modes(data):
    """Summary: the best model by L2 in each mode of each sweep.

    The rows are not tied to a single configuration: each mode has its own
    winner chosen among all 24 models of the sweep. Rows therefore differ not
    only by the presence of L-BFGS but also by law and batch size.
    """
    cols = ["Mode", "Steps", "Adam, s", "L-BFGS, s", "Total, s", "L2 (100time)"]
    lines = [header(cols)]

    for steps in sorted(data):
        best = {kind: min(data[steps][kind], key=lambda r: r["l2"])
                for kind in ("adam", "lbfgs")}
        top = min(best.values(), key=lambda r: r["l2"])
        for kind, title in (("adam", "Only Adam"), ("lbfgs", "Adam + L-BFGS")):
            r = best[kind]
            mark = "**" if r is top else ""
            lines.append(f"| {title}, `{name(r)}` | {steps} | "
                         f"{SEC(r['t_adam'])} | {SEC(r['t_lbfgs'])} | "
                         f"{SEC(r['t_total'])} | {mark}{r['l2']:.4f}{mark} |")
    return "\n".join(lines)


def table_best_per_law(data, steps):
    """Best by L2 within each law, both modes: six models per sweep."""
    cols = ["Mode", "Model", "Adam, s", "L-BFGS, s", "Total, s",
            "L2 (100time)", "Linf", "Mass balance"]
    lines = [header(cols)]

    picked = []
    for kind, title in (("adam", "Only Adam"), ("lbfgs", "Adam + L-BFGS")):
        for law in LAWS:
            group = [r for r in data[steps][kind] if r["law"] == law]
            picked.append((title, min(group, key=lambda r: r["l2"])))

    winner = min(picked, key=lambda p: p[1]["l2"])[1]
    for title, r in picked:
        mark = "**" if r is winner else ""
        lines.append(f"| {title} | `{name(r)}` | {SEC(r['t_adam'])} | "
                     f"{SEC(r['t_lbfgs'])} | {SEC(r['t_total'])} | "
                     f"{mark}{r['l2']:.4f}{mark} | {r['linf']:.4f} | "
                     f"{r['mb']:.2e} |")
    return "\n".join(lines), winner


# ──────────────────────────────── plots ─────────────────────────────────────
def chart_three(data, picker, ylabel, title, path, fmt, logy=False,
                sharey=True):
    """Three bars per panel, one panel per sweep: 1 GPU and two extremes on 2 GPUs.

    picker(d) -> [(label, value), ...] decides what is taken from each sweep;
    time and L2 pick different configurations.

    sharey=False exists for the time chart: 5000 and 15000 steps differ by a
    factor of three, and on a shared axis the left panel collapses so that the
    differences between bars stop being readable.
    """
    steps_list = sorted(data)
    colors = ["#5b8ff9", "#2e7d32", "#c0392b"]

    fig, axes = plt.subplots(1, len(steps_list), figsize=(11, 5.5),
                             sharey=sharey)
    for ax, steps in zip(np.atleast_1d(axes), steps_list):
        labels, values = zip(*picker(data[steps]))
        bars = ax.bar(labels, values, color=colors)
        ax.tick_params(axis="x", labelsize=8.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, format(v, fmt),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{steps} Adam steps")
        ax.grid(axis="y", alpha=0.3, which="both")
        if logy:
            ax.set_yscale("log")
        ax.margins(y=0.25)

    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pick_time(d):
    """Adam stage time inside the Adam + L-BFGS pipeline, three bars.

    All three bars measure the same thing: how long the Adam stage took for a
    model that then went into L-BFGS. On the left is the sequential run on one
    card, on the right the fastest and the slowest configuration on two. L-BFGS
    is not included: it always runs on a single card and is not parallelised.
    """
    fast = min(d["lbfgs"], key=lambda r: r["t_adam"])
    slow = max(d["lbfgs"], key=lambda r: r["t_adam"])
    tag = "Adam time\n(Adam + L-BFGS)"
    return [(f"1 GPU\n{tag}", d["seq"]["t_adam"]),
            (f"2 GPU DDP, fastest\n{tag}\n{name(fast)}", fast["t_adam"]),
            (f"2 GPU DDP, slowest\n{tag}\n{name(slow)}", slow["t_adam"])]


def pick_l2(d):
    """L2 after L-BFGS: 1 GPU against the best and the worst run on 2 GPUs."""
    best = min(d["lbfgs"], key=lambda r: r["l2"])
    worst = max(d["lbfgs"], key=lambda r: r["l2"])
    return [("1 GPU\nAdam + L-BFGS", d["seq"]["l2"]),
            (f"2 GPU DDP\nbest L2\n{name(best)}", best["l2"]),
            (f"2 GPU DDP\nworst L2\n{name(worst)}", worst["l2"])]


# ───────────────────────────────── main ─────────────────────────────────────
MODE_TITLE = {"adam": "Adam only", "lbfgs": "Adam + L-BFGS"}


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    seq_all = json.load(open(SEQ_METRICS))

    data = {}
    for steps, paths in SWEEPS.items():
        ref = json.load(open(REFS[steps]["ddp_ref"]))
        ref_time = {"t_adam": ref["training_time_sec"], "t_lbfgs": None,
                    "t_total": ref["training_time_sec"]}
        data[steps] = {
            "adam": load_ddp(paths["adam"], "adam"),
            "lbfgs": load_ddp(paths["lbfgs"], "lbfgs"),
            "ref": as_row(ref, ref_time),
            "seq": as_row(seq_all[str(steps)],
                          parse_seq_time(REFS[steps]["seq_txt"])),
        }

    steps_list = sorted(data)
    parts = [
        "# DDP results, 2 GPUs: PINN for 1D Buckley-Leverett",
        "",
        "Network 2-20-20x7-1 (3021 parameters), N_F = 10000 collocation points, "
        "2x RTX 5090. Batch sizes are given **per GPU**, the global batch is "
        "twice as large. Three lr laws at K = 2 GPUs: `baseline` lr = 1e-3, "
        "`linearK` lr = 1e-3 * K, `sqrtK` lr = 1e-3 * sqrt(K). Eight batch "
        "sizes from 32 to 4096, so 8 models per law.",
        "",
        "`L2 (100time)` is the mean relative L2 over t = linspace(0.01, 1, 100) "
        "on the grid x = linspace(0, 1, 200); for models with L-BFGS this is the "
        "value after the second stage. `Linf` is max |PINN - Exact| over the "
        "whole grid. `Mass balance` is the mass balance error, which uses no "
        "reference solution: the mean of |E(t)| / t over t >= 0.1, where "
        "E(t) = [M(t) - M(0)] - int_0^t [F(0,tau) - F(1,tau)] dtau, "
        "M(t) = int_0^1 s dx and F = f(s) - eps * s_x.",
        "", "---", "",
        "# Summary: the best model of each mode", "",
        "Each row holds the best model by L2 among the 24 configurations of a "
        "sweep (3 laws x 8 batch sizes). The winners differ in law and batch "
        "size, so rows differ by more than the presence of L-BFGS. The breakdown "
        "by law is in section 3, the full sweeps in sections 1 and 2.", "",
        table_best_modes(data), "",
    ]

    # ── sections 1 and 2: law x sweep ──
    for section, kind in ((1, "adam"), (2, "lbfgs")):
        parts += ["---", "", f"# {section}. DDP, {MODE_TITLE[kind]}", ""]
        n = 0
        for law in LAWS:
            for steps in steps_list:
                n += 1
                rows = [r for r in data[steps][kind] if r["law"] == law]
                parts += [
                    f"## {section}.{n} lr({law}), {steps} Adam steps", "",
                    table_by_batch(rows, kind), "",
                    total_line(rows, kind), "",
                ]

    # ── section 3: the best models of each sweep ──
    parts += ["---", "", "# 3. Best models by L2", ""]
    for i, steps in enumerate(steps_list, 1):
        table, winner = table_best_per_law(data, steps)
        mode = ("Adam only" if winner["t_lbfgs"] is None else "Adam + L-BFGS")
        tkey = "t_adam" if winner["t_lbfgs"] is None else "t_total"
        parts += [
            f"## 3.{i} {steps} Adam steps: the best model of each law in both "
            f"modes", "",
            table, "",
            f"**SUMMARY.** Best model of the sweep: `{name(winner)}` in "
            f"{mode} mode. L2 {winner['l2']:.4f}, Linf {winner['linf']:.4f}, "
            f"mass balance {winner['mb']:.2e}, time {winner[tkey]:.1f} s.", "",
        ]

    with open(os.path.join(REPORT_DIR, "tables.md"), "w") as fh:
        fh.write("\n".join(parts))

    chart_three(data, pick_time, "Adam stage time, s",
                "Adam stage time inside the Adam + L-BFGS pipeline:\n"
                "1 GPU vs 2 GPU DDP",
                os.path.join(REPORT_DIR, "time_comparison.png"), ".1f")
    chart_three(data, pick_l2, "relative L2, 100 time slices (log scale)",
                "Accuracy after L-BFGS: 1 GPU vs 2 GPU DDP\n"
                "(relative L2 over 100 time slices)",
                os.path.join(REPORT_DIR, "l2_comparison.png"), ".4f", logy=True)

    print("\n".join(parts))
    print(f"\n\nsaved to {REPORT_DIR}/: tables.md, time_comparison.png, "
          f"l2_comparison.png")


if __name__ == "__main__":
    main()
