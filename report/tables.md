# DDP results, 2 GPUs: PINN for 1D Buckley-Leverett

Network 2-20-20x7-1 (3021 parameters), N_F = 10000 collocation points, 2x RTX 5090. Batch sizes are given **per GPU**, the global batch is twice as large. Three lr laws at K = 2 GPUs: `baseline` lr = 1e-3, `linearK` lr = 1e-3 * K, `sqrtK` lr = 1e-3 * sqrt(K). Eight batch sizes from 32 to 4096, so 8 models per law.

`L2 (100time)` is the mean relative L2 over t = linspace(0.01, 1, 100) on the grid x = linspace(0, 1, 200); for models with L-BFGS this is the value after the second stage. `Linf` is max |PINN - Exact| over the whole grid. `Mass balance` is the mass balance error, which uses no reference solution: the mean of |E(t)| / t over t >= 0.1, where E(t) = [M(t) - M(0)] - int_0^t [F(0,tau) - F(1,tau)] dtau, M(t) = int_0^1 s dx and F = f(s) - eps * s_x.

---

# Summary: the best model of each mode

Each row holds the best model by L2 among the 24 configurations of a sweep (3 laws x 8 batch sizes). The winners differ in law and batch size, so rows differ by more than the presence of L-BFGS. The breakdown by law is in section 3, the full sweeps in sections 1 and 2.

| Mode | Steps | Adam, s | L-BFGS, s | Total, s | L2 (100time) |
|---|---|---|---|---|---|
| Only Adam, `lr(sqrtK)_bs4096` | 5000 | 21.4 | — | 21.4 | 0.0895 |
| Adam + L-BFGS, `lr(baseline)_bs512` | 5000 | 20.6 | 528.9 | 549.5 | **0.0571** |
| Only Adam, `lr(sqrtK)_bs1024` | 15000 | 60.0 | — | 60.0 | 0.0670 |
| Adam + L-BFGS, `lr(sqrtK)_bs512` | 15000 | 60.0 | 480.9 | 540.9 | **0.0644** |

---

# 1. DDP, Adam only

## 1.1 lr(baseline), 5000 Adam steps

| Batch | Adam, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|
| 32 | 20.3 | 0.3984 | 0.7027 | 7.39e-02 | `lr(baseline)_bs32` |
| 64 | 20.9 | 0.4452 | 0.7014 | 1.00e-01 | `lr(baseline)_bs64` |
| 128 | 21.1 | 0.4905 | 0.6943 | 9.26e-02 | `lr(baseline)_bs128` |
| 256 | 20.3 | 0.3146 | 0.7115 | 5.31e-02 | `lr(baseline)_bs256` |
| 512 | 20.1 | 0.2897 | 0.8014 | 5.91e-03 | `lr(baseline)_bs512` |
| 1024 | 21.0 | 0.2612 | 0.7355 | 1.49e-02 | `lr(baseline)_bs1024` |
| 2048 | 21.3 | 0.2863 | 0.7068 | 3.29e-02 | `lr(baseline)_bs2048` |
| 4096 | 21.2 | 0.1830 | 0.6961 | 2.00e-02 | `lr(baseline)_bs4096` |

**SUMMARY.** Best by L2: `bs4096`, L2 0.1830, mass balance 2.00e-02, time 21.2 s. Time across all batches stays within 20.1...21.3 s (6 % spread), so it does not depend on the batch size and picking a batch by time makes no sense.

## 1.2 lr(baseline), 15000 Adam steps

| Batch | Adam, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|
| 32 | 63.1 | 0.3501 | 0.7089 | 5.60e-02 | `lr(baseline)_bs32` |
| 64 | 61.5 | 0.2981 | 0.7084 | 4.88e-02 | `lr(baseline)_bs64` |
| 128 | 62.7 | 0.5033 | 0.6946 | 7.86e-02 | `lr(baseline)_bs128` |
| 256 | 60.5 | 0.1833 | 0.6972 | 1.93e-02 | `lr(baseline)_bs256` |
| 512 | 60.3 | 0.2043 | 0.7765 | 1.78e-02 | `lr(baseline)_bs512` |
| 1024 | 61.8 | 0.1885 | 0.6955 | 1.50e-02 | `lr(baseline)_bs1024` |
| 2048 | 63.3 | 0.2855 | 0.7074 | 3.92e-02 | `lr(baseline)_bs2048` |
| 4096 | 61.6 | 0.1817 | 0.6929 | 1.73e-02 | `lr(baseline)_bs4096` |

**SUMMARY.** Best by L2: `bs4096`, L2 0.1817, mass balance 1.73e-02, time 61.6 s. Time across all batches stays within 60.3...63.3 s (5 % spread), so it does not depend on the batch size and picking a batch by time makes no sense.

## 1.3 lr(linearK), 5000 Adam steps

| Batch | Adam, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|
| 32 | 20.1 | 0.4823 | 0.7103 | 9.96e-02 | `lr(linearK)_bs32` |
| 64 | 20.4 | 0.3669 | 0.7716 | 3.06e-02 | `lr(linearK)_bs64` |
| 128 | 20.6 | 0.5175 | 0.7152 | 1.18e-01 | `lr(linearK)_bs128` |
| 256 | 20.2 | 0.1542 | 0.6879 | 1.47e-03 | `lr(linearK)_bs256` |
| 512 | 20.4 | 0.4742 | 0.7246 | 8.79e-02 | `lr(linearK)_bs512` |
| 1024 | 21.0 | 0.1644 | 0.6989 | 1.24e-02 | `lr(linearK)_bs1024` |
| 2048 | 20.0 | 0.1065 | 0.6669 | 5.68e-03 | `lr(linearK)_bs2048` |
| 4096 | 20.0 | 0.1754 | 0.6967 | 1.69e-02 | `lr(linearK)_bs4096` |

**SUMMARY.** Best by L2: `bs2048`, L2 0.1065, mass balance 5.68e-03, time 20.0 s. Time across all batches stays within 20.0...21.0 s (5 % spread), so it does not depend on the batch size and picking a batch by time makes no sense.

## 1.4 lr(linearK), 15000 Adam steps

| Batch | Adam, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|
| 32 | 59.8 | 0.2785 | 0.7063 | 4.22e-02 | `lr(linearK)_bs32` |
| 64 | 60.5 | 0.3022 | 0.7082 | 4.40e-02 | `lr(linearK)_bs64` |
| 128 | 59.7 | 0.4959 | 0.7024 | 9.83e-02 | `lr(linearK)_bs128` |
| 256 | 62.4 | 0.1466 | 0.6898 | 7.49e-03 | `lr(linearK)_bs256` |
| 512 | 60.6 | 0.2976 | 0.7444 | 5.10e-02 | `lr(linearK)_bs512` |
| 1024 | 62.9 | 0.0862 | 0.6467 | 7.85e-04 | `lr(linearK)_bs1024` |
| 2048 | 64.2 | 0.0850 | 0.6461 | 8.44e-04 | `lr(linearK)_bs2048` |
| 4096 | 63.5 | 0.0887 | 0.6505 | 1.01e-03 | `lr(linearK)_bs4096` |

**SUMMARY.** Best by L2: `bs2048`, L2 0.0850, mass balance 8.44e-04, time 64.2 s. Time across all batches stays within 59.7...64.2 s (7 % spread), so it does not depend on the batch size and picking a batch by time makes no sense.

## 1.5 lr(sqrtK), 5000 Adam steps

| Batch | Adam, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|
| 32 | 19.9 | 0.5166 | 0.7215 | 1.24e-01 | `lr(sqrtK)_bs32` |
| 64 | 19.9 | 0.3528 | 0.7125 | 6.41e-02 | `lr(sqrtK)_bs64` |
| 128 | 20.3 | 0.3024 | 0.7125 | 7.13e-02 | `lr(sqrtK)_bs128` |
| 256 | 21.4 | 0.2718 | 0.9323 | 1.56e-02 | `lr(sqrtK)_bs256` |
| 512 | 20.7 | 0.4252 | 0.7030 | 8.32e-02 | `lr(sqrtK)_bs512` |
| 1024 | 20.8 | 0.1408 | 0.6915 | 1.54e-02 | `lr(sqrtK)_bs1024` |
| 2048 | 20.9 | 0.4916 | 0.7079 | 9.55e-02 | `lr(sqrtK)_bs2048` |
| 4096 | 21.4 | 0.0895 | 0.6470 | 3.99e-03 | `lr(sqrtK)_bs4096` |

**SUMMARY.** Best by L2: `bs4096`, L2 0.0895, mass balance 3.99e-03, time 21.4 s. Time across all batches stays within 19.9...21.4 s (8 % spread), so it does not depend on the batch size and picking a batch by time makes no sense.

## 1.6 lr(sqrtK), 15000 Adam steps

| Batch | Adam, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|
| 32 | 61.2 | 0.5116 | 0.7088 | 1.01e-01 | `lr(sqrtK)_bs32` |
| 64 | 60.4 | 0.2503 | 0.7096 | 3.42e-02 | `lr(sqrtK)_bs64` |
| 128 | 63.6 | 0.1240 | 0.6768 | 4.85e-03 | `lr(sqrtK)_bs128` |
| 256 | 62.2 | 0.2430 | 0.8876 | 2.22e-02 | `lr(sqrtK)_bs256` |
| 512 | 62.1 | 0.3750 | 0.7035 | 6.61e-02 | `lr(sqrtK)_bs512` |
| 1024 | 60.0 | 0.0670 | 0.7197 | 2.67e-03 | `lr(sqrtK)_bs1024` |
| 2048 | 63.3 | 0.2954 | 0.7053 | 5.17e-02 | `lr(sqrtK)_bs2048` |
| 4096 | 62.5 | 0.1045 | 0.6693 | 2.53e-03 | `lr(sqrtK)_bs4096` |

**SUMMARY.** Best by L2: `bs1024`, L2 0.0670, mass balance 2.67e-03, time 60.0 s. Time across all batches stays within 60.0...63.6 s (6 % spread), so it does not depend on the batch size and picking a batch by time makes no sense.

---

# 2. DDP, Adam + L-BFGS

## 2.1 lr(baseline), 5000 Adam steps

| Batch | Total, s | Adam, s | L-BFGS, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|---|---|
| 32 | 481.8 | 21.1 | 460.6 | 0.0671 | 0.6107 | 5.51e-03 | `lr(baseline)_bs32` |
| 64 | 376.7 | 20.5 | 356.1 | 0.0644 | 0.6040 | 4.68e-03 | `lr(baseline)_bs64` |
| 128 | 1248.8 | 20.9 | 1227.9 | 0.0605 | 0.5955 | 1.25e-02 | `lr(baseline)_bs128` |
| 256 | 644.5 | 21.2 | 623.3 | 0.0675 | 0.6126 | 5.93e-03 | `lr(baseline)_bs256` |
| 512 | 549.5 | 20.6 | 528.9 | 0.0571 | 0.5793 | 5.20e-04 | `lr(baseline)_bs512` |
| 1024 | 473.1 | 20.7 | 452.4 | 0.0840 | 0.6342 | 9.00e-03 | `lr(baseline)_bs1024` |
| 2048 | 506.2 | 19.9 | 486.3 | 0.0842 | 0.6319 | 7.21e-03 | `lr(baseline)_bs2048` |
| 4096 | 318.4 | 20.0 | 298.4 | 0.0832 | 0.6311 | 9.14e-03 | `lr(baseline)_bs4096` |

**SUMMARY.** Best by L2: `bs512`, L2 0.0571, mass balance 5.20e-04, time 549.5 s. Time across batches ranges over 318.4...1248.8 s; the fastest one, `bs4096` at 318.4 s, gives L2 0.0832, so the time saving is paid for in accuracy.

## 2.2 lr(baseline), 15000 Adam steps

| Batch | Total, s | Adam, s | L-BFGS, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|---|---|
| 32 | 567.0 | 62.6 | 504.4 | 0.1043 | 0.6484 | 1.64e-02 | `lr(baseline)_bs32` |
| 64 | 402.2 | 61.1 | 341.1 | 0.0688 | 0.6109 | 6.03e-03 | `lr(baseline)_bs64` |
| 128 | 2133.8 | 59.6 | 2074.2 | 0.0723 | 0.6172 | 2.86e-03 | `lr(baseline)_bs128` |
| 256 | 494.1 | 62.4 | 431.8 | 0.0693 | 0.6155 | 6.48e-03 | `lr(baseline)_bs256` |
| 512 | 518.2 | 61.1 | 457.1 | 0.0701 | 0.6149 | 5.76e-03 | `lr(baseline)_bs512` |
| 1024 | 511.1 | 61.0 | 450.0 | 0.0851 | 0.6338 | 1.03e-02 | `lr(baseline)_bs1024` |
| 2048 | 900.8 | 60.2 | 840.6 | 0.0903 | 0.6366 | 5.87e-03 | `lr(baseline)_bs2048` |
| 4096 | 461.8 | 61.5 | 400.3 | 0.0741 | 0.6190 | 6.62e-03 | `lr(baseline)_bs4096` |

**SUMMARY.** Best by L2: `bs64`, L2 0.0688, mass balance 6.03e-03, time 402.2 s. Time across batches ranges over 402.2...2133.8 s; the fastest one, `bs64` at 402.2 s, gives L2 0.0688, so the time saving is paid for in accuracy.

## 2.3 lr(linearK), 5000 Adam steps

| Batch | Total, s | Adam, s | L-BFGS, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|---|---|
| 32 | 449.2 | 20.5 | 428.6 | 0.0675 | 0.6103 | 4.76e-03 | `lr(linearK)_bs32` |
| 64 | 431.9 | 20.4 | 411.5 | 0.0732 | 0.6186 | 6.23e-03 | `lr(linearK)_bs64` |
| 128 | 555.3 | 21.1 | 534.2 | 0.0800 | 0.6268 | 4.55e-03 | `lr(linearK)_bs128` |
| 256 | 403.0 | 19.9 | 383.1 | 0.0687 | 0.6151 | 5.66e-03 | `lr(linearK)_bs256` |
| 512 | 1266.3 | 20.1 | 1246.3 | 0.1345 | 0.8222 | 3.37e-02 | `lr(linearK)_bs512` |
| 1024 | 531.5 | 20.4 | 511.0 | 0.0658 | 0.6067 | 4.76e-03 | `lr(linearK)_bs1024` |
| 2048 | 312.2 | 20.5 | 291.7 | 0.0721 | 0.6168 | 6.02e-03 | `lr(linearK)_bs2048` |
| 4096 | 353.3 | 20.0 | 333.3 | 0.0671 | 0.6092 | 5.16e-03 | `lr(linearK)_bs4096` |

**SUMMARY.** Best by L2: `bs1024`, L2 0.0658, mass balance 4.76e-03, time 531.5 s. Time across batches ranges over 312.2...1266.3 s; the fastest one, `bs2048` at 312.2 s, gives L2 0.0721, so the time saving is paid for in accuracy.

## 2.4 lr(linearK), 15000 Adam steps

| Batch | Total, s | Adam, s | L-BFGS, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|---|---|
| 32 | 522.6 | 63.7 | 458.9 | 0.0694 | 0.6145 | 6.06e-03 | `lr(linearK)_bs32` |
| 64 | 401.3 | 61.9 | 339.5 | 0.0830 | 0.6292 | 7.98e-03 | `lr(linearK)_bs64` |
| 128 | 473.7 | 62.8 | 410.9 | 0.0735 | 0.6188 | 2.96e-03 | `lr(linearK)_bs128` |
| 256 | 466.2 | 62.6 | 403.6 | 0.0688 | 0.6174 | 6.63e-03 | `lr(linearK)_bs256` |
| 512 | 608.4 | 61.8 | 546.6 | 0.0770 | 0.6243 | 3.36e-03 | `lr(linearK)_bs512` |
| 1024 | 199.9 | 62.6 | 137.3 | 0.0653 | 0.6065 | 4.20e-03 | `lr(linearK)_bs1024` |
| 2048 | 505.9 | 61.8 | 444.1 | 0.0730 | 0.6188 | 5.80e-03 | `lr(linearK)_bs2048` |
| 4096 | 225.0 | 63.4 | 161.6 | 0.0690 | 0.6119 | 5.15e-03 | `lr(linearK)_bs4096` |

**SUMMARY.** Best by L2: `bs1024`, L2 0.0653, mass balance 4.20e-03, time 199.9 s. Time across batches ranges over 199.9...608.4 s; the fastest one, `bs1024` at 199.9 s, gives L2 0.0653, so the time saving is paid for in accuracy.

## 2.5 lr(sqrtK), 5000 Adam steps

| Batch | Total, s | Adam, s | L-BFGS, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|---|---|
| 32 | 645.9 | 21.4 | 624.5 | 0.0656 | 0.6031 | 5.39e-03 | `lr(sqrtK)_bs32` |
| 64 | 570.4 | 20.3 | 550.1 | 0.0867 | 0.6342 | 1.06e-02 | `lr(sqrtK)_bs64` |
| 128 | 716.4 | 19.8 | 696.6 | 0.0669 | 0.6091 | 5.58e-03 | `lr(sqrtK)_bs128` |
| 256 | 1915.9 | 21.5 | 1894.4 | 0.1940 | 0.8465 | 9.49e-02 | `lr(sqrtK)_bs256` |
| 512 | 479.3 | 19.9 | 459.5 | 0.0732 | 0.6180 | 4.80e-05 | `lr(sqrtK)_bs512` |
| 1024 | 467.4 | 20.1 | 447.3 | 0.0708 | 0.6155 | 6.59e-03 | `lr(sqrtK)_bs1024` |
| 2048 | 633.9 | 21.6 | 612.3 | 0.0692 | 0.6137 | 4.83e-03 | `lr(sqrtK)_bs2048` |
| 4096 | 509.6 | 21.1 | 488.5 | 0.0704 | 0.6150 | 6.09e-03 | `lr(sqrtK)_bs4096` |

**SUMMARY.** Best by L2: `bs32`, L2 0.0656, mass balance 5.39e-03, time 645.9 s. Time across batches ranges over 467.4...1915.9 s; the fastest one, `bs1024` at 467.4 s, gives L2 0.0708, so the time saving is paid for in accuracy.

## 2.6 lr(sqrtK), 15000 Adam steps

| Batch | Total, s | Adam, s | L-BFGS, s | L2 (100time) | Linf | Mass balance | Model |
|---|---|---|---|---|---|---|---|
| 32 | 822.9 | 62.3 | 760.6 | 0.0907 | 0.6394 | 2.73e-03 | `lr(sqrtK)_bs32` |
| 64 | 454.4 | 61.8 | 392.6 | 0.0736 | 0.6203 | 7.62e-03 | `lr(sqrtK)_bs64` |
| 128 | 484.9 | 61.5 | 423.3 | 0.0672 | 0.6120 | 6.19e-03 | `lr(sqrtK)_bs128` |
| 256 | 2058.5 | 62.9 | 1995.7 | 0.2553 | 0.8228 | 1.16e-01 | `lr(sqrtK)_bs256` |
| 512 | 540.9 | 60.0 | 480.9 | 0.0644 | 0.6020 | 7.21e-03 | `lr(sqrtK)_bs512` |
| 1024 | 278.7 | 62.9 | 215.8 | 0.0723 | 0.6220 | 6.12e-03 | `lr(sqrtK)_bs1024` |
| 2048 | 530.3 | 62.4 | 467.9 | 0.0693 | 0.6143 | 6.18e-03 | `lr(sqrtK)_bs2048` |
| 4096 | 1061.3 | 62.9 | 998.4 | 0.0716 | 0.6164 | 6.42e-03 | `lr(sqrtK)_bs4096` |

**SUMMARY.** Best by L2: `bs512`, L2 0.0644, mass balance 7.21e-03, time 540.9 s. Time across batches ranges over 278.7...2058.5 s; the fastest one, `bs1024` at 278.7 s, gives L2 0.0723, so the time saving is paid for in accuracy.

---

# 3. Best models by L2

## 3.1 5000 Adam steps: the best model of each law in both modes

| Mode | Model | Adam, s | L-BFGS, s | Total, s | L2 (100time) | Linf | Mass balance |
|---|---|---|---|---|---|---|---|
| Only Adam | `lr(baseline)_bs4096` | 21.2 | — | 21.2 | 0.1830 | 0.6961 | 2.00e-02 |
| Only Adam | `lr(linearK)_bs2048` | 20.0 | — | 20.0 | 0.1065 | 0.6669 | 5.68e-03 |
| Only Adam | `lr(sqrtK)_bs4096` | 21.4 | — | 21.4 | 0.0895 | 0.6470 | 3.99e-03 |
| Adam + L-BFGS | `lr(baseline)_bs512` | 20.6 | 528.9 | 549.5 | **0.0571** | 0.5793 | 5.20e-04 |
| Adam + L-BFGS | `lr(linearK)_bs1024` | 20.4 | 511.0 | 531.5 | 0.0658 | 0.6067 | 4.76e-03 |
| Adam + L-BFGS | `lr(sqrtK)_bs32` | 21.4 | 624.5 | 645.9 | 0.0656 | 0.6031 | 5.39e-03 |

**SUMMARY.** Best model of the sweep: `lr(baseline)_bs512` in Adam + L-BFGS mode. L2 0.0571, Linf 0.5793, mass balance 5.20e-04, time 549.5 s.

## 3.2 15000 Adam steps: the best model of each law in both modes

| Mode | Model | Adam, s | L-BFGS, s | Total, s | L2 (100time) | Linf | Mass balance |
|---|---|---|---|---|---|---|---|
| Only Adam | `lr(baseline)_bs4096` | 61.6 | — | 61.6 | 0.1817 | 0.6929 | 1.73e-02 |
| Only Adam | `lr(linearK)_bs2048` | 64.2 | — | 64.2 | 0.0850 | 0.6461 | 8.44e-04 |
| Only Adam | `lr(sqrtK)_bs1024` | 60.0 | — | 60.0 | 0.0670 | 0.7197 | 2.67e-03 |
| Adam + L-BFGS | `lr(baseline)_bs64` | 61.1 | 341.1 | 402.2 | 0.0688 | 0.6109 | 6.03e-03 |
| Adam + L-BFGS | `lr(linearK)_bs1024` | 62.6 | 137.3 | 199.9 | 0.0653 | 0.6065 | 4.20e-03 |
| Adam + L-BFGS | `lr(sqrtK)_bs512` | 60.0 | 480.9 | 540.9 | **0.0644** | 0.6020 | 7.21e-03 |

**SUMMARY.** Best model of the sweep: `lr(sqrtK)_bs512` in Adam + L-BFGS mode. L2 0.0644, Linf 0.6020, mass balance 7.21e-03, time 540.9 s.
