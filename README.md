# GDLC

1.This code corresponds to the work:  "A Generalized Deep Learning Clustering Algorithm Based on Non-Negative Matrix Factorization " .

2.The main function of this code is run_GDLC.m.

---

## Reproduction Study — Group 24

This fork is a **reproduction study**, carried out for a college research-paper
reproduction assignment, of:

> D. Wang, T. Li, P. Deng, F. Zhang, W. Huang, P. Zhang, J. Liu.
> *"A Generalized Deep Learning Clustering Algorithm Based on Non-Negative
> Matrix Factorization."* ACM Transactions on Knowledge Discovery from Data, 2023.

**The GDLC algorithm and this MATLAB implementation are the work of the paper's
authors.** The original implementation is at
<https://github.com/Code706/GDLC> (Code706), and all credit for the method
belongs there. Our group did **not** invent GDLC. Everything below describes
only the reproduction tooling and the small experimental extensions we added on
top of the released code.

### The mathematical baseline is preserved

`GDLC.m` was **parameterised, not redesigned**. In its default configuration —
which is what you get when you call it the original way,
`GDLC(fea', nClass, gnd)` — every update equation, the initialisation, the
`rand('twister',5489)` seeding, the objective-function computation and the
per-round console output are byte-for-byte the released ones. Only hardcoded
constants (`C`, `eta`, `alpha1/2`, `beta1/2`, the round count) were replaced by
fields of an optional `params` struct that defaults to exactly those constants.

`run_GDLC.m` is **untouched** and remains the authors' entry point.

### Engineering / reproducibility changes we made

| Change | Why |
|---|---|
| `logsig.m` compatibility shim | MATLAB's built-in `logsig` ships with the Deep Learning Toolbox, which is not installed on our machines. The shim is `1./(1+exp(-x))` — mathematically identical, so this is an environment fix, not an algorithm change. |
| Optional `params` argument on `GDLC.m` | Lets dataset-specific published hyper-parameters be supplied without editing the algorithm. Old 3-argument calls still work unchanged. |
| Fifth output `extra` | Exposes the learned `Q`/`W`, final labels, per-round ACC/NMI/objective, and the learning-rate schedule for plotting. The first four outputs are unchanged. |
| `run_reproduction.m` | Automated multi-dataset execution (BASEHOCK, PCMAC, SMK_CAN_187) with the authors' preprocessing pipeline. |
| Runtime measurement | `tic`/`toc` around each dataset run, so runtimes are comparable to the paper's Table. |
| Per-round logging | ACC / NMI / objective are recorded for every one of the 10 GDLC rounds. |
| CSV export | `results/*.csv` — machine-readable, nothing hardcoded. |
| Automatic plots | `figures/*.png` — presentation-ready comparison and convergence figures. |
| Original-vs-reproduced comparison | `results/reproduction_summary.csv` and `results/reproduction_report.txt` put the paper's published numbers next to ours with absolute and percentage differences. |
| `validate_setup.m` | Pre-flight check: files, datasets, toolbox dependencies, syntax, metric unit tests, and backward-compatibility/fixed-mode-equivalence assertions. |

### Datasets

All three `.mat` files store the feature matrix in `X` (samples × features) and
the labels in `Y` (samples × 1, classes `{1,2}`):

| Dataset | Samples | Features | Classes |
|---|---|---|---|
| BASEHOCK | 1993 | 4862 | 2 |
| PCMAC | 1943 | 3289 | 2 |
| SMK_CAN_187 | 187 | 19993 | 2 |

### Published hyper-parameters used for reproduction

| Dataset | C | eta | alpha1 | beta1 | alpha2 | beta2 | rounds |
|---|---|---|---|---|---|---|---|
| BASEHOCK | 2 | 3.5e-3 | 1e-1 | 1e-1 | 1e-1 | 1e-1 | 10 |
| PCMAC | 2 | 5e-3 | 5e-2 | 5e-2 | 5e-2 | 5e-2 | 10 |
| SMK_CAN_187 | 2 | 2e-1 | 5e-1 | 5e-1 | 5e-1 | 5e-1 | 10 |

---

## PAPER RESULTS (reference values only)

These are transcribed from the publication. They are **reference values for
comparison** and are never used as reproduced outputs.

| Dataset | Paper ACC | Paper NMI | Paper Runtime (s) |
|---|---|---|---|
| BASEHOCK | 0.991 | 0.935 | 23.80 |
| PCMAC | 1.000 | 1.000 | 15.60 |
| SMK_CAN_187 | 0.995 | 0.957 | 9.20 |

## OUR REPRODUCED RESULTS

**Not yet filled in.** Reproduced ACC / NMI / ARI / Purity / runtime values are
written by `run_reproduction.m` at execution time into:

- `results/reproduction_summary.csv`
- `results/reproduction_report.txt`
- `results/<DATASET>_rounds.csv`

No reproduced number in this repository is hand-entered. Run the pipeline (see
*How to run* below) and read the numbers from those files.

---

## Experimental Extensions

These are **our** additions, clearly separated from the paper's contribution.

### 1. Adaptive (decaying) learning-rate GDLC

The original algorithm uses a single fixed SGD step size `eta` for all rounds.
We added an optional schedule:

```
eta_t = eta_0 / (1 + decay * t)          % t = GDLC round index, decay = 0.1
```

**Purpose.** The paper reports that GDLC is sensitive to `eta`; we test whether
shrinking the step size in later rounds changes convergence or final clustering
quality. **Only the scalar step size changes** — the gradient expressions for
`w`, `q`, `m`, `n` are untouched. With `lrMode = 'fixed'` (the default) the
schedule is constant and the behaviour is identical to the original code.

We make **no claim** that the decaying schedule improves performance. Read
`results/extensions/BASEHOCK_extension_results.csv` and decide from the actual
numbers.

### 2. Bias ablation

GDLC's reconstruction is `X ≈ σ(W)σ(Q)ᵀ + σ(m)1ᵀ + 1σ(n)ᵀ`, i.e. it carries
per-feature (`m`) and per-sample (`n`) generalized bias vectors. Setting
`params.useBias = false` gives the mathematically consistent no-bias variant:

- the additive `σ(m_i) + σ(n_j)` term is removed from the residual `temp`;
- the `m` and `n` SGD updates are skipped entirely (they carry no gradient);
- the generalized-bias transforms Eq. (25) and Eq. (26) are skipped;
- the bias column is dropped from the objective's factorisation
  (`Ux = W`, `Vx = Q` instead of `[W, m]`, `[Q, n]`), and the
  `alpha1‖m‖² + alpha2‖n‖²` regularisation term becomes 0.

The `W` / `Q` element-update logic is otherwise unchanged. `useBias = true` is
the default and is exactly the original behaviour.

`m` and `n` are still *drawn* from the RNG in no-bias mode so that the random
initialisation of `w` and `q` is identical across both variants — only their
use is disabled.

### 3. Extra evaluation metrics

The paper reports ACC and NMI, which remain our **primary reproduction
metrics** and use the released `bestMap` / `MutualInfo` code unmodified. We
additionally report:

- **Adjusted Rand Index** — `metrics/adjusted_rand_index.m`
- **Clustering Purity** — `metrics/clustering_purity.m`

Both are dependency-free (no Statistics Toolbox), handle non-contiguous and
arbitrary numeric cluster ids, and are unit-tested by `metrics/test_metrics.m`
(identical partitions give ARI ≈ 1 and Purity = 1).

### 4. Optional latent-dimension sensitivity

`run_c_sensitivity.m` sweeps `C ∈ {1, 2, 4, 8}` on BASEHOCK with everything
else at the published settings, recording ACC / NMI / ARI / Purity / runtime.

### 5. Optional PCA visualisation

`plot_Q_pca.m` projects the learned low-dimensional representation `Q` onto its
first two principal components (computed via SVD, no toolbox required). Ground
truth is used **only** for scatter colouring, never for training.

---

## How to run

From the repository root in MATLAB:

```matlab
% 0. Pre-flight check (recommended, fast)
clear; clc; close all;
validate_setup

% 1. Original authors' baseline — unchanged behaviour
clear; clc; close all;
run_GDLC

% 2. Full reproduction across all three datasets
clear; clc; close all;
run_reproduction

% 3. Our extension experiments (BASEHOCK)
clear; clc; close all;
run_extension_experiments

% 4. Optional: latent-dimension sensitivity
clear; clc; close all;
run_c_sensitivity

% 5. Optional: PCA of the learned representation (needs step 2 first)
clear; clc; close all;
plot_Q_pca
```

Inspecting the results:

```matlab
summary = readtable('results/reproduction_summary.csv');
disp(summary);

extensions = readtable('results/extensions/BASEHOCK_extension_results.csv');
disp(extensions);

csens = readtable('results/extensions/BASEHOCK_C_sensitivity.csv');
disp(csens);
```

## Output files

**Results**

```
results/BASEHOCK_rounds.csv
results/PCMAC_rounds.csv
results/SMK_CAN_187_rounds.csv
results/reproduction_summary.csv
results/reproduction_report.txt
results/BASEHOCK_repro_extra.mat
results/extensions/BASEHOCK_extension_results.csv
results/extensions/BASEHOCK_Original_FixedLR_rounds.csv
results/extensions/BASEHOCK_AdaptiveLR_rounds.csv
results/extensions/BASEHOCK_NoBias_rounds.csv
results/extensions/BASEHOCK_C_sensitivity.csv          (optional)
```

**Figures**

```
figures/acc_comparison.png
figures/nmi_comparison.png
figures/runtime_comparison.png
figures/BASEHOCK_convergence.png
figures/PCMAC_convergence.png
figures/SMK_CAN_187_convergence.png
figures/BASEHOCK_objective.png
figures/PCMAC_objective.png
figures/SMK_CAN_187_objective.png
figures/extensions/BASEHOCK_method_comparison.png
figures/extensions/BASEHOCK_runtime_comparison.png
figures/extensions/learning_rate_schedule.png
figures/extensions/BASEHOCK_ACC_convergence_compare.png
figures/extensions/BASEHOCK_NMI_convergence_compare.png
figures/extensions/BASEHOCK_C_sensitivity.png          (optional)
figures/extensions/BASEHOCK_Q_PCA.png                  (optional)
```

## Toolbox dependencies

| Function | Origin | Status |
|---|---|---|
| `logsig` | Deep Learning Toolbox | Local shim `logsig.m` provided — no toolbox needed. |
| `randsample` | Statistics and ML Toolbox | Required by the released `litekmeans.m`. Confirmed working in our baseline run, so it was **not** replaced. |
| `im2double` | Image Processing Toolbox | Used by the authors' preprocessing; kept as-is. |
| `exportgraphics` | MATLAB R2020a+ | Used for figure export, with a `print -dpng` fallback. |

`validate_setup.m` reports on each of these.

## Attribution

Original algorithm, paper and MATLAB implementation:
Dexian Wang, Tianrui Li, Ping Deng, Fan Zhang, Wei Huang, Pengfei Zhang, Jia Liu —
<https://github.com/Code706/GDLC>.
Supporting utilities `NormalizeFea.m`, `litekmeans.m`, `bestMap.m`,
`MutualInfo.m`, `hungarian.m` are by Deng Cai et al. and are unmodified.
