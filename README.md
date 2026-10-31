# TRACE: Feature Selection for Trajectory Inference

Implementation of TRACE from the IEEE ICDM 2026 paper

> Jing Wang, Jie Shen, Sayantan Kumar, Misaki Matsuura, Jeremy C. Weiss.
> **Feature Selection for Trajectory Inference.** IEEE International Conference on Data Mining (ICDM), 2026.

TRACE (TRajectory-Aligned Coupled Encoder) selects the few **groups** of features that predict a whole
**trajectory** `y = (y_1, ..., y_T)`, such as a patient's recovery curve, rather than a single number.

```
 training only:  y ──teacher f──► z^y ──decoder ψ──► ŷ        Stage 1: fit f and ψ, then freeze both
                                  ▲ align
 x = [x^(1) … x^(G)] ──gates s_g──► student g ──► z^x ──ψ──► ŷ  Stage 2: one gate per group, sparsity penalty
                                                               Support: keep the top-B groups, refit
```

- **Teacher** `f`: encodes the observed trajectory into a short code `z^y` (attention, PCA or DCT).
- **Student** `g`: a small encoder per feature group. Each group is multiplied by one stochastic gate, then the groups are fused into `z^x`.
- **Shared decoder** `ψ`: turns a code back into a trajectory. It is shared by teacher and student and frozen in Stage 2.
- **Loss** (Stage 2): GLS trajectory error `(y-ŷ)ᵀ Σ⁻¹ (y-ŷ)` + `β‖z^x − z^y‖²` + `λ Σ_g w_g P(gate_g > 0)`.
- **Missing visits**: optionally replace `y` by a cross-fitted AIPW pseudo-outcome (`use_aipw=True`).

## Contents

| file | content |
|---|---|
| `trace_fs.py` | the method: `TRACE` and `aipw_pseudo_outcome` (Algorithm 1 of the paper) |
| `TRACE_demo.ipynb` | usage demo with outputs: load a data file → input format → fit → output format → save / reload |
| `data/example.csv` | small synthetic dataset used by the demo (random numbers, known true groups) |

This repository contains the method and a usage demo; the scripts for the paper's experiments are not
included. The Long COVID cohort analysed in the paper is controlled-access and is not distributed.

## Install

```bash
pip install -r requirements.txt        # numpy, pandas, scikit-learn, torch, joblib, jupyter
```

Everything runs on a CPU.

## Quick start

```python
import numpy as np
from trace_fs import TRACE

rng = np.random.default_rng(0)
n, G, d, T = 400, 6, 3, 4                       # 400 subjects, 6 groups of 3 features, 4 visits
X = rng.normal(size=(n, G * d))
groups = [list(range(g * d, (g + 1) * d)) for g in range(G)]   # group g = columns 3g..3g+2

t = np.arange(T) / (T - 1)
Y = (np.outer(X[:, 0] + X[:, 1], 1 - t)         # group 0 drives the early visits (linear)
     + np.outer(X[:, 3] ** 2, t)                # group 1 drives the late visits (quadratic)
     + 0.5 * rng.normal(size=(n, T)))           # groups 2-5 are pure noise
Y[rng.random((n, T)) < 0.1] = np.nan            # 10% of visits are missing (NaN)

model = TRACE(groups, budget=2, use_aipw=True).fit(X[:300], Y[:300])
print(model.group_prob_.round(2))               # P(gate > 0) for each group
print(model.support_)                           # selected groups
Y_hat = model.predict(X[300:])                  # (100, 4) predicted trajectories
```

Output (the gate probabilities can differ in the second decimal with the number of CPU threads):

```
[0.99 1.   0.03 0.03 0.03 0.03]
[0, 1]
```

`TRACE_demo.ipynb` runs the same example from a CSV file (`data/example.csv`) and also shows how to save
and reload a fitted model.

## Input format

`TRACE(groups, budget=None, **parameters).fit(X, Y, R=None, times=None)`

| input | shape | meaning |
|---|---|---|
| `X` | `(n, p)` float | baseline covariates, one row per subject |
| `Y` | `(n, T)` float | trajectory, one column per visit; `NaN` = visit not observed |
| `groups` | list of `G` lists | column indices of `X` that form each feature group; every group gets one gate |
| `R` *(optional)* | `(n, T)` 0/1 | observation mask (default `~isnan(Y)`) |
| `times` *(optional)* | `(n, T)` float | visit times, used by the attention teacher (default `1..T`) |

Covariates are standardized inside `fit`, so `X` can be on its raw scale. Covariates must be complete;
if you append missingness indicators, put each indicator in the same group as its variable.

## Output format

| attribute / method | shape | meaning |
|---|---|---|
| `group_prob_` | `(G,)` | gate probability `P(s_g > 0) = Φ(α_g / σ_s)` of each group |
| `support_` | list | selected groups (indices into `groups`): the top `budget` by gate probability, or every group above 0.5 if `budget=None` |
| `predict(X_new)` | `(m, T)` | predicted trajectories on the original scale of `Y`; uses only the selected groups |

A fitted model can be saved with `joblib.dump(model, path)` and reloaded with `joblib.load(path)`.

## Parameters

| parameter | symbol in paper | default | meaning |
|---|---|---|---|
| `groups` | `x^(1..G)` | required | list of column-index lists, one per feature group |
| `budget` | `B` | `None` | number of groups to keep; `None` keeps every group with gate probability > 0.5 |
| `k` | `k` | `3` | width of the shared code `z` used by teacher and student |
| `teacher` | `f_φ` | `"attention"` | `"attention"`: learned self-attention encoder with an MLP decoder; `"pca"` / `"dct"`: fixed rank-`k` basis with its transpose as decoder (low variance for short `T`) |
| `beta` | `β` | `1.0` | weight of the teacher–student alignment `‖z^x − z^y‖²` |
| `lam` | `λ` | `1.5` | weight of the group-sparsity penalty (expected number of active groups); larger → fewer groups |
| `kappa_w` | `κ_w` | `0.5` | group-size cost `w_g = (d_g / median d)^κ_w`; `0` uniform, `0.5` square-root size, `1` linear size. Raise it when group sizes are very unequal |
| `sigma_gate` | `σ_s` | `0.5` | noise level of the stochastic gates |
| `use_aipw` | — | `False` | complete missing visits with the cross-fitted AIPW pseudo-outcome; if `False`, a missing visit is filled with that visit's mean |
| `aipw_folds` | `K` | `5` | cross-fitting folds for the AIPW nuisance models |
| `pi_floor` | `η` | `0.05` | lower clip on the estimated probability that a visit is observed |
| `n_restarts` | `R` | `1` | average gate probabilities over `R` restarts before top-B (restart-averaged selection) |
| `d_model`, `n_heads` | `d_m` | `16`, `2` | width and heads of the attention teacher |
| `d_hidden` | `d_h` | `16` | width of each group encoder `e_g` |
| `epochs_teacher`, `epochs_student`, `epochs_refit` | `E_1`, `E_2` | `300`, `400`, `300` | full-batch Adam steps for Stage 1, Stage 2 and the fixed-support refit |
| `lr` | — | `5e-3` | Adam learning rate |
| `weight_decay` | `ν` | `1e-4` | weight decay (capacity control) |
| `seed` | — | `0` | random seed |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wang2026trace,
  title     = {Feature Selection for Trajectory Inference},
  author    = {Wang, Jing and Shen, Jie and Kumar, Sayantan and Matsuura, Misaki and Weiss, Jeremy C.},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```
