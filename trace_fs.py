"""TRACE: TRajectory-Aligned Coupled Encoder (IEEE ICDM 2026).

Group feature selection when the response is a whole trajectory y in R^T.

    Stage 1  teacher f: y -> z^y in R^k and decoder psi: z -> y; both are then frozen.
    Stage 2  student g: (x, gates s) -> z^x, one stochastic gate per feature group;
             loss = GLS(y, psi(z^x)) + beta * ||z^x - z^y||^2 + lam * sum_g w_g P(s_g > 0).
    Support  the top-B groups by gate probability; the student is refit on that support.
    Missing  optional cross-fitted AIPW pseudo-outcome replaces y (use_aipw=True).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def aipw_pseudo_outcome(X, Y, R, n_folds=5, pi_floor=0.05, seed=0, pi_C=0.01):
    """Cross-fitted AIPW completion of a partly observed trajectory.

    The history of visit t is (x, observed y_<t, r_<t). An unobserved y_it becomes
    mu_t(H); an observed one becomes mu_t(H) + (y_it - mu_t(H)) / pi_t(H).
    pi_C is the inverse L2 strength of the propensity model; a strong penalty keeps
    1/pi stable when the history has many covariates.
    """
    n, T = Y.shape
    Y0 = np.where(R == 1, Y, 0.0)
    Yt = np.empty((n, T))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        for t in range(T):
            H = np.c_[X, Y0[:, :t], R[:, :t]]
            obs = tr[R[tr, t] == 1]
            mu = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(H[obs], Y[obs, t]).predict(H[te])
            if R[tr, t].min() == 1:
                pi = np.ones(len(te))
            else:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=pi_C, max_iter=1000))
                pi = clf.fit(H[tr], R[tr, t]).predict_proba(H[te])[:, 1]
            pi = np.maximum(pi, pi_floor)
            Yt[te, t] = mu + R[te, t] / pi * (Y0[te, t] - mu)
    return Yt


def _gls(yhat, y, Sinv):
    d = yhat - y
    return ((d @ Sinv) * d).sum(1).mean()


def _phi(z):
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


def _time_embedding(times, dim):
    freqs = torch.exp(-math.log(100.0) * torch.arange(dim // 2) / max(dim // 2, 1))
    ang = times.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class AttentionTeacher(nn.Module):
    """Tokens [y_t, r_t, gamma(t)] -> masked self-attention -> mean pool -> z^y; MLP decoder psi."""

    def __init__(self, T, k, d_model=16, n_heads=2, d_time=4):
        super().__init__()
        self.d_time = d_time
        self.tok = nn.Linear(2 + d_time, d_model)
        self.pos = nn.Parameter(torch.zeros(T, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Linear(d_model, k)
        self.dec = nn.Sequential(nn.Linear(k, d_model), nn.GELU(), nn.Linear(d_model, T))

    def encode(self, y, r, times, visible):
        u = torch.cat([y.unsqueeze(-1), r.unsqueeze(-1), _time_embedding(times, self.d_time)], -1)
        u = self.tok(u) + self.pos
        a, _ = self.attn(u, u, u, key_padding_mask=visible == 0)
        q = visible.unsqueeze(-1)
        return self.out(((u + a) * q).sum(1) / q.sum(1).clamp_min(1.0))

    def decode(self, z):
        return self.dec(z)


class LinearTeacher(nn.Module):
    """Fixed rank-k basis C (T x k): z^y = y C and psi(z) = z C^T ('pca' or 'dct')."""

    def __init__(self, C):
        super().__init__()
        self.register_buffer("C", torch.as_tensor(np.ascontiguousarray(C), dtype=torch.float32))

    def encode(self, y, r, times, visible):
        return y @ self.C

    def decode(self, z):
        return z @ self.C.T


class GroupStudent(nn.Module):
    """z^x(s) = W_fus sum_g s_g e_g(x^(g)) + b_fus, with one small MLP e_g per group."""

    def __init__(self, sizes, k, d_hidden=16):
        super().__init__()
        self.enc = nn.ModuleList(
            nn.Sequential(nn.Linear(int(d), d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))
            for d in sizes
        )
        self.fus = nn.Linear(d_hidden, k)

    def forward(self, xs, s):
        return self.fus(sum(s[g] * e(x) for g, (e, x) in enumerate(zip(self.enc, xs))))


def _dct_basis(T, k):
    t = np.arange(T)
    C = np.stack([np.cos(np.pi * (t + 0.5) * j / T) for j in range(k)], 1)
    return C / np.linalg.norm(C, axis=0)


@dataclass
class TRACE:
    """Budgeted group selector for trajectory outcomes. See README.md for every parameter."""

    groups: list
    budget: int | None = None
    k: int = 3
    teacher: str = "attention"
    beta: float = 1.0
    lam: float = 1.5
    kappa_w: float = 0.5
    sigma_gate: float = 0.5
    use_aipw: bool = False
    aipw_folds: int = 5
    pi_floor: float = 0.05
    n_restarts: int = 1
    d_model: int = 16
    n_heads: int = 2
    d_hidden: int = 16
    epochs_teacher: int = 300
    epochs_student: int = 400
    epochs_refit: int = 300
    lr: float = 5e-3
    weight_decay: float = 1e-4
    seed: int = 0
    group_prob_: np.ndarray = field(default=None, init=False, repr=False)
    support_: list = field(default=None, init=False, repr=False)

    def fit(self, X, Y, R=None, times=None):
        """X: (n, p) covariates. Y: (n, T) trajectories, NaN = not observed.
        R: (n, T) 0/1 observation mask (default: ~isnan(Y)). times: (n, T) visit times (default 1..T)."""
        torch.manual_seed(self.seed)
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        n, T = Y.shape
        R = (~np.isnan(Y)).astype(float) if R is None else np.asarray(R, float)
        self.groups = [np.asarray(g, int) for g in self.groups]
        G = len(self.groups)

        self.x_mean_, self.x_std_ = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.x_mean_) / self.x_std_
        if self.use_aipw:
            Ystar = aipw_pseudo_outcome(Xs, Y, R, self.aipw_folds, self.pi_floor, self.seed)
        else:
            Ystar = np.where(R == 1, Y, np.nanmean(np.where(R == 1, Y, np.nan), 0))
        self.y_mean_, self.y_std_ = Ystar.mean(0), Ystar.std(0) + 1e-8
        Ybar = (Ystar - self.y_mean_) / self.y_std_
        Sinv = np.linalg.inv(np.cov(Ybar.T).reshape(T, T) + 1e-2 * np.eye(T))
        Sinv *= T / np.trace(Sinv)
        times = np.tile(np.arange(1.0, T + 1), (n, 1)) if times is None else np.asarray(times, float)
        y, r, tt, Si = (torch.tensor(a, dtype=torch.float32) for a in (Ybar, R, times, Sinv))

        self.teacher_ = self._fit_teacher(y, r, tt, Si, Ybar)
        with torch.no_grad():
            zy = self.teacher_.encode(y, r, tt, torch.ones_like(r))
        self.k_ = zy.shape[1]
        xs = [torch.tensor(Xs[:, g], dtype=torch.float32) for g in self.groups]
        sizes = np.array([len(g) for g in self.groups], float)
        w = torch.tensor((sizes / np.median(sizes)) ** self.kappa_w, dtype=torch.float32)

        probs = []
        for rep in range(self.n_restarts):
            torch.manual_seed(self.seed + 7919 * rep)
            student = GroupStudent(sizes, self.k_, self.d_hidden)
            alpha = nn.Parameter(torch.full((G,), 0.5))
            self._train_student(student, xs, y, zy, Si, alpha=alpha, w=w)
            probs.append(_phi(alpha.detach() / self.sigma_gate).numpy())
        self.group_prob_ = np.mean(probs, 0)

        order = sorted(range(G), key=lambda g: (-self.group_prob_[g], g))
        keep = order[: self.budget] if self.budget else [g for g in order if self.group_prob_[g] > 0.5]
        self.support_ = sorted(keep)
        self.mask_ = torch.zeros(G)
        self.mask_[self.support_] = 1.0
        self.student_ = GroupStudent(sizes, self.k_, self.d_hidden)
        self._train_student(self.student_, xs, y, zy, Si, mask=self.mask_)
        return self

    def predict(self, X):
        Xs = (np.asarray(X, float) - self.x_mean_) / self.x_std_
        xs = [torch.tensor(Xs[:, g], dtype=torch.float32) for g in self.groups]
        with torch.no_grad():
            yb = self.teacher_.decode(self.student_(xs, self.mask_)).numpy()
        return yb * self.y_std_ + self.y_mean_

    def _fit_teacher(self, y, r, times, Sinv, Ybar):
        T = y.shape[1]
        if self.teacher in ("pca", "dct"):
            k = min(self.k, T)
            if self.teacher == "pca":
                C = np.linalg.eigh(np.cov(Ybar.T).reshape(T, T))[1][:, ::-1][:, :k]
            else:
                C = _dct_basis(T, k)
            return LinearTeacher(C)
        if self.teacher != "attention":
            raise ValueError("teacher must be 'attention', 'pca' or 'dct'")
        teacher = AttentionTeacher(T, self.k, self.d_model, self.n_heads)
        opt = torch.optim.Adam(teacher.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rows = torch.arange(len(y))
        for _ in range(self.epochs_teacher):
            visible = torch.ones_like(r)
            if T > 1:  # denoising: hide one random coordinate in half of the rows
                hide = torch.rand(len(y)) < 0.5
                visible[rows[hide], torch.randint(T, (int(hide.sum()),))] = 0.0
            loss = _gls(teacher.decode(teacher.encode(y, r, times, visible)), y, Sinv)
            opt.zero_grad()
            loss.backward()
            opt.step()
        for prm in teacher.parameters():
            prm.requires_grad_(False)
        return teacher

    def _train_student(self, student, xs, y, zy, Sinv, alpha=None, w=None, mask=None):
        groups = [{"params": student.parameters(), "weight_decay": self.weight_decay}]
        if alpha is not None:
            groups.append({"params": [alpha], "weight_decay": 0.0})
        opt = torch.optim.Adam(groups, lr=self.lr)
        epochs = self.epochs_student if alpha is not None else self.epochs_refit
        for _ in range(epochs):
            if alpha is not None:
                s = (alpha + self.sigma_gate * torch.randn_like(alpha)).clamp(0, 1)
            else:
                s = mask
            zx = student(xs, s)
            loss = _gls(self.teacher_.decode(zx), y, Sinv) + self.beta * ((zx - zy) ** 2).sum(1).mean()
            if alpha is not None:
                loss = loss + self.lam * (w * _phi(alpha / self.sigma_gate)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()


__all__ = ["TRACE", "aipw_pseudo_outcome"]
