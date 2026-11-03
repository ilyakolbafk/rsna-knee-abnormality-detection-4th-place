"""Общие функции замеров для Core6: AUC по меткам, macro, бутстрэп по исследованиям, ранги."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CORE6 = ["laymond_qwen3", "yunus_gemini", "p3_xhigh", "qwen_family", "opus_v1", "sol6_max_v1"]
CORE5 = ["laymond_qwen3", "lixin_sol56", "yunus_gemini", "p3_xhigh", "qwen_family"]


def auc_fast(y: np.ndarray, s: np.ndarray) -> float:
    """AUC с ничьими = 0,5 (ранговая формула, совпадает с sklearn)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_label_auc(df: pd.DataFrame, col: str, mask=None) -> pd.Series:
    d = df if mask is None else df[mask]
    out = {}
    for lab, g in d.groupby("label", observed=True):
        out[lab] = auc_fast(g.gold.to_numpy(), g[col].to_numpy())
    return pd.Series(out).reindex(TARGETS)


def macro(df: pd.DataFrame, col: str, mask=None) -> float:
    return float(per_label_auc(df, col, mask).mean())


def summary(df: pd.DataFrame, col: str) -> dict:
    p = df[col].to_numpy()
    y = df.gold.to_numpy()
    pred = (p >= 0.5).astype(int)
    return {
        "macro58": macro(df, col),
        "macro_dev40": macro(df, col, df.split == "dev40"),
        "macro_hold18": macro(df, col, df.split == "hold18"),
        "correct": int((pred == y).sum()),
        "FN": int(((pred == 0) & (y == 1)).sum()),
        "FP": int(((pred == 1) & (y == 0)).sum()),
        "correct_hold18": int((pred[df.split == "hold18"] == y[df.split == "hold18"]).sum()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def paired_bootstrap(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 2000, seed: int = 0,
                     mask=None) -> dict:
    """Разница macro AUC (a − b), парный бутстрэп по исследованиям (case_id)."""
    d = df if mask is None else df[mask]
    rng = np.random.default_rng(seed)
    cases = d.case_id.unique()
    # предварительно: матрицы по метке × исследование
    piv_y = d.pivot(index="case_id", columns="label", values="gold").reindex(cases)[TARGETS].to_numpy()
    piv_a = d.pivot(index="case_id", columns="label", values=col_a).reindex(cases)[TARGETS].to_numpy()
    piv_b = d.pivot(index="case_id", columns="label", values=col_b).reindex(cases)[TARGETS].to_numpy()
    n = len(cases)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ya, aa, bb = piv_y[idx], piv_a[idx], piv_b[idx]
        da = []
        for j in range(12):
            yj = ya[:, j]
            if yj.sum() == 0 or yj.sum() == len(yj):
                continue
            da.append(auc_fast(yj, aa[:, j]) - auc_fast(yj, bb[:, j]))
        deltas[b] = np.mean(da)
    point = macro(d, col_a) - macro(d, col_b)
    return {"delta": point, "ci_lo": float(np.quantile(deltas, 0.025)),
            "ci_hi": float(np.quantile(deltas, 0.975)), "p_gt0": float((deltas > 0).mean())}


def within_label_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ранг-перцентиль внутри метки по самой таблице (0..1)."""
    out = df.copy()
    for c in cols:
        out[c] = df.groupby("label", observed=True)[c].rank(method="average", pct=True)
    return out


def population_percentile(gold: pd.DataFrame, pop: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Перцентиль значения Gold58-клетки внутри распределения на 4407 (по метке и члену). Разворачиваемо на train."""
    out = gold.copy()
    for c in cols:
        for lab in TARGETS:
            ref = np.sort(pop.loc[pop.label == lab, c].dropna().to_numpy())
            m = out.label == lab
            v = out.loc[m, c].to_numpy()
            lo = np.searchsorted(ref, v, side="left")
            hi = np.searchsorted(ref, v, side="right")
            out.loc[m, c] = (lo + hi) / 2 / len(ref)
    return out


def fmt(x: float, nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}".replace(".", ",")
