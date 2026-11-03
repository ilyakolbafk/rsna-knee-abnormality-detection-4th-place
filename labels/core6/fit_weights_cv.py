"""Подбор долей членов ансамбля с честной оценкой: вложенная повторная 5-fold CV по исследованиям.

Для каждого набора членов и варианта весов:
  - in-sample: веса подобраны на всех 58 и оценены на них же («с подглядыванием», верхняя граница);
  - CV: 5-fold по исследованиям × R повторов; веса подбираются только на 4 фолдах, применяются к пятому;
    pooled OOF macro AUC на 58 — честная оценка того, что даёт процедура подбора;
  - bagged OOF: средний OOF-скор клетки по повторам → AUC и парный бутстрэп против равных весей.

Варианты весов:
  per-label free     — свой вектор на каждую метку, w ∈ симплекс (MILP, точный оптимум)
  per-label bounded  — то же, но w_k ∈ [1/(2k), 2/k] (не дальше ×2 от равных)
  shared free        — один вектор на все 12 меток (поиск по точной цели, см. search_auc.py)
  shared bounded     — то же с границами
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from evalib2 import CORE6, TARGETS, auc_fast, paired_bootstrap
from milp_auc import fit_weights_milp
from search_auc import search_weights

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 10
K = 5

m = pd.read_csv(ROOT / "data/gold58_master.csv")
m["astra_x3"] = m[["astra_r1", "astra_r2", "astra_r3"]].mean(axis=1)
m["astra_med"] = m[["astra_r1", "astra_r2", "astra_r3"]].median(axis=1)
cases = np.array(sorted(m.case_id.unique()))
lab_index = {lab: i for i, lab in enumerate(TARGETS)}

SETS = {
    "Core6": CORE6,
    "Core5new (Core6 − laymond)": [c for c in CORE6 if c != "laymond_qwen3"],
    "Core7 (Core6 + lixin)": CORE6 + ["lixin_sol56"],
    "Core7 (Core6 + Astra ×3)": CORE6 + ["astra_x3"],
}
VARIANTS = [
    ("per-label free", "milp", "free"),
    ("per-label bounded ×2", "milp", "bounded"),
    ("shared free", "search", "free"),
    ("shared bounded ×2", "search", "bounded"),
]
PER_LABEL_SETS = {"Core6", "Core5new (Core6 − laymond)"}


def bounds_for(kind: str, k: int):
    return (0.0, 1.0) if kind == "free" else (1.0 / (2 * k), 2.0 / k)


def blocks_for(df: pd.DataFrame, members: list[str], case_ids) -> dict:
    d = df[df.case_id.isin(case_ids)]
    out = {}
    for lab in TARGETS:
        g = d[d.label == lab]
        out[lab] = (g[members].to_numpy(float), g.gold.to_numpy(int))
    return out


def fit(blocks: dict, method: str, kind: str, k: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Вернуть веса по метке (для shared — один и тот же вектор во всех метках)."""
    lo, hi = bounds_for(kind, k)
    if method == "milp":
        w = {}
        for lab, (X, y) in blocks.items():
            r = fit_weights_milp([(X, y)], lo=lo, hi=hi, time_limit=45)
            w[lab] = r["w"]
        return w
    r = search_weights(list(blocks.values()), lo=lo, hi=hi, n_random=60_000, n_local=30, seed=seed)
    return {lab: r["w"] for lab in blocks}


def apply(df: pd.DataFrame, members: list[str], w: dict[str, np.ndarray], case_ids) -> pd.Series:
    d = df[df.case_id.isin(case_ids)]
    out = pd.Series(index=d.index, dtype=float)
    for lab in TARGETS:
        idx = d.index[d.label == lab]
        out.loc[idx] = d.loc[idx, members].to_numpy(float) @ w[lab]
    return out


def macro_of(df: pd.DataFrame, s: pd.Series, mask=None) -> tuple[float, dict]:
    d = df if mask is None else df[mask]
    per = {lab: auc_fast(d.loc[d.label == lab, "gold"], s.loc[d.index[d.label == lab]]) for lab in TARGETS}
    assert not any(np.isnan(v) for v in per.values()), "NaN AUC: пустой класс или NaN в скорах"
    return float(np.mean(list(per.values()))), per


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


results = []
fold_weights = []
bagged = m[["case_id", "uid", "label", "split", "gold"]].copy()
t0 = time.time()
for set_name, members in SETS.items():
    k = len(members)
    m["equal"] = m[members].mean(axis=1)
    eq_macro, eq_per = macro_of(m, m["equal"])
    eq_hold, _ = macro_of(m, m["equal"], m.split == "hold18")
    for var_name, method, kind in VARIANTS:
        if method == "milp" and set_name not in PER_LABEL_SETS:
            continue
        tag = f"{set_name} | {var_name}"
        log(f"START {tag}")
        # in-sample
        w_all = fit(blocks_for(m, members, cases), method, kind, k)
        s_in = apply(m, members, w_all, cases)
        in_macro, in_per = macro_of(m, s_in)
        # CV
        oof_macros, oof_holds, oof_per_list = [], [], []
        oof_scores = np.zeros((R, len(m)))
        for r in range(R):
            kf = KFold(K, shuffle=True, random_state=1000 + r)
            s_oof = pd.Series(index=m.index, dtype=float)
            for f, (tr, te) in enumerate(kf.split(cases)):
                w_f = fit(blocks_for(m, members, cases[tr]), method, kind, k, seed=r * 10 + f)
                s_oof.loc[m.case_id.isin(cases[te])] = apply(m, members, w_f, cases[te])
                for lab in (TARGETS if method == "milp" else [TARGETS[0]]):
                    fold_weights.append({"set": set_name, "variant": var_name, "repeat": r, "fold": f,
                                         "label": lab if method == "milp" else "shared",
                                         **{mm: float(x) for mm, x in zip(members, w_f[lab])}})
            oof_scores[r] = s_oof.to_numpy()
            mac, per = macro_of(m, s_oof)
            hold, _ = macro_of(m, s_oof, m.split == "hold18")
            oof_macros.append(mac)
            oof_holds.append(hold)
            oof_per_list.append(per)
            log(f"  {tag} repeat {r}: OOF macro {mac:.4f} (hold18 {hold:.4f}); equal {eq_macro:.4f}")
        col = f"oof_bagged::{tag}"
        m[col] = oof_scores.mean(axis=0)
        bagged[col] = m[col]
        bag_macro, bag_per = macro_of(m, m[col])
        bag_hold, _ = macro_of(m, m[col], m.split == "hold18")
        boot = paired_bootstrap(m, col, "equal", n_boot=2000, seed=7)
        results.append({
            "set": set_name, "variant": var_name, "k": k, "members": members,
            "equal_macro58": eq_macro, "equal_hold18": eq_hold,
            "insample_macro58": in_macro, "insample_per_label": in_per,
            "insample_weights": {lab: [float(x) for x in w_all[lab]] for lab in TARGETS} if method == "milp"
            else [float(x) for x in w_all[TARGETS[0]]],
            "oof_macro58_mean": float(np.mean(oof_macros)), "oof_macro58_sd": float(np.std(oof_macros, ddof=1)),
            "oof_macro58_min": float(np.min(oof_macros)), "oof_macro58_max": float(np.max(oof_macros)),
            "oof_hold18_mean": float(np.mean(oof_holds)),
            "oof_per_label_mean": {lab: float(np.mean([p[lab] for p in oof_per_list])) for lab in TARGETS},
            "bagged_oof_macro58": bag_macro, "bagged_oof_hold18": bag_hold,
            "bagged_vs_equal": boot, "repeats": R, "folds": K,
        })
        log(f"DONE {tag}: in-sample {in_macro:.4f} | OOF {np.mean(oof_macros):.4f} ± {np.std(oof_macros, ddof=1):.4f} "
            f"| bagged {bag_macro:.4f} (Δ vs equal {boot['delta']:+.4f} [{boot['ci_lo']:+.4f}; {boot['ci_hi']:+.4f}]) "
            f"| equal {eq_macro:.4f} | {time.time() - t0:.0f}s")
        json.dump(results, open(OUT / "weights_cv.json", "w"), ensure_ascii=False, indent=1)
        pd.DataFrame(fold_weights).to_csv(OUT / "fold_weights.csv", index=False)
        bagged.to_csv(OUT / "oof_bagged_scores.csv", index=False)

summary = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in results])
summary["bagged_delta"] = [r["bagged_vs_equal"]["delta"] for r in results]
summary["bagged_ci_lo"] = [r["bagged_vs_equal"]["ci_lo"] for r in results]
summary["bagged_ci_hi"] = [r["bagged_vs_equal"]["ci_hi"] for r in results]
summary.to_csv(OUT / "weights_cv_summary.csv", index=False)
log("ALL DONE")
print(summary.round(4).to_string(index=False))
