"""Переносится ли картина с Gold58 на остальные 4349 исследований без знания Gold.

Три проверки:
 1. Спирмен между членами внутри метки: Gold58 против 4407 (одна ли структура согласия).
 2. Согласие с независимым от текста источником — OOF картиночной модели AUX60 (5 фолдов, 4345 не-Gold
    исследований): Спирмен и AUC против OOF ≥ 0,5 для каждого члена и ансамбля, по меткам.
    Оговорка: AUX60 учился на псевдометках семейства Core5, поэтому члены Core5 получают фору.
 3. Порядок источников по AUC на Gold58 против порядка по согласию с AUX60 на 4345.
Выходы: results/transfer_*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evalib2 import CORE5, CORE6, TARGETS, auc_fast, per_label_auc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

pop = pd.read_csv(ROOT / "data/train4407_core6_long.csv")
oof = pd.read_csv(ROOT / "data/aux60_oof_long.csv")
m = pd.read_csv(ROOT / "data/gold58_master.csv")

pop["core5_equal"] = pop[CORE5].mean(axis=1)
pop["core6_equal"] = pop[CORE6].mean(axis=1)
pop["core5new"] = pop[[c for c in CORE6 if c != "laymond_qwen3"]].mean(axis=1)
pop["core7_lixin"] = pop[CORE6 + ["lixin_sol56"]].mean(axis=1)
pop["opus_sol"] = pop[["opus_v1", "sol6_max_v1"]].mean(axis=1)
pop["core4_old"] = pop[CORE6[:4]].mean(axis=1)
m["core5_equal"] = m[CORE5].mean(axis=1)
m["core6_equal"] = m[CORE6].mean(axis=1)
m["core5new"] = m[[c for c in CORE6 if c != "laymond_qwen3"]].mean(axis=1)
m["core7_lixin"] = m[CORE6 + ["lixin_sol56"]].mean(axis=1)
m["opus_sol"] = m[["opus_v1", "sol6_max_v1"]].mean(axis=1)
m["core4_old"] = m[CORE6[:4]].mean(axis=1)

SOURCES = CORE6 + ["lixin_sol56", "core4_old", "opus_sol", "core5_equal", "core6_equal", "core5new", "core7_lixin"]

# --- 2. согласие с AUX60 OOF на 4345 не-Gold
ng = pop[~pop.is_gold58].merge(oof, on=["StudyInstanceUID", "label"], validate="one_to_one")
assert ng.StudyInstanceUID.nunique() == 4345
rows_sp, rows_auc = [], []
for lab in TARGETS:
    g = ng[ng.label == lab]
    yb = (g.aux60_oof >= 0.5).astype(int).to_numpy()
    for s in SOURCES:
        gg = g[[s, "aux60_oof"]].dropna()
        rows_sp.append({"label": lab, "source": s, "spearman_oof_4345": spearmanr(gg[s], gg.aux60_oof).statistic})
        rows_auc.append({"label": lab, "source": s, "auc_vs_oof05_4345": auc_fast(yb[g[s].notna()], g[s].dropna())})
sp = pd.DataFrame(rows_sp).pivot(index="source", columns="label", values="spearman_oof_4345")[TARGETS]
au = pd.DataFrame(rows_auc).pivot(index="source", columns="label", values="auc_vs_oof05_4345")[TARGETS]
sp["MEAN"] = sp.mean(axis=1)
au["MEAN"] = au.mean(axis=1)

# то же на Gold58 (AUX60 там — 5-fold-предсказания, столбец aux60), и AUC по Gold
rows = []
for lab in TARGETS:
    g = m[m.label == lab]
    for s in SOURCES:
        rows.append({"label": lab, "source": s, "spearman_aux60_gold58": spearmanr(g[s], g.aux60).statistic})
spg = pd.DataFrame(rows).pivot(index="source", columns="label", values="spearman_aux60_gold58")[TARGETS]
spg["MEAN"] = spg.mean(axis=1)
gold_auc = pd.DataFrame({s: per_label_auc(m, s) for s in SOURCES}).T[TARGETS]
gold_auc["MEAN"] = gold_auc.mean(axis=1)

# --- 3. согласованность порядков источников: Gold58 AUC против согласия с OOF на 4345
order_rows = []
singles = CORE6 + ["lixin_sol56"]
for lab in TARGETS + ["MEAN"]:
    a = gold_auc.loc[singles, lab]
    b = sp.loc[singles, lab]
    c = au.loc[singles, lab]
    order_rows.append({"label": lab, "spearman(goldAUC, oofSpearman)": spearmanr(a, b).statistic,
                       "spearman(goldAUC, aucVsOOF)": spearmanr(a, c).statistic,
                       "best_by_gold": a.idxmax(), "best_by_oof_spearman": b.idxmax(), "worst_by_gold": a.idxmin(),
                       "worst_by_oof_spearman": b.idxmin()})
order = pd.DataFrame(order_rows)

# --- сравнение ансамблей на 4345: разница Спирмена с OOF, бутстрэп по исследованиям
def boot_delta(col_a, col_b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    uids = ng.StudyInstanceUID.unique()
    wide = {c: ng.pivot(index="StudyInstanceUID", columns="label", values=c).reindex(uids)[TARGETS].to_numpy()
            for c in [col_a, col_b, "aux60_oof"]}
    deltas = np.empty(n_boot)
    n = len(uids)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d = []
        for j in range(12):
            A, B, O = wide[col_a][idx, j], wide[col_b][idx, j], wide["aux60_oof"][idx, j]
            ok = ~np.isnan(A) & ~np.isnan(B)
            d.append(spearmanr(A[ok], O[ok]).statistic - spearmanr(B[ok], O[ok]).statistic)
        deltas[b] = np.mean(d)
    return {"delta": float(sp.loc[col_a, "MEAN"] - sp.loc[col_b, "MEAN"]),
            "ci_lo": float(np.quantile(deltas, 0.025)), "ci_hi": float(np.quantile(deltas, 0.975))}

ens_cmp = []
for a, b in [("core6_equal", "core5_equal"), ("core5new", "core6_equal"), ("core7_lixin", "core6_equal"),
             ("opus_sol", "core4_old"), ("core6_equal", "opus_sol")]:
    ens_cmp.append({"a": a, "b": b, **boot_delta(a, b)})
ens_cmp = pd.DataFrame(ens_cmp)

sp.to_csv(OUT / "transfer_spearman_oof_4345.csv")
au.to_csv(OUT / "transfer_auc_vs_oof05_4345.csv")
spg.to_csv(OUT / "transfer_spearman_aux60_gold58.csv")
gold_auc.to_csv(OUT / "transfer_gold58_auc.csv")
order.to_csv(OUT / "transfer_order_consistency.csv", index=False)
ens_cmp.to_csv(OUT / "transfer_ensemble_deltas_4345.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30)
print("Spearman with AUX60 OOF, 4345 non-Gold:\n", sp.round(3).to_string())
print("\nAUC vs OOF>=0.5, 4345:\n", au.round(3).to_string())
print("\nSpearman with AUX60 on Gold58 (5-fold preds):\n", spg.round(3).to_string())
print("\nGold58 AUC:\n", gold_auc.round(3).to_string())
print("\nOrder consistency:\n", order.round(2).to_string(index=False))
print("\nEnsemble deltas (mean Spearman with OOF, 4345):\n", ens_cmp.round(4).to_string(index=False))
