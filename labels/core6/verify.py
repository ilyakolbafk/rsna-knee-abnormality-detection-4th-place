"""Независимые проверки: AUC против sklearn, числа Codex, точность MILP, устойчивость вывода CV к сидy/схеме.

Выход: results/verify.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

from evalib2 import CORE6, TARGETS, auc_fast
from milp_auc import fit_weights_milp
from search_auc import search_weights

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
m = pd.read_csv(ROOT / "data/gold58_master.csv")
m["core6_equal"] = m[CORE6].mean(axis=1)
checks = {}

# 1. AUC функция против sklearn на всех столбцах
maxdiff = 0.0
for c in CORE6 + ["lixin_sol56", "core6_equal", "core5", "v61", "aux60"]:
    for lab in TARGETS:
        g = m[m.label == lab]
        maxdiff = max(maxdiff, abs(auc_fast(g.gold, g[c]) - roc_auc_score(g.gold, g[c])))
checks["auc_vs_sklearn_maxabs"] = maxdiff

# 2. числа Codex (core6-sol-max-full-report-20260925/REPORT.md): macro 0.9194414320, Brier 0.1023398487
macro6 = np.mean([roc_auc_score(m[m.label == l].gold, m[m.label == l].core6_equal) for l in TARGETS])
brier6 = float(np.mean((m.core6_equal - m.gold) ** 2))
checks["core6_equal_macro_vs_codex"] = {"ours": float(macro6), "codex": 0.9194414320, "ok": abs(macro6 - 0.9194414320) < 1e-9}
checks["core6_equal_brier_vs_codex"] = {"ours": brier6, "codex": 0.1023398487, "ok": abs(brier6 - 0.1023398487) < 1e-9}

# 3. MILP: in-sample оптимум по метке не хуже равных весов и не хуже поиска
rows = []
for lab in TARGETS:
    g = m[m.label == lab]
    X, y = g[CORE6].to_numpy(float), g.gold.to_numpy(int)
    r = fit_weights_milp([(X, y)], time_limit=60)
    s = search_weights([(X, y)], n_random=100_000, seed=1)
    rows.append({"label": lab, "auc_equal": auc_fast(y, X.mean(1)), "auc_milp": auc_fast(y, X @ r["w"]),
                 "auc_search": auc_fast(y, X @ s["w"]), "z_milp": r["z_sum"], "z_search": s["z_sum"],
                 "milp_status": r["status"][:40], "n_conflict": r["n_conflict"], "n_pairs": r["pairs"]})
milp_tab = pd.DataFrame(rows)
checks["milp_insample"] = milp_tab.to_dict("records")
checks["milp_ge_equal_all_labels"] = bool((milp_tab.auc_milp >= milp_tab.auc_equal - 1e-12).all())
checks["milp_vs_search_max_abs_auc"] = float((milp_tab.auc_milp - milp_tab.auc_search).abs().max())
checks["conflict_pair_share"] = float(milp_tab.n_conflict.sum() / milp_tab.n_pairs.sum())

# 4. другая схема CV: leave-one-study-out, per-label bounded ×2 (поиск, дёшево) и shared bounded ×2
cases = np.array(sorted(m.case_id.unique()))
lo, hi = 1 / 12, 2 / 6


def loo(variant: str):
    s_oof = pd.Series(index=m.index, dtype=float)
    for tr, te in LeaveOneOut().split(cases):
        d = m[m.case_id.isin(cases[tr])]
        blocks = [(d[d.label == l][CORE6].to_numpy(float), d[d.label == l].gold.to_numpy(int)) for l in TARGETS]
        if variant == "shared":
            w = search_weights(blocks, lo=lo, hi=hi, n_random=20_000, n_local=15, seed=int(te[0]))["w"]
            ws = {l: w for l in TARGETS}
        else:
            ws = {l: search_weights([b], lo=lo, hi=hi, n_random=20_000, n_local=15, seed=int(te[0]))["w"]
                  for l, b in zip(TARGETS, blocks)}
        idx = m.index[m.case_id == cases[te[0]]]
        for l in TARGETS:
            ii = idx[m.loc[idx, "label"] == l]
            s_oof.loc[ii] = m.loc[ii, CORE6].to_numpy(float) @ ws[l]
    return float(np.mean([auc_fast(m[m.label == l].gold, s_oof[m.label == l]) for l in TARGETS]))


if "--loo" in sys.argv:
    checks["loo_per_label_bounded_x2_macro"] = loo("per-label")
    checks["loo_shared_bounded_x2_macro"] = loo("shared")
    checks["equal_macro"] = float(macro6)

json.dump(checks, open(OUT / "verify.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in checks.items() if k != "milp_insample"}, ensure_ascii=False, indent=1, default=float))
print(milp_tab.round(4).to_string(index=False))
