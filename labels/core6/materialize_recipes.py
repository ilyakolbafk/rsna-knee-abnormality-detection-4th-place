"""Материализовать зафиксированные рецепты на 4407 исследованиях и записать контракт с хэшами.

Рецепты (mean-p, равные доли):
  core6_equal   — laymond_qwen3, yunus_gemini, p3_xhigh, qwen_family, opus_v1, sol6_max_v1 (= Codex core6_equal_train4407.csv)
  core5new_equal — то же без laymond_qwen3
На двух не-Gold исследованиях у laymond нет скоров: core6_equal там = среднее пяти (reader_count=5, как в контракте Codex);
для core5new это ничего не меняет.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evalib2 import CORE6, TARGETS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
long = pd.read_csv(ROOT / "data/train4407_core6_long.csv", dtype={"StudyInstanceUID": str})
codex = pd.read_csv(ROOT / "inputs/.codex/runs/core6-sol-max-full-report-20260925/core6_equal_train4407.csv",
                    dtype={"StudyInstanceUID": str})
CORE5NEW = [c for c in CORE6 if c != "laymond_qwen3"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wide(col: str) -> pd.DataFrame:
    w = long.pivot(index="StudyInstanceUID", columns="label", values=col).reset_index()
    w = codex[["StudyInstanceUID"]].merge(w, on="StudyInstanceUID", validate="one_to_one")
    return w[["StudyInstanceUID", *TARGETS]]


long["core5new_equal"] = long[CORE5NEW].mean(axis=1)
assert long.core5new_equal.notna().all()
c6 = wide("core6_equal")
c5n = wide("core5new_equal")
assert np.allclose(c6[TARGETS].to_numpy(), codex[TARGETS].to_numpy(), atol=1e-9)
c6.to_csv(OUT / "core6_equal_train4407.csv", index=False)
c5n.to_csv(OUT / "core5new_equal_train4407.csv", index=False)

contract = {
    "version": "1.0",
    "date": "2026-09-26",
    "grain": "StudyInstanceUID × 12 targets, probabilities in [0,1]",
    "basis": "arithmetic mean of member probabilities (mean-p); rank/percentile averaging measured and rejected",
    "recipes": {
        "core6_equal": {"members": CORE6, "weights": [1 / 6] * 6,
                        "missing_policy": "24 cells on 2 non-Gold UIDs without laymond: mean of the other five",
                        "gold58_macro_auc": 0.9194414320, "file": "core6_equal_train4407.csv",
                        "sha256": sha(OUT / "core6_equal_train4407.csv"), "identical_to_codex_file": True},
        "core5new_equal": {"members": CORE5NEW, "weights": [1 / 5] * 5, "missing_policy": "none needed",
                           "gold58_macro_auc": None, "file": "core5new_equal_train4407.csv",
                           "sha256": sha(OUT / "core5new_equal_train4407.csv")},
    },
    "weight_fitting_verdict": "Any weights fitted on Gold58 (per-label or shared, MILP-exact or search, bounded or free) "
                              "have lower out-of-fold macro AUC than equal weights; proxy weights fitted on 4345 non-Gold "
                              "studies against AUX60 OOF are also not better on Gold58. Equal weights are frozen.",
    "inputs_sha256": json.load(open(ROOT / "data/build_manifest.json"))["inputs_sha256"],
}
json.dump(contract, open(OUT / "ENSEMBLE_CONTRACT_CORE6_20260926.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(contract["recipes"], ensure_ascii=False, indent=1))
