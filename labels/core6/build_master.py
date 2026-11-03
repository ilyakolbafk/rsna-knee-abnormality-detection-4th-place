"""Собрать мастер-таблицы для замеров Core6.

Выходы:
  data/gold58_master.csv   — 696 клеток (58 × 12): gold, все члены Core6, lixin, повторы Opus/Fable/Sol/Astra/ultra, v6.1, AUX60
  data/train4407_core6_long.csv — 52 884 клетки: шесть членов Core6 + lixin, reader_count, core6_equal (сверено с Codex)
  data/build_manifest.json — хэши входов и проверки
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
I = ROOT / "inputs"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# члены Core6 в порядке ENSEMBLE_CONTRACT.yaml (core6-sol-max-full-report-20260925)
CORE6 = ["laymond_qwen3", "yunus_gemini", "p3_xhigh", "qwen_family", "opus_v1", "sol6_max_v1"]

SRC = {
    "img": I / "Claude outputs/core6_text/gold58_cells_core6_v1_with_image.csv",
    "rep": I / "Claude outputs/core6_text/gold58_cells_core6_repeats.csv",
    "sol": I / ".codex/runs/core6-text-sol-max-prompt-v1-20260925/SCORED_CELLS.csv",
    "third": I / ".codex/runs/core6-astra-sol-third-repeat-20260926/SCORED_CELLS.csv",
    "ultra": I / ".codex/runs/core6-sol-ultra-gold58-two-run-20260926/SCORED_CELLS.csv",
    "gold": I / "data/gold58_mri/gold58_labels.csv",
    "panel": I / ".codex/runs/rsna-knee-report-label-shift-core5-vs-legacy-v1/results/full_comparison_panel_4407x12.jsonl.gz",
    "opus_all": I / "Claude outputs/core6_text/train4349/opus_v1_train4407.csv",
    "sol_all": I / ".codex/runs/core6-sol-max-full-report-20260925/sol6_max_v1_train4407.csv",
    "core6_all": I / ".codex/runs/core6-sol-max-full-report-20260925/core6_equal_train4407.csv",
    "aux60_oof": I / ".codex/runs/rsna-oof-core5-next-20260912/OOF_AUX60_FIXED/probabilities.csv",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    img = pd.read_csv(SRC["img"])
    rep = pd.read_csv(SRC["rep"])
    sol = pd.read_csv(SRC["sol"])
    third = pd.read_csv(SRC["third"])
    ultra = pd.read_csv(SRC["ultra"])
    gold = pd.read_csv(SRC["gold"])

    m = img[["case_id", "uid", "label", "gold", "laymond_qwen3", "lixin_sol56", "yunus_gemini",
             "p3_xhigh", "qwen_family", "v61", "aux60", "core5"]].copy()
    m = m.merge(rep[["uid", "label", "opus_r1", "opus_r2", "opus_r3", "fable_r1", "fable_r2", "fable_r3"]],
                on=["uid", "label"], validate="one_to_one")
    m = m.merge(sol[["uid", "label", "gold", "sol6_max_v1", "laymond_qwen3"]]
                .rename(columns={"gold": "gold_chk", "laymond_qwen3": "lay_chk"}),
                on=["uid", "label"], validate="one_to_one")
    m = m.merge(third[["uid", "label", "sol_r1", "sol_r2", "sol_r3", "astra_r1", "astra_r2", "astra_r3", "core6_equal"]]
                .rename(columns={"core6_equal": "core6_equal_codex"}),
                on=["uid", "label"], validate="one_to_one")
    m = m.merge(ultra[["uid", "label", "ultra_r1", "ultra_r2"]], on=["uid", "label"], validate="one_to_one")
    assert len(m) == 696 and m.uid.nunique() == 58
    assert (m.gold == m.gold_chk).all() and np.allclose(m.laymond_qwen3, m.lay_chk)

    # официальный gold из data/gold58_mri/gold58_labels.csv
    gl = gold.melt(id_vars=["gold_study_index", "StudyInstanceUID"], value_vars=TARGETS,
                   var_name="label", value_name="gold_file")
    m = m.merge(gl.rename(columns={"StudyInstanceUID": "uid"}), on=["uid", "label"], validate="one_to_one")
    assert (m.gold == m.gold_file).all()
    m["opus_v1"] = m["opus_r1"]
    m["core6_equal"] = m[CORE6].mean(axis=1)
    assert np.allclose(m.core6_equal, m.core6_equal_codex, atol=1e-9)
    m["split"] = np.where(m.case_id <= "G040", "dev40", "hold18")
    m["label"] = pd.Categorical(m.label, TARGETS, ordered=True)
    m = m.sort_values(["case_id", "label"]).reset_index(drop=True)
    cols = ["case_id", "gold_study_index", "uid", "label", "split", "gold",
            "laymond_qwen3", "yunus_gemini", "p3_xhigh", "qwen_family", "opus_v1", "sol6_max_v1",
            "lixin_sol56", "opus_r2", "opus_r3", "fable_r1", "fable_r2", "fable_r3",
            "sol_r1", "sol_r2", "sol_r3", "astra_r1", "astra_r2", "astra_r3", "ultra_r1", "ultra_r2",
            "v61", "aux60", "core5", "core6_equal"]
    m = m[cols]
    m.to_csv(OUT / "gold58_master.csv", index=False)

    # --- 4407 × 12 длинная таблица шести членов
    with gzip.open(SRC["panel"], "rt") as f:
        panel = pd.DataFrame(json.loads(s) for s in f)
    panel = panel.rename(columns={"target": "label", "old_laymond_qwen3": "laymond_qwen3",
                                  "old_yunus_gemini": "yunus_gemini", "old_lixin_sol56": "lixin_sol56"})
    panel = panel[["StudyInstanceUID", "label", "split", "laymond_qwen3", "yunus_gemini", "p3_xhigh",
                   "qwen_family", "lixin_sol56", "new_core5_equal"]]
    assert len(panel) == 52884
    opus = pd.read_csv(SRC["opus_all"], dtype={"StudyInstanceUID": str})
    sol_all = pd.read_csv(SRC["sol_all"], dtype={"StudyInstanceUID": str})
    c6 = pd.read_csv(SRC["core6_all"], dtype={"StudyInstanceUID": str})
    ol = opus.melt(id_vars="StudyInstanceUID", value_vars=TARGETS, var_name="label", value_name="opus_v1")
    sl = sol_all.melt(id_vars="StudyInstanceUID", value_vars=TARGETS, var_name="label", value_name="sol6_max_v1")
    cl = c6.melt(id_vars="StudyInstanceUID", value_vars=TARGETS, var_name="label", value_name="core6_equal_codex")
    long = panel.merge(ol, on=["StudyInstanceUID", "label"], validate="one_to_one") \
                .merge(sl, on=["StudyInstanceUID", "label"], validate="one_to_one") \
                .merge(cl, on=["StudyInstanceUID", "label"], validate="one_to_one")
    assert len(long) == 52884
    long["reader_count"] = long[CORE6].notna().sum(axis=1)
    assert long.reader_count.value_counts().to_dict() == {6: 52860, 5: 24}
    long["core6_equal"] = long[CORE6].mean(axis=1, skipna=True)
    assert np.allclose(long.core6_equal, long.core6_equal_codex, atol=1e-9)
    long["is_gold58"] = long.StudyInstanceUID.isin(m.uid)
    assert long.is_gold58.sum() == 696
    # сверка Gold58-строк длинной таблицы с мастер-таблицей
    chk = long[long.is_gold58].merge(m[["uid", "label", "opus_v1", "sol6_max_v1", "laymond_qwen3"]]
                                     .rename(columns={"uid": "StudyInstanceUID"}),
                                     on=["StudyInstanceUID", "label"], suffixes=("", "_m"))
    for c in ["opus_v1", "sol6_max_v1", "laymond_qwen3"]:
        assert np.allclose(chk[c], chk[c + "_m"]), c
    long["label"] = pd.Categorical(long.label, TARGETS, ordered=True)
    long = long.sort_values(["StudyInstanceUID", "label"]).reset_index(drop=True)
    long = long[["StudyInstanceUID", "label", "split", "is_gold58", *CORE6, "lixin_sol56",
                 "reader_count", "core6_equal", "new_core5_equal"]]
    long.to_csv(OUT / "train4407_core6_long.csv", index=False)

    # AUX60 OOF (4345 исследований) — для прокси-подбора без Gold
    oof = pd.read_csv(SRC["aux60_oof"], dtype={"StudyInstanceUID": str})[["StudyInstanceUID", *TARGETS]]
    assert len(oof) == 4345
    oofl = oof.melt(id_vars="StudyInstanceUID", value_vars=TARGETS, var_name="label", value_name="aux60_oof")
    oofl.to_csv(OUT / "aux60_oof_long.csv", index=False)
    print("OOF studies in long:", long.StudyInstanceUID.isin(oof.StudyInstanceUID).sum() // 12,
          "of which gold58:", long[long.is_gold58].StudyInstanceUID.isin(oof.StudyInstanceUID).sum() // 12)

    manifest = {
        "inputs_sha256": {k: sha(v) for k, v in SRC.items()},
        "outputs_sha256": {p.name: sha(p) for p in [OUT / "gold58_master.csv", OUT / "train4407_core6_long.csv",
                                                     OUT / "aux60_oof_long.csv"]},
        "core6_members": CORE6,
        "checks": {
            "gold58_cells": len(m), "gold_matches_gold58_labels_csv": True,
            "core6_equal_matches_codex_gold58": True, "core6_equal_matches_codex_4407": True,
            "reader_count_distribution": {"6": 52860, "5": 24},
            "positives_per_label_gold58": m.groupby("label", observed=True).gold.sum().astype(int).to_dict(),
            "dev40_studies": int(m[m.split == "dev40"].uid.nunique()),
            "hold18_studies": int(m[m.split == "hold18"].uid.nunique()),
        },
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest["checks"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
