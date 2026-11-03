"""Fail-closed, mechanical readout of the frozen five-run B/L16 screen.

This is a shortlist calculation, not a winner selection or a five-fold gate.
The bootstrap conditions on the trained predictions and resamples studies; it
does not estimate training-seed variance or a posterior probability of benefit.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .constants import ID_COLUMN, TARGET_COLUMNS
from .pseudo_oof import binarise_pseudo_labels, evaluate_oof_frame, macro_auc_from_arrays
from .screen_queue import MANDATORY, run_id, validate_design, verify_completion
from .training_state import file_sha256, json_sha256

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260905
LOCAL_CONFIG_KEYS = ("local_head", "local_gate_init")
METRIC_PARAMETERS = dict(min_confidence=0.5, positive_threshold=0.5, abstention_band=0.0)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _boolean(series: pd.Series, name: str) -> np.ndarray:
    mapping = {True: True, False: False, 1: True, 0: False, "True": True, "False": False,
               "true": True, "false": False, "1": True, "0": False}
    if series.isna().any() or not series.isin(mapping).all():
        raise ValueError(f"{name} must contain explicit booleans")
    return series.map(mapping).to_numpy(dtype=bool)


def expected_oof_metadata(valid: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Mirror only the frozen dataset label conversion, never load image pixels.

    The loader rounds labels to float32 and prediction export emits NaN for
    masked cells. Confidence is copied from the original manifest as float64.
    """
    required = {ID_COLUMN, "fold", "is_gold", "report_group", *TARGET_COLUMNS,
                *(f"confidence__{t}" for t in TARGET_COLUMNS)}
    missing = required - set(valid.columns)
    if missing:
        raise ValueError(f"Manifest lacks frozen metadata: {sorted(missing)}")
    result = valid[[ID_COLUMN, "fold", "is_gold", "report_group"]].copy()
    if result[[ID_COLUMN, "report_group"]].isna().any().any():
        raise ValueError("Study/report group identifiers cannot be missing")
    gold = _boolean(valid.is_gold, "is_gold")
    result["is_gold"] = gold
    training = config["train"]
    for target in TARGET_COLUMNS:
        values = pd.to_numeric(valid[target], errors="raise").to_numpy(float)
        if np.isinf(values).any() or ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)).any():
            raise ValueError(f"Label outside [0,1]: {target}")
        if not np.isin(values[gold & np.isfinite(values)], [0., 1.]).all():
            raise ValueError(f"Gold labels must be binary: {target}")
        mask = np.isfinite(values)
        explicit = f"label_mask__{target}"
        if explicit in valid:
            # Dataset treats a missing explicit mask as false.
            mask &= _boolean(valid[explicit].fillna(False), explicit)
        confidence_column = f"confidence__{target}"
        confidence = pd.to_numeric(valid[confidence_column], errors="raise").to_numpy(float)
        if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
            raise ValueError(f"Invalid frozen confidence: {target}")
        weights = confidence.astype(np.float32)
        weights[weights < float(training.get("pseudo_min_confidence", 0.0))] = 0
        weights = np.clip(weights, 0, 1) * float(training.get("pseudo_weight", 1.0))
        mask &= gold | (weights > 0)
        result[target] = np.where(mask, values.astype(np.float32).astype(float), np.nan)
        result[confidence_column] = confidence
    return result.sort_values(ID_COLUMN).reset_index(drop=True)


def verify_oof_metadata(oof: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    if not oof[ID_COLUMN].is_unique or set(oof[ID_COLUMN]) != set(expected[ID_COLUMN]):
        raise ValueError("OOF UID set differs from frozen holdout")
    actual = oof.sort_values(ID_COLUMN).reset_index(drop=True)
    for column in expected:
        if column not in actual:
            raise ValueError(f"OOF lacks frozen metadata: {column}")
        if column == "is_gold":
            same = np.array_equal(_boolean(actual[column], column), expected[column].to_numpy())
        elif column in {ID_COLUMN, "report_group"}:
            same = actual[column].notna().all() and np.array_equal(
                actual[column].astype(str).to_numpy(), expected[column].astype(str).to_numpy())
        else:
            same = np.array_equal(pd.to_numeric(actual[column], errors="raise").to_numpy(float),
                                  expected[column].to_numpy(float), equal_nan=True)
        if not same:
            raise ValueError(f"OOF frozen metadata mismatch: {column}")
    return actual


def _semantic_config(resolved: dict) -> dict:
    semantic = deepcopy(resolved)
    if semantic["model"].get("encoder_checkpoint"):
        semantic["model"]["encoder_checkpoint"] = "<content-hash>"
    if semantic["data"].get("cache_format") == "grayscale_shards_v1":
        for key in ("cache_root", "cache_roots", "images_root"):
            semantic["data"].pop(key, None)
    if semantic["data"].get("centering_manifest"):
        semantic["data"]["centering_manifest"] = "<content-hash>"
    return semantic


def comparable_identity(identity: dict, *, remove_seed: bool = False,
                        remove_local: bool = False, remove_fold: bool = False) -> dict:
    """Remove only declared treatment fields; runtime/code/assets remain binding."""
    body = deepcopy(identity)
    body.pop("sha256", None)
    if remove_seed:
        body["semantic_config"].pop("seed")
    if remove_local:
        for key in LOCAL_CONFIG_KEYS:
            body["semantic_config"]["model"].pop(key, None)
    if remove_fold:
        for key in ("fold", "train_frame_sha256", "valid_frame_sha256"):
            body.pop(key)
    return body


def _summary(samples: list[float], point: float | None) -> dict:
    values = np.asarray(samples, dtype=float)
    return {"delta": point, "ci_low": float(np.quantile(values, .025)) if len(values) else None,
            "ci_high": float(np.quantile(values, .975)) if len(values) else None,
            "usable_iterations": len(values)}


def paired_readout(candidate: pd.DataFrame, baseline: pd.DataFrame, *, seed: int) -> dict:
    """Candidate minus baseline, shared study draws for all target diagnostics."""
    a = candidate.sort_values(ID_COLUMN).reset_index(drop=True)
    b = baseline.sort_values(ID_COLUMN).reset_index(drop=True)
    # This function is independently safe to call: labels, masks and groups must match.
    metadata = a[[ID_COLUMN, "fold", "is_gold", "report_group", *TARGET_COLUMNS,
                  *(f"confidence__{t}" for t in TARGET_COLUMNS)]]
    verify_oof_metadata(b, metadata)
    labels = a[list(TARGET_COLUMNS)].to_numpy(float)
    confidence = a[[f"confidence__{t}" for t in TARGET_COLUMNS]].to_numpy(float)
    gold = _boolean(a.is_gold, "is_gold")
    pa = a[[f"pred__{t}" for t in TARGET_COLUMNS]].to_numpy(float)
    pb = b[[f"pred__{t}" for t in TARGET_COLUMNS]].to_numpy(float)
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
        raise ValueError("Paired predictions must be finite")
    binary, mask = binarise_pseudo_labels(labels, confidence, gold, **METRIC_PARAMETERS)
    gold_mask = gold[:, None] & np.isfinite(labels) & np.isin(labels, [0., 1.])
    scores_a, scores_b = evaluate_oof_frame(a, **METRIC_PARAMETERS), evaluate_oof_frame(b, **METRIC_PARAMETERS)
    distributions = {scope: {"macro": [], **{t: [] for t in TARGET_COLUMNS}}
                     for scope in ("pseudo", "gold")}
    calibration_samples = []
    target_count_range = {"pseudo": [], "gold": []}
    # Match evaluate_oof_frame's soft-calibration clipping/finite-cell policy.
    error_delta = (np.clip(pa, 1e-6, 1-1e-6)-labels)**2 - (np.clip(pb, 1e-6, 1-1e-6)-labels)**2
    rng = np.random.default_rng(seed)
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = rng.integers(0, len(a), size=len(a))
        for scope, use_mask in (("pseudo", mask), ("gold", gold_mask)):
            aa = macro_auc_from_arrays(binary[sample], pa[sample], use_mask[sample])
            bb = macro_auc_from_arrays(binary[sample], pb[sample], use_mask[sample])
            target_count_range[scope].append(aa["valid_target_count"])
            if aa["macro_auc"] is not None:
                distributions[scope]["macro"].append(aa["macro_auc"]-bb["macro_auc"])
            for target in TARGET_COLUMNS:
                if aa["per_target"][target] is not None:
                    distributions[scope][target].append(aa["per_target"][target]-bb["per_target"][target])
        errors = error_delta[sample]
        if np.isfinite(errors).any():
            calibration_samples.append(float(np.nanmean(errors)))
    report = {"direction": "L16 minus B", "resampling_unit": "StudyInstanceUID",
              "seed": seed, "requested_iterations": BOOTSTRAP_ITERATIONS,
              "ci_method": "paired study bootstrap percentile 95%",
              "conditional_on_fixed_trained_predictions": True}
    for scope in ("pseudo", "gold"):
        point_a, point_b = scores_a[scope], scores_b[scope]
        delta = None if point_a["macro_auc"] is None else point_a["macro_auc"]-point_b["macro_auc"]
        report[scope] = _summary(distributions[scope]["macro"], delta)
        report[scope]["valid_target_count_range"] = [min(target_count_range[scope]), max(target_count_range[scope])]
        report[scope]["per_target"] = {
            t: _summary(distributions[scope][t], None if point_a["per_target"][t] is None
                        else point_a["per_target"][t]-point_b["per_target"][t]) for t in TARGET_COLUMNS}
    brier_a, brier_b = scores_a["calibration"]["brier"], scores_b["calibration"]["brier"]
    report["soft_brier"] = _summary(calibration_samples, None if brier_a is None else brier_a-brier_b)
    report["soft_brier"]["lower_is_better"] = True
    return report


def _verify_run(folder: Path, run: dict, manifest: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    from .train import build_fold_partitions
    expected_ids = set(manifest.loc[manifest.fold == run["fold"], ID_COLUMN])
    marker = verify_completion(folder, fold=run["fold"], epochs=12, expected_ids=expected_ids,
                               expected_identity_sha256=run["identity_sha256"])
    identity = _read_json(folder / "training_identity.json")
    resolved = yaml.safe_load((folder / "resolved_config.yaml").read_text())
    semantic = identity["semantic_config"]
    if semantic != _semantic_config(resolved):
        raise ValueError("Resolved config differs from identity")
    if (identity["fold"] != run["fold"] or semantic["seed"] != run["seed"]
            or identity["target_order"] != list(TARGET_COLUMNS)):
        raise ValueError("Run fold/seed/target order differs from identity")
    if identity["device_type"] != "cuda" or not identity.get("accelerator", {}).get("device_names"):
        raise ValueError("Frozen paid screen requires a recorded CUDA accelerator")
    if (run["arm"] == "L16") != (semantic["model"].get("local_head") == "l16"):
        raise ValueError("Run arm differs from identity")
    tr = semantic["train"]
    if (tr["epochs"] != 12 or tr["selection_metric"] != "pseudo_macro_auc"
            or tr.get("pseudo_eval_min_confidence", .5) != .5
            or tr["validation_scope"] != "all_holdout" or tr["max_train_tokens"] != 46
            or tr["batch_size"] * tr["accumulate_steps"] != 8
            or tr.get("data_rng_mode") != "study_epoch_v1" or semantic["data"]["image_size"] != 224):
        raise ValueError("Run scoring/selection policy differs from frozen screen")
    train, valid = build_fold_partitions(manifest, fold=run["fold"], n_splits=5, validation_scope="all_holdout")
    if identity["train_frame_sha256"] != _frame_hash(train) or identity["valid_frame_sha256"] != _frame_hash(valid):
        raise ValueError("Frozen train/valid frame hashes differ from manifest")
    if identity.get("series_frame_sha256") is not None:
        raise ValueError("Sharded screen expects no separate series frame")
    oof = pd.read_csv(folder / "oof_predictions.csv", dtype={ID_COLUMN: str}, float_precision="round_trip")
    oof = verify_oof_metadata(oof, expected_oof_metadata(valid, resolved))
    scorecard = evaluate_oof_frame(oof, **METRIC_PARAMETERS)
    summary = _read_json(folder / "summary.json")
    history = _read_json(folder / "history.json")
    best_epoch = summary["best_epoch"]
    if marker["best_epoch"] != best_epoch or best_epoch not in range(12):
        raise ValueError("Best epoch differs between completion and summary")
    values = [record["metrics"]["pseudo"]["macro_auc"] for record in history]
    if any(v is None or not np.isfinite(v) for v in values):
        raise ValueError("History has an undefined primary selection metric")
    if best_epoch != int(np.argmax(values)):
        raise ValueError("Selected epoch is not the first maximum of frozen pseudo selection")
    measured = scorecard["pseudo"]["macro_auc"]
    if measured is None or not np.isclose(measured, values[best_epoch], atol=1e-12, rtol=0):
        raise ValueError("OOF score differs from selected history metric")
    if not np.isclose(measured, summary["best_selection_score"], atol=1e-12, rtol=0):
        raise ValueError("OOF score differs from summary")
    if summary.get("selection_metric") != "pseudo_macro_auc":
        raise ValueError("Summary selection policy differs from identity")
    expected_weights = "ema" if float(tr.get("ema_decay", 0)) > 0 else "raw"
    if summary.get("selected_weights") != expected_weights:
        raise ValueError("Selected weight type differs from frozen recipe")
    for key, name in (("checkpoint", "best.pt"), ("predictions", "oof_predictions.csv")):
        if summary["best_artifacts"][key]["sha256"] != marker["files"][name]["sha256"]:
            raise ValueError("Summary best artifact differs from exported artifact")
    if marker["files"].get("best_validation_predictions.csv", {}).get("sha256") != marker["files"]["oof_predictions.csv"]["sha256"]:
        raise ValueError("OOF differs from selected best validation predictions")
    report = {"run_id": run["run_id"], "arm": run["arm"], "fold": run["fold"], "seed": run["seed"],
              "identity_sha256": identity["sha256"], "completion_sha256": file_sha256(folder / "COMPLETED.json"),
              "artifact_hashes": marker["files"], "best_epoch_zero_based": best_epoch,
              "selected_weights": summary.get("selected_weights"), "scorecard": scorecard,
              "curves": [{"epoch_zero_based": r["epoch"], "train_loss": r.get("train_loss"),
                          "pseudo_macro_auc": r["metrics"]["pseudo"]["macro_auc"],
                          "gold_macro_auc": r["metrics"].get("gold", {}).get("macro_auc"),
                          "train_seconds": r.get("train_seconds"), "validation_seconds": r.get("validation_seconds"),
                          "skipped_optimizer_steps": r.get("skipped_optimizer_steps")} for r in history]}
    return report, oof, identity


def verify_diagnostics(path: Path, contract: dict, runs: dict, manifest: pd.DataFrame) -> dict:
    """Verify payloads through the exporter verifier, then bind them to this queue."""
    from .local_diagnostics import verify_diagnostics_bundle
    files = [path] if path.is_file() else sorted(path.rglob("DIAGNOSTICS_COMPLETE.json"))
    if not files:
        raise FileNotFoundError("DIAGNOSTICS_COMPLETE.json is missing")
    verified, selection_hashes, selections, all_studies = {}, set(), [], set()
    expected_runs = {run_id("L16", f, 20260904) for f in (0, 3)}
    for completion_file in files:
        value = _read_json(completion_file)
        root = completion_file.parent.resolve()
        for name in value["files"]:
            payload = root / name
            if not payload.resolve().is_relative_to(root):
                raise ValueError("Unsafe diagnostic payload path")
        value = verify_diagnostics_bundle(root)
        rid = value["run_id"]
        if rid not in expected_runs or rid in verified:
            raise ValueError("Unexpected or duplicate diagnostic run")
        if rid not in runs:
            raise FileNotFoundError(f"Verified training output missing for diagnostics: {rid}")
        run = runs[rid]
        if (value["status"] != "COMPLETE" or value["fold"] != run["fold"]
                or value["training_identity_sha256"] != run["identity_sha256"]
                or value["checkpoint_sha256"] != run["artifact_hashes"]["best.pt"]["sha256"]
                or value["checkpoint_epoch"] != run["best_epoch_zero_based"]
                or value["labels"] != list(TARGET_COLUMNS)):
            raise ValueError("Diagnostic checkpoint/identity/fold/label binding mismatch")
        checks = value["checks"]
        selection_hashes.add((value["selection_sha256"], value["selection_contract_sha256"]))
        selection = pd.read_csv(root / "diagnostic_studies.csv", dtype={ID_COLUMN: str})
        selection_contract = _read_json(root / "selection_contract.json")
        if selection_contract["manifest_sha256"] != contract["input_files"][contract["manifest"]]:
            raise ValueError("Diagnostic selection uses a different frozen manifest")
        if not selection[ID_COLUMN].is_unique or not set(selection.fold) <= {0, 3}:
            raise ValueError("Invalid frozen diagnostic selection")
        if set(selection[ID_COLUMN]) - set(manifest[ID_COLUMN]):
            raise ValueError("Diagnostic selection is outside frozen manifest")
        lookup = manifest.set_index(ID_COLUMN).fold
        if not np.array_equal(selection.fold.to_numpy(), selection[ID_COLUMN].map(lookup).to_numpy()):
            raise ValueError("Diagnostic selection fold mismatch")
        selected = set(value["study_uids"])
        if len(selection) < 20 and not selection_contract.get("shortfall_reason"):
            raise ValueError("Reduced diagnostic selection requires its frozen shortfall reason")
        selections.append(selection)
        all_studies.update(selected)
        verified[rid] = {"manifest_sha256": file_sha256(completion_file), "study_count": len(selected),
                         "checks": checks, "files": value["files"]}
    if set(verified) != expected_runs:
        raise FileNotFoundError("Mandatory diagnostics missing for one L16 fold")
    if len(selection_hashes) != 1:
        raise ValueError("Diagnostic folds use different frozen selections")
    if set(selections[0][ID_COLUMN]) != all_studies or not 10 <= len(all_studies) <= 20:
        raise ValueError("Diagnostic selection must cover 10–20 unique holdout studies")
    # Freeze selection against the queue, rather than accepting a post-result replacement.
    frozen_hashes = set(contract["input_files"].values())
    selection_pair = next(iter(selection_hashes))
    if not set(selection_pair) <= frozen_hashes:
        raise ValueError("Diagnostic selection was not frozen in queue inputs")
    return {"status": "VERIFIED", "studies": len(all_studies), "runs": verified,
            "selection_sha256": selection_pair[0], "selection_contract_sha256": selection_pair[1]}


def analyze_screen(contract_path: str | Path, *, output_root: str | Path | None = None,
                   manifest_path: str | Path | None = None,
                   diagnostics_path: str | Path | None = None) -> dict[str, Any]:
    """Verify a frozen queue contract and relocated backups without running jobs."""
    contract_path = Path(contract_path)
    report: dict[str, Any] = {
        "format_version": 1, "status": "INCOMPLETE", "numeric_gate_pass": None,
        "scientific_screen_status": "INCOMPLETE",
        "threshold": None, "d_seed": None, "mean_delta": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_parameters": METRIC_PARAMETERS, "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED, "runs": {}, "comparisons": {}, "errors": [], "missing": [],
        "claim_strength": "descriptive paired screen; no final winner or fivefold authorization",
        "limitations": ["One seed difference is sensitivity, not a training-variance estimate.",
                       "Study bootstrap conditions on trained predictions; best-epoch and prior fold reuse remain selection effects.",
                       "Gold/per-target/Brier and attention require review; attention is not clinical localization."],
        "source_hashes": {name: file_sha256(Path(__file__).with_name(name)) for name in
                          ("screen_analysis.py", "pseudo_oof.py", "screen_queue.py", "training_state.py", "train.py", "local_diagnostics.py")},
    }
    try:
        contract = _read_json(contract_path)
        report["source_hashes"]["contract"] = file_sha256(contract_path)
        validate_design(contract)  # Historical analysis intentionally does not enforce the current deadline.
        root = Path(output_root or contract["output_root"])
        manifest_file = Path(manifest_path or contract["manifest"])
        expected_manifest_sha = contract["input_files"][contract["manifest"]]
        if file_sha256(manifest_file) != expected_manifest_sha:
            raise ValueError("Manifest bytes differ from frozen queue input")
        report["source_hashes"]["manifest"] = expected_manifest_sha
        manifest = pd.read_csv(manifest_file, dtype={ID_COLUMN: str})
        if not manifest[ID_COLUMN].is_unique or manifest[ID_COLUMN].isna().any():
            raise ValueError("Manifest study IDs must be complete and unique")
        if manifest.groupby("report_group", dropna=False).fold.nunique().max() != 1:
            raise ValueError("Frozen report groups cross folds")
        frames, identities = {}, {}
        for run in contract["runs"]:
            rid = run["run_id"]
            folder = root / rid
            if not (folder / "COMPLETED.json").exists():
                report["missing"].append(f"completed run: {rid}")
                continue
            try:
                run_report, frames[rid], identities[rid] = _verify_run(folder, run, manifest)
                report["runs"][rid] = run_report
            except (ValueError, KeyError, TypeError, OSError) as error:
                report["errors"].append(f"{rid}: {error}")
        if identities:
            reference = comparable_identity(next(iter(identities.values())), remove_seed=True, remove_local=True, remove_fold=True)
            for rid, identity in identities.items():
                common = comparable_identity(identity, remove_seed=True, remove_local=True, remove_fold=True)
                if common != reference:
                    report["errors"].append(f"{rid}: non-arm/non-seed code/runtime/assets/recipe mismatch")
            for fold in (0, 3):
                b, l = run_id("B", fold, 20260904), run_id("L16", fold, 20260904)
                if b in identities and l in identities and comparable_identity(identities[b], remove_local=True) != comparable_identity(identities[l], remove_local=True):
                    report["errors"].append(f"fold {fold}: B/L identity differs outside local keys")
            first, repeat = run_id("B", 3, 20260904), run_id("B", 3, 20260914)
            if first in identities and repeat in identities and comparable_identity(identities[first], remove_seed=True) != comparable_identity(identities[repeat], remove_seed=True):
                report["errors"].append("Baseline seed repeat changes non-seed identity fields")
        # A diagnostic adapter below must verify the export, not merely its presence.
        if diagnostics_path is None:
            report["missing"].append("verified diagnostics manifest")
        else:
            report["diagnostics"] = verify_diagnostics(Path(diagnostics_path), contract, report["runs"], manifest)
        if report["errors"]:
            report["status"] = "FAIL"
            return report
        for fold in (0, 3):
            b, l = run_id("B", fold, 20260904), run_id("L16", fold, 20260904)
            if b in frames and l in frames:
                report["comparisons"][str(fold)] = paired_readout(frames[l], frames[b], seed=BOOTSTRAP_SEED + fold)
        first, repeat = run_id("B", 3, 20260904), run_id("B", 3, 20260914)
        if first in report["runs"] and repeat in report["runs"]:
            scores = [report["runs"][rid]["scorecard"]["pseudo"]["macro_auc"] for rid in (first, repeat)]
            if None not in scores:
                report["d_seed"] = abs(scores[1]-scores[0])
                report["threshold"] = max(.003, 2*report["d_seed"])
        if len(report["comparisons"]) == 2 and report["threshold"] is not None:
            deltas = [report["comparisons"][str(f)]["pseudo"]["delta"] for f in (0, 3)]
            if None not in deltas:
                report["mean_delta"] = float(np.mean(deltas))
                report["numeric_gate_pass"] = bool(all(d > 0 for d in deltas) and report["mean_delta"] >= report["threshold"])
        if report["missing"] or report["numeric_gate_pass"] is None:
            report["status"] = "INCOMPLETE"
        else:
            report["status"] = "SHORTLIST_REQUIRES_REVIEW" if report["numeric_gate_pass"] else "REJECTED_NUMERIC_GATE"
            report["scientific_screen_status"] = report["status"]
        return report
    except FileNotFoundError as error:
        report["missing"].append(str(error))
        report["status"] = "INCOMPLETE"
        return report
    except (ValueError, KeyError, TypeError, OSError) as error:
        report["errors"].append(str(error))
        report["status"] = "FAIL"
        return report
