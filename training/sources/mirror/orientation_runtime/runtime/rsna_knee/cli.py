"""Command-line entry point for audits, folds, caching, training, and inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .cache import build_cache
from .config import load_config, with_path_overrides
from .folds import (
    assign_grouped_folds,
    fold_diagnostics,
    prepare_training_manifest,
)
from .metadata import audit_metadata
from .reports import audit_reports
from .report_teacher import evaluate_report_teacher


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _load_runtime_config(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    return with_path_overrides(
        config,
        images_root=getattr(arguments, "images_root", None),
        cache_root=getattr(arguments, "cache_root", None),
        cache_roots=getattr(arguments, "cache_roots", None),
    )


def command_audit_metadata(arguments: argparse.Namespace) -> None:
    train = pd.read_csv(arguments.train_csv)
    series = pd.read_csv(arguments.series_csv)
    summary = audit_metadata(train, series)
    _write_json(summary, arguments.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_audit_reports(arguments: argparse.Namespace) -> None:
    train = pd.read_csv(arguments.train_csv)
    summary, duplicates = audit_reports(train)
    _write_json(summary, arguments.output_json)
    if arguments.duplicates_csv:
        path = Path(arguments.duplicates_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        duplicates.to_csv(path, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_make_folds(arguments: argparse.Namespace) -> None:
    train = pd.read_csv(arguments.train_csv)
    pseudo = pd.read_csv(arguments.pseudo_csv) if arguments.pseudo_csv else None
    manifest = prepare_training_manifest(train, pseudo)
    manifest = assign_grouped_folds(
        manifest,
        n_splits=arguments.n_splits,
        seed=arguments.seed,
    )
    output = Path(arguments.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    diagnostics = fold_diagnostics(manifest)
    diagnostics_path = arguments.diagnostics_json or output.with_suffix(".diagnostics.json")
    _write_json(diagnostics, diagnostics_path)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


def command_report_teacher_pilot(arguments: argparse.Namespace) -> None:
    train = pd.read_csv(arguments.train_csv, dtype={"StudyInstanceUID": str})
    teacher, decisions, summary = evaluate_report_teacher(
        train,
        bootstrap_iterations=arguments.bootstrap_iterations,
        seed=arguments.seed,
    )
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher.to_csv(output_dir / "report_teacher_all.csv", index=False)
    decisions.to_csv(output_dir / "gold_decisions_long.csv", index=False)
    decisions.loc[~decisions["correct_at_0_5"]].to_csv(
        output_dir / "gold_disagreements.csv", index=False
    )
    _write_json(summary, output_dir / "audit_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_make_pseudo_labels(arguments: argparse.Namespace) -> None:
    from .weak_labels import evaluate_rule_teacher, run_rule_teacher

    train = pd.read_csv(arguments.train_csv)
    pseudo, evidence = run_rule_teacher(train)
    diagnostics, errors = evaluate_rule_teacher(
        train,
        pseudo,
        evidence,
        n_bootstrap=arguments.bootstrap_samples,
        seed=arguments.seed,
    )
    output = Path(arguments.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pseudo.to_csv(output, index=False)
    diagnostics_path = arguments.diagnostics_json or output.with_suffix(".diagnostics.json")
    _write_json(diagnostics, diagnostics_path)
    if arguments.errors_csv:
        errors_path = Path(arguments.errors_csv)
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(errors_path, index=False)
    if arguments.evidence_csv:
        evidence_path = Path(arguments.evidence_csv)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence.to_csv(evidence_path, index=False)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


def command_make_consensus_labels(arguments: argparse.Namespace) -> None:
    from .consensus import (
        build_public_consensus,
        evaluate_public_consensus,
        load_public_teacher,
    )

    train = pd.read_csv(arguments.train_csv, dtype={"StudyInstanceUID": str})
    teachers = {
        "pilkwang": load_public_teacher(arguments.pilkwang_csv, "pilkwang"),
        "steven_v2": load_public_teacher(arguments.steven_csv, "steven_v2"),
        "lixin_sol56": load_public_teacher(arguments.lixin_csv, "lixin_sol56"),
    }
    consensus = build_public_consensus(
        train,
        teachers,
        weight_mode=arguments.weight_mode,
    )
    diagnostics = evaluate_public_consensus(
        train,
        consensus,
        teachers,
        n_bootstrap=arguments.bootstrap_samples,
        seed=arguments.seed,
    )
    output = Path(arguments.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    consensus.to_csv(output, index=False)
    diagnostics_path = arguments.diagnostics_json or output.with_suffix(
        ".diagnostics.json"
    )
    _write_json(diagnostics, diagnostics_path)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


def command_select_pilot(arguments: argparse.Namespace) -> None:
    from .pilot import select_pilot_studies

    manifest = pd.read_csv(arguments.manifest_csv, dtype={"StudyInstanceUID": str})
    balance_manifests = None
    if arguments.balance_manifest_csv:
        balance_manifests = {}
        for item in arguments.balance_manifest_csv:
            if "=" not in item:
                raise ValueError("--balance-manifest-csv must use NAME=PATH")
            name, path = item.split("=", maxsplit=1)
            if not name or name in balance_manifests:
                raise ValueError(f"Invalid or duplicate balance manifest name: {name!r}")
            balance_manifests[name] = pd.read_csv(
                path, dtype={"StudyInstanceUID": str}
            )
    selected, diagnostics = select_pilot_studies(
        manifest,
        balance_manifests=balance_manifests,
        pseudo_count=arguments.pseudo_count,
        minimum_confidence=arguments.minimum_confidence,
        seed=arguments.seed,
    )
    output = Path(arguments.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    diagnostics_path = arguments.diagnostics_json or output.with_suffix(
        ".diagnostics.json"
    )
    _write_json(diagnostics, diagnostics_path)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


def _load_study_ids(path: str | Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if "StudyInstanceUID" not in frame:
        raise ValueError("IDs CSV must contain StudyInstanceUID")
    return frame["StudyInstanceUID"].tolist()


def command_inventory_kaggle_dicoms(arguments: argparse.Namespace) -> None:
    from .remote_zip import open_kaggle_archive, inventory_subset

    study_ids = _load_study_ids(arguments.ids_csv)
    reader, archive = open_kaggle_archive(
        arguments.competition,
        block_size=arguments.block_mib * 2**20,
        cache_blocks=arguments.cache_blocks,
    )
    try:
        inventory, summary = inventory_subset(archive, study_ids)
        summary.update(
            {
                "archive_bytes": reader.size,
                "range_requests": reader.requests_made,
                "range_bytes_received": reader.bytes_received,
            }
        )
    finally:
        archive.close()
        reader.close()
    output = Path(arguments.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output, index=False)
    summary_path = arguments.summary_json or output.with_suffix(".summary.json")
    _write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_download_kaggle_dicoms(arguments: argparse.Namespace) -> None:
    from .remote_zip import extract_subset, inventory_subset, open_kaggle_archive

    study_ids = _load_study_ids(arguments.ids_csv)
    reader, archive = open_kaggle_archive(
        arguments.competition,
        block_size=arguments.block_mib * 2**20,
        cache_blocks=arguments.cache_blocks,
    )
    last_report = {"time": 0.0, "study": ""}

    def report(state: dict[str, Any]) -> None:
        now = state["elapsed_seconds"]
        study_changed = state["study_uid"] != last_report["study"]
        if now - last_report["time"] >= 20 or study_changed:
            gib = state["written_bytes"] / 2**30
            rate = gib / max(now, 1e-9) * 3600
            print(
                f"[{state['position']}/{state['total_files']}] "
                f"{state['study_uid']} | {gib:.2f} GiB written | {rate:.1f} GiB/h",
                flush=True,
            )
            last_report.update({"time": now, "study": state["study_uid"]})

    try:
        inventory, inventory_summary = inventory_subset(archive, study_ids)
        extraction = extract_subset(
            archive,
            inventory,
            arguments.destination,
            minimum_free_bytes=int(arguments.min_free_gib * 2**30),
            verify_existing=not arguments.trust_existing_size,
            progress=report,
        )
        summary = {
            **inventory_summary,
            "archive_bytes": reader.size,
            "range_requests": reader.requests_made,
            "range_bytes_received": reader.bytes_received,
            "url_refreshes": reader.url_refreshes,
            "destination": str(Path(arguments.destination).resolve()),
            "extraction": extraction,
        }
    finally:
        archive.close()
        reader.close()
    if arguments.inventory_csv:
        inventory_path = Path(arguments.inventory_csv)
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(inventory_path, index=False)
    _write_json(summary, arguments.summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_prepare_image_pilot(arguments: argparse.Namespace) -> None:
    from .oof import make_repeated_gold_folds, prepare_condition_manifests

    ids = pd.read_csv(arguments.ids_csv, dtype={"StudyInstanceUID": str})
    source_paths = {
        "rule_only": arguments.rules_manifest,
        "public_v52": arguments.public_manifest,
        "evidence_gated": arguments.evidence_manifest,
    }
    sources = {
        name: pd.read_csv(path, dtype={"StudyInstanceUID": str})
        for name, path in source_paths.items()
    }
    manifests = prepare_condition_manifests(ids, sources)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        manifest.to_csv(output_dir / f"manifest_{name}.csv", index=False)
    folds, diagnostics = make_repeated_gold_folds(
        manifests["gold_only"],
        n_splits=arguments.n_splits,
        seeds=tuple(arguments.seeds),
        trials=arguments.trials,
    )
    folds.to_csv(output_dir / "repeated_gold_folds.csv", index=False)
    diagnostics.update(
        {
            "pilot_studies": int(len(ids)),
            "gold_studies": int(len(manifests["gold_only"])),
            "pseudo_studies": int(len(ids) - len(manifests["gold_only"])),
            "conditions": list(manifests),
        }
    )
    _write_json(diagnostics, output_dir / "repeated_gold_folds.diagnostics.json")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


def command_extract_frozen_features(arguments: argparse.Namespace) -> None:
    from .feature_oof import extract_frozen_features

    config = _load_runtime_config(arguments)
    manifest = pd.read_csv(arguments.manifest_csv, dtype={"StudyInstanceUID": str})
    if arguments.limit is not None:
        manifest = manifest.head(arguments.limit).copy()
    series = (
        pd.read_csv(arguments.series_csv, dtype={"StudyInstanceUID": str})
        if arguments.series_csv
        else None
    )
    last_report = 0.0

    def report(state: dict[str, Any]) -> None:
        nonlocal last_report
        if state["elapsed_seconds"] - last_report >= 15 or state["position"] == state["total"]:
            rate = state["valid_tokens"] / max(state["elapsed_seconds"], 1e-9)
            print(
                f"[{state['position']}/{state['total']}] {state['uid']} | "
                f"{rate:.1f} tokens/s",
                flush=True,
            )
            last_report = state["elapsed_seconds"]

    summary = extract_frozen_features(
        config,
        manifest,
        series,
        arguments.output_npz,
        device_name=arguments.device,
        progress=report,
    )
    _write_json(summary, arguments.summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_run_feature_oof(arguments: argparse.Namespace) -> None:
    from .feature_oof import (
        evaluate_repeated_oof,
        run_repeated_feature_oof,
    )

    config = load_config(arguments.config)
    manifest_dir = Path(arguments.manifest_dir)
    condition_names = list(arguments.conditions)
    conditions = {
        name: pd.read_csv(
            manifest_dir / f"manifest_{name}.csv", dtype={"StudyInstanceUID": str}
        )
        for name in condition_names
    }
    folds = pd.read_csv(
        manifest_dir / "repeated_gold_folds.csv", dtype={"StudyInstanceUID": str}
    )

    def report(state: dict[str, Any]) -> None:
        print(
            f"[{state['position']}/{state['total']}] {state['condition']} "
            f"repeat={state['repeat']} fold={state['fold']} "
            f"loss={state['final_loss_mean']:.4f}",
            flush=True,
        )

    device_name = arguments.device
    if device_name == "auto":
        import torch

        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    predictions, training = run_repeated_feature_oof(
        config,
        arguments.features_npz,
        conditions,
        folds,
        device_name=device_name,
        progress=report,
    )
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oof_predictions_all_repeats.csv", index=False)
    _write_json(training, output_dir / "training_diagnostics.json")
    # Metrics must be derived from the exact persisted public artifact.  AUC is
    # rank-based, so even a CSV round-trip can reorder nearly tied float values.
    canonical_predictions = pd.read_csv(
        output_dir / "oof_predictions_all_repeats.csv",
        dtype={"StudyInstanceUID": str},
    )
    averaged, metrics = evaluate_repeated_oof(
        canonical_predictions,
        bootstrap_iterations=arguments.bootstrap_iterations,
        seed=arguments.bootstrap_seed,
    )
    averaged.to_csv(output_dir / "oof_predictions_repeat_mean.csv", index=False)
    _write_json(metrics, output_dir / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def command_evaluate_feature_oof(arguments: argparse.Namespace) -> None:
    from .feature_oof import evaluate_repeated_oof

    predictions = pd.read_csv(
        arguments.predictions_csv, dtype={"StudyInstanceUID": str}
    )
    averaged, metrics = evaluate_repeated_oof(
        predictions,
        bootstrap_iterations=arguments.bootstrap_iterations,
        seed=arguments.bootstrap_seed,
    )
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    averaged.to_csv(output_dir / "oof_predictions_repeat_mean.csv", index=False)
    _write_json(metrics, output_dir / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def command_build_centering_manifest(arguments: argparse.Namespace) -> None:
    from .centering_manifest import build_coronal_first_manifest

    manifest = pd.read_csv(arguments.manifest_csv, dtype={"StudyInstanceUID": str})
    series = pd.read_csv(arguments.series_csv, dtype={"StudyInstanceUID": str})
    last_report = 0.0

    def report(state: dict[str, Any]) -> None:
        nonlocal last_report
        if (
            state["elapsed_seconds"] - last_report >= 15
            or state["position"] == state["total"]
        ):
            print(
                f"[{state['position']}/{state['total']}] {state['study_uid']} | "
                f"{state['series']} series",
                flush=True,
            )
            last_report = state["elapsed_seconds"]

    summary = build_coronal_first_manifest(
        manifest,
        series,
        images_root=arguments.images_root,
        output_path=arguments.output_manifest,
        progress=report,
    )
    _write_json(summary, arguments.summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_build_cache(arguments: argparse.Namespace) -> None:
    config = _load_runtime_config(arguments)
    manifest = pd.read_csv(arguments.manifest_csv)
    series = pd.read_csv(arguments.series_csv)
    started = time.monotonic()
    last_report = 0.0

    def report(state: dict[str, int]) -> None:
        nonlocal last_report
        elapsed = time.monotonic() - started
        if elapsed - last_report >= 15:
            print(
                f"[{state['studies_seen']}] studies | "
                f"{state['series_cached']} new series cached | "
                f"{state['missing_slots']} missing slots",
                flush=True,
            )
            last_report = elapsed

    summary = build_cache(
        manifest,
        series,
        config["data"],
        limit=arguments.limit,
        progress=report,
    )
    if arguments.output_json:
        _write_json(summary, arguments.output_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_build_grayscale_cache(arguments: argparse.Namespace) -> None:
    from .grayscale_cache import build_grayscale_cache_part

    config = _load_runtime_config(arguments)
    manifest = pd.read_csv(arguments.manifest_csv, dtype={"StudyInstanceUID": str})
    series = pd.read_csv(arguments.series_csv, dtype={"StudyInstanceUID": str})
    last_report = 0.0

    def report(state: dict[str, Any]) -> None:
        nonlocal last_report
        if (
            state["elapsed_seconds"] - last_report >= 15
            or state["study_position"] == state["study_total"]
        ):
            print(
                f"[{state['study_position']}/{state['study_total']}] "
                f"{state['study_uid']} | {state['valid_slices']} valid slices | "
                f"{state['pixel_failures']} pixel failures",
                flush=True,
            )
            last_report = state["elapsed_seconds"]

    summary = build_grayscale_cache_part(
        manifest,
        series,
        config["data"],
        output_root=arguments.output_root,
        part_index=arguments.part_index,
        num_parts=arguments.num_parts,
        shard_studies=arguments.shard_studies,
        limit=arguments.limit,
        verify_existing_hashes=arguments.verify_existing_hashes,
        progress=report,
    )
    if arguments.summary_json:
        _write_json(summary, arguments.summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_verify_grayscale_cache(arguments: argparse.Namespace) -> None:
    from .grayscale_cache import verify_grayscale_cache

    expected = None
    if arguments.manifest_csv:
        manifest = pd.read_csv(
            arguments.manifest_csv, dtype={"StudyInstanceUID": str}
        )
        expected = manifest["StudyInstanceUID"].astype(str)
    summary = verify_grayscale_cache(
        arguments.cache_roots,
        expected_study_ids=expected,
        full_hash=arguments.full_hash,
    )
    if arguments.summary_json:
        _write_json(summary, arguments.summary_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def apply_config_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (YAML-parsed values) to a config."""

    import copy

    import yaml

    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        dotted, raw_value = item.split("=", 1)
        keys = dotted.split(".")
        cursor: Any = result
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return result


def limit_manifest_per_fold(manifest: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Keep the first ``limit`` rows of every fold (gold rows first) for smokes."""

    if limit is None:
        return manifest
    if limit <= 0:
        raise ValueError("--limit-studies must be positive")
    if "fold" not in manifest:
        return manifest.head(limit).copy()
    gold = manifest.get("is_gold", pd.Series(False, index=manifest.index)).astype(bool)
    ordered = manifest.assign(_gold_first=(~gold).astype(int)).sort_values(
        ["fold", "_gold_first"], kind="stable"
    )
    limited = ordered.groupby("fold", sort=False).head(limit).drop(columns=["_gold_first"])
    return limited.sort_index().reset_index(drop=True)


def command_train(arguments: argparse.Namespace) -> None:
    from .train import train_fold

    config = apply_config_overrides(_load_runtime_config(arguments), arguments.set)
    manifest = limit_manifest_per_fold(
        pd.read_csv(arguments.manifest_csv, dtype={"StudyInstanceUID": str}),
        arguments.limit_studies,
    )
    series = (
        pd.read_csv(arguments.series_csv, dtype={"StudyInstanceUID": str})
        if arguments.series_csv
        else None
    )
    summary = train_fold(
        config,
        manifest,
        series,
        fold=arguments.fold,
        output_dir=arguments.output_dir,
        resume=arguments.resume,
        max_epochs_this_run=arguments.max_epochs_this_run,
        device_name=arguments.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_overfit_smoke(arguments: argparse.Namespace) -> None:
    from .smoke import run_overfit_smoke

    config = _load_runtime_config(arguments)
    manifest = pd.read_csv(arguments.manifest_csv)
    series = (
        pd.read_csv(arguments.series_csv, dtype={"StudyInstanceUID": str})
        if arguments.series_csv
        else None
    )
    summary = run_overfit_smoke(
        config,
        manifest,
        series,
        output_dir=arguments.output_dir,
        max_studies=arguments.max_studies,
        steps=arguments.steps,
        evaluate_every=arguments.evaluate_every,
        learning_rate=arguments.learning_rate,
        device_name=arguments.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_infer(arguments: argparse.Namespace) -> None:
    from .infer import predict_submission

    config = _load_runtime_config(arguments)
    test = pd.read_csv(arguments.test_csv)
    series = (
        pd.read_csv(arguments.series_csv, dtype={"StudyInstanceUID": str})
        if arguments.series_csv
        else None
    )
    submission = predict_submission(
        config,
        test,
        series,
        arguments.checkpoints,
        output_path=arguments.output_csv,
    )
    print(f"Wrote {len(submission)} rows to {arguments.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsna-knee")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata_parser = subparsers.add_parser("audit-metadata")
    metadata_parser.add_argument("--train-csv", required=True)
    metadata_parser.add_argument("--series-csv", required=True)
    metadata_parser.add_argument("--output", required=True)
    metadata_parser.set_defaults(handler=command_audit_metadata)

    reports_parser = subparsers.add_parser("audit-reports")
    reports_parser.add_argument("--train-csv", required=True)
    reports_parser.add_argument("--output-json", required=True)
    reports_parser.add_argument("--duplicates-csv")
    reports_parser.set_defaults(handler=command_audit_reports)

    folds_parser = subparsers.add_parser("make-folds")
    folds_parser.add_argument("--train-csv", required=True)
    folds_parser.add_argument("--pseudo-csv")
    folds_parser.add_argument("--output-csv", required=True)
    folds_parser.add_argument("--diagnostics-json")
    folds_parser.add_argument("--n-splits", type=int, default=5)
    folds_parser.add_argument("--seed", type=int, default=20260828)
    folds_parser.set_defaults(handler=command_make_folds)

    teacher_parser = subparsers.add_parser("report-teacher-pilot")
    teacher_parser.add_argument("--train-csv", required=True)
    teacher_parser.add_argument("--output-dir", required=True)
    teacher_parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    teacher_parser.add_argument("--seed", type=int, default=20260829)
    teacher_parser.set_defaults(handler=command_report_teacher_pilot)

    pseudo_parser = subparsers.add_parser("make-pseudo-labels")
    pseudo_parser.add_argument("--train-csv", required=True)
    pseudo_parser.add_argument("--output-csv", required=True)
    pseudo_parser.add_argument("--diagnostics-json")
    pseudo_parser.add_argument("--errors-csv")
    pseudo_parser.add_argument("--evidence-csv")
    pseudo_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    pseudo_parser.add_argument("--seed", type=int, default=20260829)
    pseudo_parser.set_defaults(handler=command_make_pseudo_labels)

    consensus_parser = subparsers.add_parser("make-consensus-labels")
    consensus_parser.add_argument("--train-csv", required=True)
    consensus_parser.add_argument("--pilkwang-csv", required=True)
    consensus_parser.add_argument("--steven-csv", required=True)
    consensus_parser.add_argument("--lixin-csv", required=True)
    consensus_parser.add_argument("--output-csv", required=True)
    consensus_parser.add_argument("--diagnostics-json")
    consensus_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    consensus_parser.add_argument("--seed", type=int, default=20260829)
    consensus_parser.add_argument(
        "--weight-mode",
        choices=["public_v52", "evidence_gated"],
        default="evidence_gated",
    )
    consensus_parser.set_defaults(handler=command_make_consensus_labels)

    pilot_parser = subparsers.add_parser("select-pilot")
    pilot_parser.add_argument("--manifest-csv", required=True)
    pilot_parser.add_argument(
        "--balance-manifest-csv",
        action="append",
        help="Teacher-neutral balance input in NAME=PATH form; may be repeated",
    )
    pilot_parser.add_argument("--output-csv", required=True)
    pilot_parser.add_argument("--diagnostics-json")
    pilot_parser.add_argument("--pseudo-count", type=int, default=300)
    pilot_parser.add_argument("--minimum-confidence", type=float, default=0.10)
    pilot_parser.add_argument("--seed", type=int, default=20260829)
    pilot_parser.set_defaults(handler=command_select_pilot)

    inventory_parser = subparsers.add_parser("inventory-kaggle-dicoms")
    inventory_parser.add_argument("--ids-csv", required=True)
    inventory_parser.add_argument("--output-csv", required=True)
    inventory_parser.add_argument("--summary-json")
    inventory_parser.add_argument(
        "--competition", default="rsna-knee-abnormality-detection"
    )
    inventory_parser.add_argument("--block-mib", type=int, default=16)
    inventory_parser.add_argument("--cache-blocks", type=int, default=8)
    inventory_parser.set_defaults(handler=command_inventory_kaggle_dicoms)

    download_parser = subparsers.add_parser("download-kaggle-dicoms")
    download_parser.add_argument("--ids-csv", required=True)
    download_parser.add_argument("--destination", required=True)
    download_parser.add_argument("--summary-json", required=True)
    download_parser.add_argument("--inventory-csv")
    download_parser.add_argument(
        "--competition", default="rsna-knee-abnormality-detection"
    )
    download_parser.add_argument("--block-mib", type=int, default=16)
    download_parser.add_argument("--cache-blocks", type=int, default=8)
    download_parser.add_argument("--min-free-gib", type=float, default=50.0)
    download_parser.add_argument(
        "--trust-existing-size",
        action="store_true",
        help="Skip CRC checks for already complete-looking files",
    )
    download_parser.set_defaults(handler=command_download_kaggle_dicoms)

    image_pilot_parser = subparsers.add_parser("prepare-image-pilot")
    image_pilot_parser.add_argument("--ids-csv", required=True)
    image_pilot_parser.add_argument("--rules-manifest", required=True)
    image_pilot_parser.add_argument("--public-manifest", required=True)
    image_pilot_parser.add_argument("--evidence-manifest", required=True)
    image_pilot_parser.add_argument("--output-dir", required=True)
    image_pilot_parser.add_argument("--n-splits", type=int, default=4)
    image_pilot_parser.add_argument(
        "--seeds", nargs="+", type=int, default=[20260829, 20260929, 20261029]
    )
    image_pilot_parser.add_argument("--trials", type=int, default=20000)
    image_pilot_parser.set_defaults(handler=command_prepare_image_pilot)

    feature_parser = subparsers.add_parser("extract-frozen-features")
    feature_parser.add_argument("--config", required=True)
    feature_parser.add_argument("--manifest-csv", required=True)
    feature_parser.add_argument("--series-csv")
    feature_parser.add_argument("--output-npz", required=True)
    feature_parser.add_argument("--summary-json", required=True)
    feature_parser.add_argument("--images-root")
    feature_parser.add_argument("--cache-root")
    feature_parser.add_argument("--cache-roots", nargs="+")
    feature_parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    feature_parser.add_argument("--limit", type=int)
    feature_parser.set_defaults(handler=command_extract_frozen_features)

    feature_oof_parser = subparsers.add_parser("run-feature-oof")
    feature_oof_parser.add_argument("--config", required=True)
    feature_oof_parser.add_argument("--features-npz", required=True)
    feature_oof_parser.add_argument("--manifest-dir", required=True)
    feature_oof_parser.add_argument("--output-dir", required=True)
    feature_oof_parser.add_argument(
        "--conditions",
        nargs="+",
        default=["gold_only", "rule_only", "public_v52", "evidence_gated"],
        choices=["gold_only", "rule_only", "public_v52", "evidence_gated"],
        help="Supervision arms to train; use evidence_gated for the first full run",
    )
    feature_oof_parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    feature_oof_parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    feature_oof_parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    feature_oof_parser.set_defaults(handler=command_run_feature_oof)

    evaluate_oof_parser = subparsers.add_parser("evaluate-feature-oof")
    evaluate_oof_parser.add_argument("--predictions-csv", required=True)
    evaluate_oof_parser.add_argument("--output-dir", required=True)
    evaluate_oof_parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    evaluate_oof_parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    evaluate_oof_parser.set_defaults(handler=command_evaluate_feature_oof)

    centering_manifest_parser = subparsers.add_parser("build-centering-manifest")
    centering_manifest_parser.add_argument("--manifest-csv", required=True)
    centering_manifest_parser.add_argument("--series-csv", required=True)
    centering_manifest_parser.add_argument("--images-root", required=True)
    centering_manifest_parser.add_argument("--output-manifest", required=True)
    centering_manifest_parser.add_argument("--summary-json", required=True)
    centering_manifest_parser.set_defaults(
        handler=command_build_centering_manifest
    )

    cache_parser = subparsers.add_parser("build-cache")
    cache_parser.add_argument("--config", required=True)
    cache_parser.add_argument("--manifest-csv", required=True)
    cache_parser.add_argument("--series-csv", required=True)
    cache_parser.add_argument("--images-root")
    cache_parser.add_argument("--cache-root")
    cache_parser.add_argument("--limit", type=int)
    cache_parser.add_argument("--output-json")
    cache_parser.set_defaults(handler=command_build_cache)

    grayscale_cache_parser = subparsers.add_parser("build-grayscale-cache")
    grayscale_cache_parser.add_argument("--config", required=True)
    grayscale_cache_parser.add_argument("--manifest-csv", required=True)
    grayscale_cache_parser.add_argument("--series-csv", required=True)
    grayscale_cache_parser.add_argument("--images-root")
    grayscale_cache_parser.add_argument("--output-root", required=True)
    grayscale_cache_parser.add_argument("--part-index", type=int, required=True)
    grayscale_cache_parser.add_argument("--num-parts", type=int, default=8)
    grayscale_cache_parser.add_argument("--shard-studies", type=int, default=64)
    grayscale_cache_parser.add_argument("--limit", type=int)
    grayscale_cache_parser.add_argument("--verify-existing-hashes", action="store_true")
    grayscale_cache_parser.add_argument("--summary-json")
    grayscale_cache_parser.set_defaults(handler=command_build_grayscale_cache)

    verify_grayscale_parser = subparsers.add_parser("verify-grayscale-cache")
    verify_grayscale_parser.add_argument("--cache-roots", nargs="+", required=True)
    verify_grayscale_parser.add_argument("--manifest-csv")
    verify_grayscale_parser.add_argument("--full-hash", action="store_true")
    verify_grayscale_parser.add_argument("--summary-json")
    verify_grayscale_parser.set_defaults(handler=command_verify_grayscale_cache)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--manifest-csv", required=True)
    train_parser.add_argument("--series-csv")
    train_parser.add_argument("--fold", type=int, required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--resume", help="same-run last_training.pt; restores a completed epoch")
    train_parser.add_argument("--max-epochs-this-run", type=int, help="stop after N epochs without changing the scheduler horizon")
    train_parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    train_parser.add_argument("--images-root")
    train_parser.add_argument("--cache-root")
    train_parser.add_argument("--cache-roots", nargs="+")
    train_parser.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --set train.epochs=1 --set data.image_size=224",
    )
    train_parser.add_argument(
        "--limit-studies",
        type=int,
        default=None,
        help="smoke: keep only the first N studies of every fold (gold first)",
    )
    train_parser.set_defaults(handler=command_train)

    smoke_parser = subparsers.add_parser("overfit-smoke")
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--manifest-csv", required=True)
    smoke_parser.add_argument("--series-csv", required=True)
    smoke_parser.add_argument("--output-dir", required=True)
    smoke_parser.add_argument("--images-root")
    smoke_parser.add_argument("--cache-root")
    smoke_parser.add_argument("--max-studies", type=int, default=2)
    smoke_parser.add_argument("--steps", type=int, default=20)
    smoke_parser.add_argument("--evaluate-every", type=int, default=2)
    smoke_parser.add_argument("--learning-rate", type=float, default=0.003)
    smoke_parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    smoke_parser.set_defaults(handler=command_overfit_smoke)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--config", required=True)
    infer_parser.add_argument("--test-csv", required=True)
    infer_parser.add_argument("--series-csv")
    infer_parser.add_argument("--checkpoints", nargs="+", required=True)
    infer_parser.add_argument("--output-csv", required=True)
    infer_parser.add_argument("--images-root")
    infer_parser.add_argument("--cache-root")
    infer_parser.add_argument("--cache-roots", nargs="+")
    infer_parser.set_defaults(handler=command_infer)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
