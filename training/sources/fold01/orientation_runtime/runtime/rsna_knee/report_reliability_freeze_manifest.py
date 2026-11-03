"""Deterministic construction of the reliability-v3 pre-Gold freeze manifest.

The builder is intentionally generic: every code and artifact byte is supplied
as an exact :class:`~rsna_knee.report_reliability_inputs.BoundFile` binding.
It refuses unbound current bytes, validates the same proxy/external schemas as
the Gold gate, and verifies the complete candidate manifest before publishing
it.  The Gold label path is copied from the frozen specification but is never
resolved, hashed, stat'ed, or opened here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from .constants import ID_COLUMN
from .report_reliability_gold_gate import (
    EXTERNAL_REQUIRED_COLUMNS,
    FREEZE_RULES,
    PROXY_REQUIRED_COLUMNS,
    GoldGateSpec,
    VerifiedProxyFreeze,
    _assert_bound_artifact_pre_gold_safe,
    _cell_sha,
    _columns_sha,
    _read_csv_allowlist,
    _resolve_inside,
    _validate_external_predictions,
    _validate_proxy_oof,
    verify_proxy_freeze_manifest,
)
from .report_reliability_inputs import BoundFile, sha256_file


BINDING_PLAN_SCHEMA_VERSION = "1.0.0"
DEFAULT_BINDING_PLAN_FILENAME = "PROXY_FREEZE_BINDING_PLAN.json"


@dataclass(frozen=True)
class FreezeManifestBuildResult:
    path: Path
    sha256: str
    verified: VerifiedProxyFreeze


@dataclass(frozen=True)
class BindingPlanBuildResult:
    path: Path
    sha256: str


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _line_sha(values: Sequence[object]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _validate_created_at(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("created_at must be a nonempty exact string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be a timezone-aware ISO-8601 value") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include an explicit timezone")
    return value


def _binding_payload(
    root: Path,
    binding: BoundFile,
    *,
    name: str,
    boundary: Path | None = None,
) -> tuple[dict[str, str], Path]:
    raw = Path(binding.path)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{name}: binding path must be workspace-relative")
    path = _resolve_inside(root, binding.path, boundary=boundary)
    if not path.is_file():
        raise FileNotFoundError(f"{name}: missing {path}")
    digest = sha256_file(path)
    if digest != binding.sha256:
        raise ValueError(f"{name}: SHA-256 mismatch")
    return {"path": raw.as_posix(), "sha256": binding.sha256}, path


def _strict_binding_map(
    values: Mapping[str, BoundFile], expected_roles: Sequence[str], *, name: str
) -> None:
    if set(values) != set(expected_roles):
        raise ValueError(
            f"{name}: exact roles required; "
            f"missing={sorted(set(expected_roles).difference(values))}, "
            f"unexpected={sorted(set(values).difference(expected_roles))}"
        )
    for role, value in values.items():
        if not isinstance(value, BoundFile):
            raise TypeError(f"{name}.{role} must be a BoundFile")
        if len(value.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in value.sha256
        ):
            raise ValueError(f"{name}.{role} has an invalid SHA-256")


def _identity_payload(
    proxy: pd.DataFrame,
    external: pd.DataFrame,
    identity_frame: pd.DataFrame,
    *,
    spec: GoldGateSpec,
) -> dict[str, Any]:
    if identity_frame[ID_COLUMN].isna().any() or identity_frame[ID_COLUMN].duplicated().any():
        raise ValueError("canonical identity manifest has null/duplicate UIDs")
    identity_frame = identity_frame.copy()
    identity_frame[ID_COLUMN] = identity_frame[ID_COLUMN].astype(str)
    if identity_frame["is_gold"].isna().any():
        raise ValueError("canonical identity manifest has null is_gold values")
    if pd.api.types.is_bool_dtype(identity_frame["is_gold"]):
        gold_mask = identity_frame["is_gold"].astype(bool)
    else:
        normalized = identity_frame["is_gold"].astype(str).str.strip().str.casefold()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError("canonical identity manifest has invalid is_gold values")
        gold_mask = normalized.eq("true")

    proxy_uids = sorted(proxy[ID_COLUMN].astype(str).unique())
    proxy_groups = sorted(proxy["report_group"].astype(str).unique())
    external_uids = sorted(external[ID_COLUMN].astype(str).unique())
    if set(proxy_uids).intersection(external_uids):
        raise ValueError("proxy and external UID cohorts overlap")
    gold_identity = set(identity_frame.loc[gold_mask, ID_COLUMN])
    external_set = set(external_uids)
    if not gold_identity.issubset(external_set):
        raise ValueError("external predictions do not cover all frozen Gold identities")
    external_gold = len(external_set.intersection(gold_identity))
    external_non_gold = len(external_set.difference(gold_identity))
    if (external_gold, external_non_gold) != (
        spec.expected_external_gold_identities,
        spec.expected_external_non_gold,
    ):
        raise ValueError("external Gold/non-Gold identity coverage drift")
    return {
        "eligible_studies": spec.expected_proxy_studies,
        "eligible_report_groups": spec.expected_proxy_report_groups,
        "proxy_decisive_cells": spec.expected_proxy_rows,
        "eligible_uid_sha256": _line_sha(proxy_uids),
        "eligible_report_group_sha256": _line_sha(proxy_groups),
        "eligible_cell_sha256": _cell_sha(proxy),
        "external_studies": spec.expected_external_studies,
        "external_cells": spec.expected_external_rows,
        "external_uid_sha256": _line_sha(external_uids),
        "external_cell_sha256": _cell_sha(external),
        "external_gold_identity_studies": external_gold,
        "external_non_gold_studies": external_non_gold,
    }


def build_proxy_freeze_manifest_payload(
    root: str | Path,
    *,
    spec: GoldGateSpec,
    created_at: str,
    feature_sets: Mapping[str, Any],
    code_bindings: Mapping[str, BoundFile],
    artifact_bindings: Mapping[str, BoundFile],
) -> dict[str, Any]:
    """Build and fully validate a deterministic pre-Gold manifest payload.

    This function does not write files.  Its only identity read is the frozen
    ``is_gold`` routing flag in the canonical identity manifest; it never reads
    the Gold label source.
    """

    root_path = Path(root).resolve()
    run_dir = _resolve_inside(root_path, spec.run_directory)
    created_at = _validate_created_at(created_at)
    _strict_binding_map(code_bindings, spec.required_code_roles, name="code bindings")
    _strict_binding_map(
        artifact_bindings, spec.required_artifact_roles, name="artifact bindings"
    )
    if _canonical_json_sha256(feature_sets) != spec.feature_sets_sha256:
        raise ValueError("feature-set payload does not match the frozen SHA-256")

    contract_binding, _ = _binding_payload(
        root_path, spec.contract, name="contract"
    )
    identity_binding, identity_path = _binding_payload(
        root_path,
        spec.canonical_identity_manifest,
        name="canonical identity manifest",
    )
    if identity_binding["sha256"] != spec.expected_source_hashes.get("manifest"):
        raise ValueError("canonical identity manifest/source hash mismatch")
    fold_binding, _ = _binding_payload(root_path, spec.fold_map, name="fold map")
    if set(spec.source_files) != set(spec.expected_source_hashes):
        raise ValueError("Gold gate source-file bindings are incomplete")
    for role, source in sorted(spec.source_files.items()):
        if source.sha256 != spec.expected_source_hashes[role]:
            raise ValueError(f"source binding/hash mismatch for {role}")
        _binding_payload(root_path, source, name=f"source {role}")

    code_payload: dict[str, dict[str, str]] = {}
    for role in sorted(code_bindings):
        code_payload[role], _ = _binding_payload(
            root_path, code_bindings[role], name=f"code {role}"
        )

    artifact_payload: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for role in sorted(artifact_bindings):
        binding, path = _binding_payload(
            root_path,
            artifact_bindings[role],
            name=f"artifact {role}",
            boundary=run_dir,
        )
        if path in seen_paths:
            raise ValueError("two artifact roles resolve to the same file")
        seen_paths.add(path)
        _assert_bound_artifact_pre_gold_safe(path, role=role)
        artifact_payload[role] = dict(binding)
        artifact_paths[role] = path

    proxy = _validate_proxy_oof(
        _read_csv_allowlist(
            artifact_paths["proxy_oof_scores"],
            PROXY_REQUIRED_COLUMNS,
            name="proxy_oof_scores",
        ),
        spec,
    )
    external = _validate_external_predictions(
        _read_csv_allowlist(
            artifact_paths["final_external_predictions"],
            EXTERNAL_REQUIRED_COLUMNS,
            name="final_external_predictions",
        ),
        spec,
    )
    for role, frame in (
        ("proxy_oof_scores", proxy),
        ("final_external_predictions", external),
    ):
        artifact_payload[role]["rows"] = int(len(frame))
        artifact_payload[role]["columns_sha256"] = _columns_sha(frame.columns)

    identity_frame = pd.read_csv(identity_path, usecols=[ID_COLUMN, "is_gold"])
    identity = _identity_payload(proxy, external, identity_frame, spec=spec)
    payload: dict[str, Any] = {
        "schema_version": spec.schema_version,
        "status": "FROZEN_PRE_GOLD_COMPLETE",
        "created_at": created_at,
        "contract": {
            **contract_binding,
            "version": spec.contract_version,
            "semantic_hash": spec.contract_semantic_hash,
        },
        "source_hashes": dict(spec.expected_source_hashes),
        "code_hashes": code_payload,
        "feature_sets": dict(feature_sets),
        "feature_sets_sha256": spec.feature_sets_sha256,
        "fold_map": fold_binding,
        "identity": identity,
        "artifacts": artifact_payload,
        "frozen_rules": json.loads(json.dumps(FREEZE_RULES)),
        # Copy only the frozen path/hash constant.  Deliberately do not resolve it.
        "gold_source": {
            "path": spec.gold_source.path,
            "sha256": spec.gold_source.sha256,
        },
        "gold_outcomes_loaded": False,
    }
    return payload


def write_proxy_freeze_manifest(
    root: str | Path,
    output_path: str | Path,
    *,
    spec: GoldGateSpec,
    created_at: str,
    feature_sets: Mapping[str, Any],
    code_bindings: Mapping[str, BoundFile],
    artifact_bindings: Mapping[str, BoundFile],
) -> FreezeManifestBuildResult:
    """Verify a candidate, then atomically create one immutable manifest file."""

    root_path = Path(root).resolve()
    run_dir = _resolve_inside(root_path, spec.run_directory)
    output = _resolve_inside(root_path, str(output_path), boundary=run_dir)
    if output.name != "PROXY_FREEZE_MANIFEST.json":
        raise ValueError("output filename must be PROXY_FREEZE_MANIFEST.json")
    if output.exists():
        raise FileExistsError("PROXY_FREEZE_MANIFEST.json already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_proxy_freeze_manifest_payload(
        root_path,
        spec=spec,
        created_at=created_at,
        feature_sets=feature_sets,
        code_bindings=code_bindings,
        artifact_bindings=artifact_bindings,
    )
    encoded = _canonical_manifest_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()

    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=".PROXY_FREEZE_MANIFEST.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        verified_candidate = verify_proxy_freeze_manifest(
            root_path,
            temporary,
            expected_manifest_sha256=digest,
            spec=spec,
        )
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                "PROXY_FREEZE_MANIFEST.json appeared concurrently; refusing overwrite"
            ) from error
        verified_final = verify_proxy_freeze_manifest(
            root_path,
            output,
            expected_manifest_sha256=digest,
            spec=spec,
        )
        if verified_candidate.manifest_sha256 != verified_final.manifest_sha256:
            raise AssertionError("published manifest differs from verified candidate")
    finally:
        temporary.unlink(missing_ok=True)
    return FreezeManifestBuildResult(
        path=output,
        sha256=digest,
        verified=verified_final,
    )


def load_binding_plan(
    path: str | Path,
) -> tuple[str, dict[str, BoundFile], dict[str, BoundFile]]:
    """Load an exact, reviewable binding plan for the CLI wrapper."""

    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("binding plan is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "created_at",
        "code_bindings",
        "artifact_bindings",
    }:
        raise ValueError("binding plan has unexpected/missing top-level keys")
    if payload["schema_version"] != BINDING_PLAN_SCHEMA_VERSION:
        raise ValueError("binding plan schema version drift")
    created_at = _validate_created_at(payload["created_at"])

    def parse_bindings(value: Any, *, name: str) -> dict[str, BoundFile]:
        if not isinstance(value, dict):
            raise ValueError(f"binding plan {name} must be an object")
        result: dict[str, BoundFile] = {}
        for role, binding in value.items():
            if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                raise ValueError(f"binding plan {name}.{role} must contain path/sha256")
            result[str(role)] = BoundFile(
                path=str(binding["path"]), sha256=str(binding["sha256"])
            )
        return result

    return (
        created_at,
        parse_bindings(payload["code_bindings"], name="code_bindings"),
        parse_bindings(payload["artifact_bindings"], name="artifact_bindings"),
    )


def build_current_binding_plan_payload(
    root: str | Path,
    *,
    spec: GoldGateSpec,
    created_at: str,
) -> dict[str, Any]:
    """Hash-bind every exact current pre-Gold code/artifact byte.

    The function only touches the paths enumerated by the frozen gate spec.  In
    particular, it never resolves, stats, hashes, or opens ``gold_source``.
    """

    root_path = Path(root).resolve()
    run_dir = _resolve_inside(root_path, spec.run_directory)
    created_at = _validate_created_at(created_at)
    if set(spec.required_code_paths) != set(spec.required_code_roles):
        raise ValueError("binding-plan code path map is incomplete")
    if set(spec.required_artifact_filenames) != set(spec.required_artifact_roles):
        raise ValueError("binding-plan artifact filename map is incomplete")

    code_bindings: dict[str, dict[str, str]] = {}
    for role in spec.required_code_roles:
        relative = spec.required_code_paths[role]
        path = _resolve_inside(root_path, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        code_bindings[role] = {"path": relative, "sha256": sha256_file(path)}

    artifact_bindings: dict[str, dict[str, str]] = {}
    run_relative = Path(spec.run_directory)
    for role in spec.required_artifact_roles:
        relative = (run_relative / spec.required_artifact_filenames[role]).as_posix()
        path = _resolve_inside(root_path, relative, boundary=run_dir)
        if not path.is_file():
            raise FileNotFoundError(path)
        _assert_bound_artifact_pre_gold_safe(path, role=role)
        artifact_bindings[role] = {
            "path": relative,
            "sha256": sha256_file(path),
        }

    return {
        "schema_version": BINDING_PLAN_SCHEMA_VERSION,
        "created_at": created_at,
        "code_bindings": code_bindings,
        "artifact_bindings": artifact_bindings,
    }


def write_current_binding_plan(
    root: str | Path,
    output_path: str | Path,
    *,
    spec: GoldGateSpec,
    created_at: str,
) -> BindingPlanBuildResult:
    """Atomically create one immutable, reviewable binding plan."""

    root_path = Path(root).resolve()
    run_dir = _resolve_inside(root_path, spec.run_directory)
    output = _resolve_inside(root_path, str(output_path), boundary=run_dir)
    if output.name != DEFAULT_BINDING_PLAN_FILENAME:
        raise ValueError(
            f"binding-plan filename must be {DEFAULT_BINDING_PLAN_FILENAME}"
        )
    if output.exists():
        raise FileExistsError(f"{DEFAULT_BINDING_PLAN_FILENAME} already exists")
    payload = build_current_binding_plan_payload(
        root_path, spec=spec, created_at=created_at
    )
    encoded = _canonical_manifest_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{DEFAULT_BINDING_PLAN_FILENAME}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        loaded_created_at, loaded_code, loaded_artifacts = load_binding_plan(temporary)
        if (
            loaded_created_at != created_at
            or set(loaded_code) != set(spec.required_code_roles)
            or set(loaded_artifacts) != set(spec.required_artifact_roles)
        ):
            raise AssertionError("binding-plan readback differs from frozen spec")
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"{DEFAULT_BINDING_PLAN_FILENAME} appeared concurrently"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(output) != digest:
        raise AssertionError("binding-plan final readback SHA drift")
    return BindingPlanBuildResult(path=output, sha256=digest)


__all__ = [
    "BINDING_PLAN_SCHEMA_VERSION",
    "DEFAULT_BINDING_PLAN_FILENAME",
    "BindingPlanBuildResult",
    "FreezeManifestBuildResult",
    "build_current_binding_plan_payload",
    "build_proxy_freeze_manifest_payload",
    "load_binding_plan",
    "write_current_binding_plan",
    "write_proxy_freeze_manifest",
]
