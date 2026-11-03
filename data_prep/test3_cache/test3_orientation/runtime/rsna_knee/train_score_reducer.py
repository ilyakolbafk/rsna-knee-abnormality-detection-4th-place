"""Fail-closed reducer for the frozen RSNA Knee train-score artifacts.

The reducer intentionally keeps raw prediction lineage separate from the two
rank fusions it can reproduce.  In particular, its final V4.5-like column is a
directional train-cohort diagnostic, not an OOF estimate or a score claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .constants import ID_COLUMN, TARGET_COLUMNS
from .source_clusters import write_deterministic_npz


PUBLIC_OOF_PANEL_SHA256 = (
    "2e3f69b34096fbb6459489f52036fa7afc9472a26219ea84a86950c5709d5b90"
)
PUBLIC_OOF_AUDIT_SHA256 = (
    "08deb686b697009a61544832f2c0fc30988ca00b67f8c46a9f846407a2860382"
)
COATNET_V4_NPZ_SHA256 = (
    "e817865b369a990b00c849f62f41c89c1b044abc4457cc1304a8c6af4eafd434"
)
COATNET_V4_AUDIT_SHA256 = (
    "51fbe8b916009991d24f3b187fddc07d1683bc7cc2fa3059d435fb7001a08af6"
)
RSNA_BASE_SOURCE_SHA256 = (
    "f2605f9e1287a3357a42c0875efa34552ac7a4d24e65e20192c086a08928d30a"
)
V14_NOTEBOOK_SHA256 = (
    "70fc2867effecf8d2566e888a4732e42116b3cb63b3b5f560805870ee1e5f7de"
)
COATNET_V4_CHECKPOINT_SHA256 = (
    "89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded"
)
COATNET_V5_CHECKPOINT_SHA256 = (
    "20d45e0732a0e9120025d0172a4d2dd0170900bc250b06742dfb7ae08a63786a"
)
DINO_CHECKPOINT_SHA256 = {
    0: "b447a665a92ed65e0f4cacf6800940ba7a6d585d0c36154ce18dee60ec48d5cd",
    1: "30bbe5f9afa24af57fd69bad293a4b9fe3ca2e1d737381bc06e469c5ed380e76",
    2: "5bdc4f72471e7c5db01f2f919ba39212087cbb5f2f76b619e4ba38446800b782",
    3: "f757ddc9531f903c6cd0729cadc7051fcdca03b281c89c39cc7957ab44fc2243",
    4: "066490a5c17b3887d1bfda1427f45b14cab416e7b1dcc16afc9a003cdcc08aad",
}

# Frozen V14/V4.5 no-Swin CoAtNet complement.
V45_COMPLEMENT_WEIGHT = {
    "ACL": 0.025,
    "MCL": 0.16,
    "Medial Meniscus": 0.025,
    "Lateral Meniscus": 0.020,
    "Medial OA": 0.10,
    "Lateral OA": 0.025,
    "PF OA": 0.12,
    "Effusion": 0.10,
    "Synovitis": 0.16,
    "Baker's": 0.12,
    "Contusion": 0.12,
    "Fracture": 0.020,
}

# V14 final-fusion weights when the V18 calibrator flag is true.  The exact
# calibrator itself is not reconstructed by this reducer; see the scope label.
V45_FINAL_COATNET_WEIGHT = {
    "ACL": 0.54,
    "MCL": 0.50,
    "Medial Meniscus": 0.565,
    "Lateral Meniscus": 0.625,
    "Medial OA": 0.50,
    "Lateral OA": 0.56,
    "PF OA": 0.50,
    "Effusion": 0.50,
    "Synovitis": 0.50,
    "Baker's": 0.50,
    "Contusion": 0.50,
    "Fracture": 0.625,
}
V45_BASE_0935_WEIGHT = {
    **{target: 0.50 for target in TARGET_COLUMNS},
    "Medial Meniscus": 0.52,
    "Lateral Meniscus": 0.54,
    "Fracture": 0.54,
}

DINO_SCOPE = "train_inference_not_oof"
DINO_RANK_MEMBER_COUNT = 5
DINO_TIE_MILP_TIME_LIMIT_SECONDS = 300.0
V5_RAW_SCOPE = "train_inference_full_model_v5_not_oof"
V5_FROZEN_SLOTS = (
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12),
)
V5_FROZEN_SLOT_OFFSETS = (0, 18, 32, 44, 52)
V5_FROZEN_SLOTS_COMPONENT_SHA256 = (
    "b09837f6d04e5c0147fba938182f33c3ee4fab90de8d935c4b02db4c3eb67234"
)
V5_EXPECTED_KAGGLE_VERSION = {0: 2, 1: 2, 2: 1}
V5_EXPECTED_ARTIFACT_REVISION = {
    0: "preserve-source-no-grad-decorator-v2",
    1: "attributed-source-native-pixel-skip-v3",
    2: "attributed-source-native-pixel-skip-v4-exact-slot-selector",
}
V5_S01_ATTRIBUTION_CORRECTION_SHA256 = (
    "9056c3eebc605a805607b4d9e8c40329949587576bb6865177e0b19c98cd2bc7"
)
V5_S01_ATTRIBUTION_CORRECTION_BINDING = {
    "shard_index": 1,
    "correction_sha256": V5_S01_ATTRIBUTION_CORRECTION_SHA256,
    "source_slug": "ilyakolb/rsna-knee-train-scores-coatnet-v5-s01-v1",
    "source_version": 2,
    "source_directory": (
        ".codex/runs/rsna-knee-full-train-inference/downloads/coatnet_v5_s01_v2"
    ),
    "source_file_sha256": {
        "RUN_AUDIT.json": (
            "12c04e24ed7fd3e423fbfeac6de52b93ea5c00016d2d0ddccc00f76c82d264a4"
        ),
        "coatnet_v5_train_raw_s01.npz": (
            "e5edaa9044e7a43fceb3539df6532c9bcdb9da35a749b36fc3b3763201318be1"
        ),
        "coatnet_v5_train_raw_s01.csv": (
            "21d24815f995b9901b1bd65bbf214190f7e50b7562b435b3e25884431ed02de0"
        ),
        "rsna-knee-train-scores-coatnet-v5-s01-v1.log": (
            "a42dd760f8c2e064400864ca4cf54eef88f3709047220679c7d404ac47bd1cd6"
        ),
    },
    "frozen_slots_component_sha256": V5_FROZEN_SLOTS_COMPONENT_SHA256,
    "effective_run_audit_canonical_sha256": (
        "48cac77f466d9d91cb3f8d03b51db1e8e808be686cc2a6404c80e5df1903bbc3"
    ),
}
DIRECTIONAL_SCOPE = (
    "directional_mixed_oof_and_train_inference_not_oof_not_performance_estimate"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid_sequence_sha256(ids: Sequence[str]) -> str:
    """Hash an ordered UID sequence without relying on NumPy string bytes."""

    return hashlib.sha256(("\n".join(map(str, ids)) + "\n").encode()).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rankpct(values: np.ndarray) -> np.ndarray:
    """Match the frozen CoAtNet/DINO double-argsort rank recipe."""

    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("rankpct requires a finite two-dimensional matrix")
    order = values.argsort(axis=0).argsort(axis=0).astype(np.float64)
    return order / max(values.shape[0] - 1, 1)


def average_rankpct(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("average_rankpct requires a finite two-dimensional matrix")
    return pd.DataFrame(values).rank(method="average", pct=True).to_numpy(np.float64)


def validate_dinov3_rank_tie_equivalence(
    probability: np.ndarray,
    rank_mean: np.ndarray,
    targets: Sequence[str],
    *,
    time_limit_seconds: float = DINO_TIE_MILP_TIME_LIMIT_SECONDS,
) -> dict[str, Any]:
    """Validate an exact double-argsort rank realization across float32 ties.

    NumPy's default argsort is allowed to permute exact-equal values differently
    across versions.  The stored mean rank is therefore valid when there is a
    joint assignment of each tie group's consecutive ordinal ranks that
    reproduces every stored five-member aggregate.  This gate encodes that
    question as one binary HiGHS feasibility problem per target and checks the
    rounded solver result against the integer equalities exactly.
    """

    probability = np.asarray(probability)
    rank_mean = np.asarray(rank_mean)
    target_names = tuple(map(str, targets))
    if probability.ndim != 3:
        raise RuntimeError("DINOv3 tie gate requires member x study x target ranks")
    member_count, study_count, target_count = probability.shape
    if member_count != DINO_RANK_MEMBER_COUNT:
        raise RuntimeError(
            f"DINOv3 tie gate requires {DINO_RANK_MEMBER_COUNT} members, "
            f"got {member_count}"
        )
    if study_count < 2:
        raise RuntimeError("DINOv3 tie gate requires at least two studies")
    if rank_mean.shape != (study_count, target_count):
        raise RuntimeError("DINOv3 tie gate rank shape drift")
    if len(target_names) != target_count or len(set(target_names)) != target_count:
        raise RuntimeError("DINOv3 tie gate target identity drift")
    if probability.dtype != np.dtype(np.float32):
        raise RuntimeError("DINOv3 tie gate requires exact float32 probabilities")
    if rank_mean.dtype != np.dtype(np.float32):
        raise RuntimeError("DINOv3 tie gate requires exact float32 stored ranks")
    if not np.isfinite(probability).all() or not np.isfinite(rank_mean).all():
        raise RuntimeError("DINOv3 tie gate forbids non-finite values")
    if (
        float(probability.min()) < 0.0
        or float(probability.max()) > 1.0
        or float(rank_mean.min()) < 0.0
        or float(rank_mean.max()) > 1.0
    ):
        raise RuntimeError("DINOv3 tie gate values are outside [0,1]")
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise RuntimeError("DINOv3 tie gate requires a positive finite time limit")

    aggregate_denominator = member_count * (study_count - 1)
    scaled_rank = rank_mean.astype(np.float64) * aggregate_denominator
    aggregate_ordinals = np.rint(scaled_rank).astype(np.int64)
    reencoded_rank = (
        aggregate_ordinals.astype(np.float64) / aggregate_denominator
    ).astype(np.float32)
    if not np.array_equal(reencoded_rank, rank_mean):
        raise RuntimeError(
            "DINOv3 stored rank mean lacks exact integer aggregate encoding"
        )
    if np.any(aggregate_ordinals < 0) or np.any(
        aggregate_ordinals > aggregate_denominator
    ):
        raise RuntimeError("DINOv3 aggregate ordinal encoding is out of range")

    expected_target_total = member_count * study_count * (study_count - 1) // 2
    target_totals = aggregate_ordinals.sum(axis=0, dtype=np.int64)
    if not np.array_equal(
        target_totals,
        np.full(target_count, expected_target_total, dtype=np.int64),
    ):
        raise RuntimeError("DINOv3 aggregate ordinal target total drift")

    target_diagnostics: dict[str, dict[str, Any]] = {}
    for target_index, target in enumerate(target_names):
        base_ordinal = np.empty((member_count, study_count), dtype=np.int64)
        tie_groups: list[tuple[np.ndarray, int, int]] = []
        tie_participants = np.zeros(study_count, dtype=bool)
        maximum_group_size = 1
        variable_count = 0

        for member_index in range(member_count):
            _, inverse, counts = np.unique(
                probability[member_index, :, target_index],
                return_inverse=True,
                return_counts=True,
            )
            starts = np.empty(len(counts), dtype=np.int64)
            starts[0] = 0
            if len(counts) > 1:
                starts[1:] = np.cumsum(counts[:-1], dtype=np.int64)
            base_ordinal[member_index] = starts[inverse]
            for group_index in np.flatnonzero(counts > 1):
                studies = np.flatnonzero(inverse == group_index).astype(
                    np.int64, copy=False
                )
                group_size = int(counts[group_index])
                tie_groups.append((studies, group_size, variable_count))
                tie_participants[studies] = True
                maximum_group_size = max(maximum_group_size, group_size)
                variable_count += group_size * group_size

        residual = (
            aggregate_ordinals[:, target_index]
            - base_ordinal.sum(axis=0, dtype=np.int64)
        )
        if np.any(residual[~tie_participants] != 0):
            raise RuntimeError(
                f"DINOv3 stored rank mean violates fixed ordinals for target {target}"
            )

        participant_studies = np.flatnonzero(tie_participants).astype(
            np.int64, copy=False
        )
        if variable_count == 0:
            target_diagnostics[target] = {
                "status": 0,
                "solver_status": "DETERMINISTIC_NO_TIES",
                "tie_groups": 0,
                "tied_rows": 0,
                "max_group": 1,
                "binary_variables": 0,
                "constraints": 0,
                "exact_integer_readback": True,
            }
            continue

        tie_assignment_constraint_count = 2 * sum(
            group_size for _, group_size, _ in tie_groups
        )
        constraint_count = tie_assignment_constraint_count + len(
            participant_studies
        )
        right_hand_side = np.empty(constraint_count, dtype=np.int64)
        right_hand_side[:tie_assignment_constraint_count] = 1
        right_hand_side[tie_assignment_constraint_count:] = residual[
            participant_studies
        ]
        participant_rows = np.full(study_count, -1, dtype=np.int64)
        participant_rows[participant_studies] = np.arange(
            tie_assignment_constraint_count,
            constraint_count,
            dtype=np.int64,
        )

        constraint_rows: list[int] = []
        variable_columns: list[int] = []
        integer_coefficients: list[int] = []
        constraint_cursor = 0
        for studies, group_size, variable_start in tie_groups:
            for local_row, study_index in enumerate(studies):
                for ordinal_offset in range(group_size):
                    variable_index = (
                        variable_start + local_row * group_size + ordinal_offset
                    )
                    constraint_rows.append(constraint_cursor)
                    variable_columns.append(variable_index)
                    integer_coefficients.append(1)
                    if ordinal_offset:
                        constraint_rows.append(
                            int(participant_rows[int(study_index)])
                        )
                        variable_columns.append(variable_index)
                        integer_coefficients.append(ordinal_offset)
                constraint_cursor += 1
            for ordinal_offset in range(group_size):
                for local_row in range(group_size):
                    constraint_rows.append(constraint_cursor)
                    variable_columns.append(
                        variable_start + local_row * group_size + ordinal_offset
                    )
                    integer_coefficients.append(1)
                constraint_cursor += 1
        if constraint_cursor != tie_assignment_constraint_count:
            raise RuntimeError(
                f"DINOv3 tie gate internal constraint drift for target {target}"
            )

        row_index = np.asarray(constraint_rows, dtype=np.int64)
        column_index = np.asarray(variable_columns, dtype=np.int64)
        coefficient = np.asarray(integer_coefficients, dtype=np.int64)
        equality_matrix = coo_array(
            (
                coefficient.astype(np.float64),
                (row_index, column_index),
            ),
            shape=(constraint_count, variable_count),
        ).tocsc()
        try:
            result = milp(
                c=np.zeros(variable_count, dtype=np.float64),
                integrality=np.ones(variable_count, dtype=np.uint8),
                bounds=Bounds(0.0, 1.0),
                constraints=LinearConstraint(
                    equality_matrix,
                    right_hand_side.astype(np.float64),
                    right_hand_side.astype(np.float64),
                ),
                options={
                    "disp": False,
                    "presolve": True,
                    "time_limit": float(time_limit_seconds),
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"DINOv3 tie-assignment MILP errored for target {target}: {exc}"
            ) from exc
        if result.status != 0 or not result.success or result.x is None:
            raise RuntimeError(
                f"DINOv3 tie-assignment MILP failed for target {target}: "
                f"status={result.status}, message={result.message}"
            )

        solution = np.asarray(result.x, dtype=np.float64)
        if solution.shape != (variable_count,) or not np.isfinite(solution).all():
            raise RuntimeError(
                f"DINOv3 tie-assignment MILP returned invalid values for target {target}"
            )
        rounded_solution = np.rint(solution)
        if (
            np.any((rounded_solution != 0.0) & (rounded_solution != 1.0))
            or float(np.max(np.abs(solution - rounded_solution))) > 1e-6
        ):
            raise RuntimeError(
                f"DINOv3 tie-assignment MILP returned non-integral values for target {target}"
            )
        binary_solution = rounded_solution.astype(np.int64)
        exact_readback = np.zeros(constraint_count, dtype=np.int64)
        np.add.at(
            exact_readback,
            row_index,
            coefficient * binary_solution[column_index],
        )
        if not np.array_equal(exact_readback, right_hand_side):
            raise RuntimeError(
                f"DINOv3 tie-assignment exact integer readback failed for target {target}"
            )

        target_diagnostics[target] = {
            "status": int(result.status),
            "solver_status": "OPTIMAL_FEASIBLE",
            "tie_groups": len(tie_groups),
            "tied_rows": int(tie_participants.sum()),
            "max_group": maximum_group_size,
            "binary_variables": variable_count,
            "constraints": constraint_count,
            "exact_integer_readback": True,
        }

    return {
        "status": "PASS_TIE_EQUIVALENT",
        "method": (
            "binary assignment of consecutive ordinal ranks within every exact-value "
            "float32 tie group, joined by each study's five-member aggregate integer "
            "rank constraint"
        ),
        "solver": "scipy.optimize.milp / HiGHS",
        "member_count": member_count,
        "study_count": study_count,
        "target_count": target_count,
        "aggregate_ordinal_denominator": aggregate_denominator,
        "stored_rank_integer_encoding_gate": True,
        "expected_per_target_aggregate_ordinal_total": expected_target_total,
        "per_target_aggregate_ordinal_totals": target_totals.tolist(),
        "conserved_target_totals_gate": True,
        "all_targets_feasible": True,
        "exact_integer_readback_gate": True,
        "targets": target_diagnostics,
    }


def _load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"{path}: missing NPZ fields {sorted(missing)}")
        return {name: archive[name].copy() for name in archive.files}


def _load_audit(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError(f"audit hash drift: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("success") is not True:
        raise RuntimeError(f"remote audit is not successful: {path}")
    return audit


def _verify_audited_artifact(audit: Mapping[str, Any], path: Path) -> str:
    artifacts = audit.get("artifacts")
    if not isinstance(artifacts, Mapping) or path.name not in artifacts:
        raise RuntimeError(f"{path}: RUN_AUDIT lacks an exact artifact hash")
    record = artifacts[path.name]
    expected = str(record.get("sha256")) if isinstance(record, Mapping) else str(record)
    if len(expected) != 64:
        raise RuntimeError(f"{path}: RUN_AUDIT artifact hash is malformed")
    actual = sha256_file(path)
    if expected != actual:
        raise RuntimeError(f"artifact hash drift: {path}: {actual} != {expected}")
    return actual


def _require_probability(name: str, values: np.ndarray, shape: tuple[int, ...]) -> None:
    if values.shape != shape:
        raise RuntimeError(f"{name} shape drift: {values.shape} != {shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains non-finite values")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise RuntimeError(f"{name} contains values outside [0,1]")


@dataclass(frozen=True)
class ReducerSpec:
    expected_studies: int = 4407
    expected_gold: int = 58
    targets: tuple[str, ...] = tuple(TARGET_COLUMNS)
    v5_shard_count: int = 3
    public_oof_panel_sha256: str = PUBLIC_OOF_PANEL_SHA256
    public_oof_audit_sha256: str = PUBLIC_OOF_AUDIT_SHA256
    coatnet_v4_npz_sha256: str = COATNET_V4_NPZ_SHA256
    coatnet_v4_audit_sha256: str = COATNET_V4_AUDIT_SHA256
    dinov3_source_sha256: str = RSNA_BASE_SOURCE_SHA256
    coatnet_v5_source_sha256: str = V14_NOTEBOOK_SHA256
    coatnet_v4_checkpoint_sha256: str = COATNET_V4_CHECKPOINT_SHA256
    coatnet_v5_checkpoint_sha256: str = COATNET_V5_CHECKPOINT_SHA256
    v5_max_dicom_read_failures_full_cohort: int = 8
    v5_max_dicom_read_failures_per_shard: int = 8
    v5_max_dicom_read_failures_per_study: int = 2
    v5_max_affected_studies_per_shard: int = 4
    v5_max_dicom_read_failure_fraction_per_shard: float = 0.0001
    v5_min_selected_series_per_study: int = 3
    v5_min_valid_slices_per_study: int = 42
    dino_checkpoint_sha256: Mapping[int, str] = field(
        default_factory=lambda: dict(DINO_CHECKPOINT_SHA256)
    )


def load_public_oof_panel(
    path: Path, audit_path: Path, spec: ReducerSpec
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if sha256_file(path) != spec.public_oof_panel_sha256:
        raise RuntimeError(f"public OOF panel hash drift: {path}")
    audit = _load_audit(audit_path, spec.public_oof_audit_sha256)
    artifacts = audit.get("artifacts", {})
    if artifacts.get(path.name) != spec.public_oof_panel_sha256:
        raise RuntimeError("public OOF audit/panel hash mismatch")

    panel = pd.read_csv(path, dtype={ID_COLUMN: str, "target": str})
    required = {
        ID_COLUMN,
        "target",
        "is_gold",
        "dino_public20_equal_oof",
        "radimagenet_v52_oof",
        "radimagenet_e11_public_oof",
        "public_image_oof_mean4",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise RuntimeError(f"public OOF panel missing columns: {sorted(missing)}")
    expected_rows = spec.expected_studies * len(spec.targets)
    if len(panel) != expected_rows or panel.duplicated([ID_COLUMN, "target"]).any():
        raise RuntimeError("public OOF panel grain/row-count drift")
    if panel["target"].drop_duplicates().tolist() != list(spec.targets):
        raise RuntimeError("public OOF target order drift")
    ids = panel.loc[panel["target"] == spec.targets[0], ID_COLUMN].tolist()
    if len(ids) != spec.expected_studies or len(set(ids)) != spec.expected_studies:
        raise RuntimeError("public OOF UID coverage drift")
    for target in spec.targets:
        if panel.loc[panel["target"] == target, ID_COLUMN].tolist() != ids:
            raise RuntimeError(f"public OOF UID order drift for target {target}")
    for column in sorted(required.difference({ID_COLUMN, "target", "is_gold"})):
        _require_probability(column, panel[column].to_numpy(float), (expected_rows,))
    gold_by_uid = panel.loc[panel["target"] == spec.targets[0], "is_gold"].astype(bool)
    if int(gold_by_uid.sum()) != spec.expected_gold:
        raise RuntimeError("public OOF Gold subset drift")
    for target in spec.targets[1:]:
        current = panel.loc[panel["target"] == target, "is_gold"].astype(bool).to_numpy()
        if not np.array_equal(current, gold_by_uid.to_numpy()):
            raise RuntimeError("is_gold varies across targets")
    return panel, ids, audit


def load_coatnet_v4(
    path: Path,
    audit_path: Path,
    ids: Sequence[str],
    gold_mask: np.ndarray,
    spec: ReducerSpec,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if sha256_file(path) != spec.coatnet_v4_npz_sha256:
        raise RuntimeError(f"CoAtNet V4 artifact hash drift: {path}")
    audit = _load_audit(audit_path, spec.coatnet_v4_audit_sha256)
    if audit.get("checkpoint_sha256") != spec.coatnet_v4_checkpoint_sha256:
        raise RuntimeError("CoAtNet V4 audit checkpoint hash drift")
    payload = _load_npz(
        path,
        {
            "study_instance_uid",
            "targets",
            "probability",
            "cohort_rank",
            "prediction_success",
            "prediction_scope",
            "checkpoint_sha256",
        },
    )
    observed_ids = payload["study_instance_uid"].astype(str).tolist()
    if observed_ids != list(ids):
        raise RuntimeError("CoAtNet V4 UID/order drift")
    if payload["targets"].astype(str).tolist() != list(spec.targets):
        raise RuntimeError("CoAtNet V4 target order drift")
    shape = (spec.expected_studies, len(spec.targets))
    _require_probability("CoAtNet V4 probability", payload["probability"], shape)
    _require_probability("CoAtNet V4 cohort rank", payload["cohort_rank"], shape)
    if not payload["prediction_success"].all():
        raise RuntimeError("CoAtNet V4 contains failed studies")
    if payload["checkpoint_sha256"].astype(str).tolist() != [
        spec.coatnet_v4_checkpoint_sha256
    ]:
        raise RuntimeError("CoAtNet V4 embedded checkpoint hash drift")
    expected_scope = np.where(gold_mask, "gold58_holdout", "in_sample_full_model")
    if not np.array_equal(payload["prediction_scope"].astype(str), expected_scope):
        raise RuntimeError("CoAtNet V4 prediction-scope drift")
    # Remote NumPy may permute exact ties differently.  Require every stored
    # column to remain an exact 0..1 rank permutation and preserve strict order.
    grid = np.arange(spec.expected_studies, dtype=np.float64) / max(
        spec.expected_studies - 1, 1
    )
    for target_index in range(len(spec.targets)):
        rank = payload["cohort_rank"][:, target_index].astype(np.float64)
        if not np.allclose(np.sort(rank), grid, atol=5e-7, rtol=0):
            raise RuntimeError("CoAtNet V4 cohort rank is not a percentile permutation")
        probability = payload["probability"][:, target_index]
        order = np.argsort(probability, kind="stable")
        strict = np.diff(probability[order]) > 0
        if np.any(np.diff(rank[order])[strict] <= 0):
            raise RuntimeError("CoAtNet V4 cohort rank violates probability order")
    return payload, {"audit": audit, "npz_sha256": spec.coatnet_v4_npz_sha256}


def load_dinov3(
    path: Path, audit_path: Path, ids: Sequence[str], spec: ReducerSpec
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    audit = _load_audit(audit_path)
    artifact_hash = _verify_audited_artifact(audit, path)
    if audit.get("source_notebook_sha256") != spec.dinov3_source_sha256:
        raise RuntimeError("DINOv3 source notebook hash drift")
    observed_checkpoint_hashes = {
        int(key): str(value) for key, value in audit.get("checkpoint_sha256", {}).items()
    }
    if observed_checkpoint_hashes != dict(spec.dino_checkpoint_sha256):
        raise RuntimeError("DINOv3 checkpoint hash drift")
    if audit.get("checkpoint_hashes_match_manifest") is not True:
        raise RuntimeError("DINOv3 checkpoint manifest gate failed")
    payload = _load_npz(
        path,
        {
            "study_instance_uid",
            "original_train_index",
            "targets",
            "model_fold",
            "probability",
            "mean_probability",
            "rank_mean",
            "valid_study",
            "scope",
        },
    )
    if payload["study_instance_uid"].astype(str).tolist() != list(ids):
        raise RuntimeError("DINOv3 UID/order drift")
    if not np.array_equal(
        payload["original_train_index"].astype(int), np.arange(spec.expected_studies)
    ):
        raise RuntimeError("DINOv3 original train-index drift")
    if payload["targets"].astype(str).tolist() != list(spec.targets):
        raise RuntimeError("DINOv3 target order drift")
    folds = payload["model_fold"].astype(int)
    if folds.tolist() != [0, 1, 2, 3, 4]:
        raise RuntimeError("DINOv3 fold/member drift")
    probability_shape = (5, spec.expected_studies, len(spec.targets))
    _require_probability("DINOv3 members", payload["probability"], probability_shape)
    _require_probability(
        "DINOv3 mean", payload["mean_probability"], probability_shape[1:]
    )
    _require_probability(
        "DINOv3 rank mean", payload["rank_mean"], probability_shape[1:]
    )
    if not payload["valid_study"].all():
        raise RuntimeError("DINOv3 contains failed studies")
    if payload["scope"].astype(str).tolist() != [DINO_SCOPE]:
        raise RuntimeError("DINOv3 prediction scope drift")
    expected_audit = {
        "prediction_scope": DINO_SCOPE,
        "study_count": spec.expected_studies,
        "unique_study_count": spec.expected_studies,
        "target_count": len(spec.targets),
        "targets": list(spec.targets),
        "member_count": 5,
        "probability_shape": [5, spec.expected_studies, len(spec.targets)],
        "valid_study_count": spec.expected_studies,
        "finite_probability_count": 5 * spec.expected_studies * len(spec.targets),
        "expected_probability_count": 5 * spec.expected_studies * len(spec.targets),
        "probability_gate": True,
        "mean_recomputation_gate": True,
        "rank_gate": True,
        "zero_decoded_slot_study_count": 0,
        "study_build_failure_count": 0,
        "dicom_failure_count": 0,
    }
    for key, expected_value in expected_audit.items():
        if audit.get(key) != expected_value:
            raise RuntimeError(
                f"DINOv3 audit field drift: {key}={audit.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    if int(audit.get("decoded_slot_count_min", 0)) <= 0:
        raise RuntimeError("DINOv3 audit contains a study with no decoded slot")
    failure_counts = audit.get("dicom_failure_counts")
    if not isinstance(failure_counts, Mapping) or any(
        int(value) != 0 for value in failure_counts.values()
    ):
        raise RuntimeError("DINOv3 DICOM failure counters are not all zero")
    if audit.get("study_build_failures") != [] or audit.get("dicom_failure_examples") != []:
        raise RuntimeError("DINOv3 audit contains failure details")
    runtime = audit.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("DINOv3 audit lacks runtime identity")
    gpu_names = [str(value) for value in runtime.get("gpu_names", [])]
    gpu_capabilities = [tuple(value) for value in runtime.get("gpu_capabilities", [])]
    if not gpu_names or any("T4" not in name for name in gpu_names):
        raise RuntimeError(f"DINOv3 runtime is not Tesla T4: {gpu_names}")
    if not gpu_capabilities or any(value < (7, 5) for value in gpu_capabilities):
        raise RuntimeError(f"DINOv3 CUDA capability drift: {gpu_capabilities}")
    expected_mean = payload["probability"].astype(np.float64).mean(axis=0)
    if not np.allclose(expected_mean, payload["mean_probability"], atol=1e-6, rtol=0):
        raise RuntimeError("DINOv3 mean probability drift")
    rank_tie_equivalence = validate_dinov3_rank_tie_equivalence(
        payload["probability"], payload["rank_mean"], spec.targets
    )
    return payload, {
        "audit": audit,
        "npz_sha256": artifact_hash,
        "rank_tie_equivalence": rank_tie_equivalence,
    }


def _apply_v5_attribution_correction(
    audit: dict[str, Any],
    audit_path: Path,
    correction_path: Path,
    shard_index: int,
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    correction_sha256 = sha256_file(correction_path)
    if shard_index != 1 or int(binding.get("shard_index", -1)) != shard_index:
        raise RuntimeError("CoAtNet V5 attribution correction is allowed only for s01")
    if correction_sha256 != binding.get("correction_sha256"):
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction file hash drift")
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    if correction.get("format") != "rsna-knee-coatnet-v5-attribution-correction-v1":
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction format drift")
    if correction.get("status") != "PASS" or correction.get("correction_scope") != (
        "AUDIT_METADATA_ONLY_NO_PREDICTION_CHANGE"
    ):
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction status drift")
    if correction.get("prediction_artifact_bytes_unchanged") is not True:
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction changes scores")
    if correction.get("remote_run_audit_immutable") is not True:
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction rewrites audit")
    if correction.get("all_other_fields_unchanged") is not True:
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction scope drift")
    if correction.get("effective_status") != "PASS_WITH_LOCAL_ATTRIBUTION_CORRECTION":
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction effective status drift")
    if correction.get("frozen_slots") != [list(slot) for slot in V5_FROZEN_SLOTS]:
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction SLOTS drift")
    expected_slots_component_sha256 = binding.get("frozen_slots_component_sha256")
    if (
        correction.get("frozen_slots_component_sha256")
        != expected_slots_component_sha256
    ):
        raise RuntimeError(
            f"CoAtNet V5 shard {shard_index} correction SLOTS component drift"
        )
    if (
        audit.get("source_component_sha256", {}).get("SLOTS")
        != expected_slots_component_sha256
    ):
        raise RuntimeError(
            f"CoAtNet V5 shard {shard_index} audit SLOTS component drift"
        )

    source = correction.get("source", {})
    if source.get("slug") != binding.get("source_slug"):
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction slug drift")
    if int(source.get("version", -1)) != int(binding.get("source_version", -2)):
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction version drift")
    if int(source.get("version", -1)) != int(audit.get("expected_kaggle_version", -2)):
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction/audit version drift")
    if source.get("directory") != binding.get("source_directory"):
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction directory drift")
    declared_directory = Path(str(source.get("directory")))
    if not declared_directory.is_absolute():
        declared_directory = Path(__file__).resolve().parents[2] / declared_directory
    if declared_directory.resolve() != audit_path.parent.resolve():
        raise RuntimeError(
            f"CoAtNet V5 shard {shard_index} correction source-location drift"
        )
    source_hashes = source.get("file_sha256", {})
    expected_source_hashes = dict(binding.get("source_file_sha256", {}))
    if source_hashes != expected_source_hashes:
        raise RuntimeError(
            f"CoAtNet V5 shard {shard_index} correction source hash-set drift"
        )
    if source_hashes.get("RUN_AUDIT.json") != sha256_file(audit_path):
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction audit hash drift")
    for filename, expected_sha256 in source_hashes.items():
        source_path = audit_path.parent / filename
        if not source_path.is_file() or sha256_file(source_path) != expected_sha256:
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} correction source hash drift: "
                f"{filename}"
            )

    original_events = audit.get("dicom_read_failures")
    if not isinstance(original_events, list):
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction events missing")
    if correction.get("original_dicom_read_failures_sha256") != canonical_json_sha256(
        original_events
    ):
        raise RuntimeError(
            f"CoAtNet V5 shard {shard_index} correction original-event hash drift"
        )
    operations = correction.get("corrections")
    if not isinstance(operations, list) or len(operations) != int(
        correction.get("correction_count", -1)
    ):
        raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction count drift")
    if len(operations) != 1:
        raise RuntimeError(
            f"CoAtNet V5 shard {shard_index} correction must contain exactly one op"
        )

    effective = copy.deepcopy(audit)
    effective_events = effective["dicom_read_failures"]
    for operation in operations:
        if operation.get("field") != "slot_fluid_sensitive":
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} correction field not allowed"
            )
        event_index = int(operation.get("event_index", -1))
        if not 0 <= event_index < len(effective_events):
            raise RuntimeError(f"CoAtNet V5 shard {shard_index} correction event drift")
        expected_pointer = (
            f"/dicom_read_failures/{event_index}/slot_fluid_sensitive"
        )
        if operation.get("json_pointer") != expected_pointer:
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} correction pointer drift"
            )
        event = effective_events[event_index]
        locator = operation.get("locator")
        if not isinstance(locator, dict) or any(
            event.get(key) != value for key, value in locator.items()
        ):
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} correction locator drift"
            )
        if operation.get("locator_sha256") != canonical_json_sha256(locator):
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} correction locator hash drift"
            )
        if operation.get("original_event_sha256") != canonical_json_sha256(event):
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} correction event hash drift"
            )
        observed = int(event.get("slot_fluid_sensitive", -999))
        if observed != int(operation.get("from", -998)):
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} correction before-value drift"
            )
        slot_index = int(event.get("slot_index", -1))
        if not 0 <= slot_index < len(V5_FROZEN_SLOTS):
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} correction slot-index drift"
            )
        derived = int(V5_FROZEN_SLOTS[slot_index][1])
        if int(operation.get("to", -997)) != derived:
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} correction replacement drift"
            )
        event["slot_fluid_sensitive"] = derived
        if operation.get("corrected_event_sha256") != canonical_json_sha256(event):
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} corrected event hash drift"
            )

    if correction.get("effective_dicom_read_failures_sha256") != canonical_json_sha256(
        effective_events
    ):
        raise RuntimeError(
            f"CoAtNet V5 shard {shard_index} correction effective-event hash drift"
        )
    if correction.get("effective_run_audit_canonical_sha256") != canonical_json_sha256(
        effective
    ):
        raise RuntimeError(
            f"CoAtNet V5 shard {shard_index} correction effective-audit hash drift"
        )
    if correction.get("effective_run_audit_canonical_sha256") != binding.get(
        "effective_run_audit_canonical_sha256"
    ):
        raise RuntimeError(
            f"CoAtNet V5 shard {shard_index} correction bound effective-audit drift"
        )
    return effective, {
        "scope": "local_provenance_overlay",
        "source_slug": source["slug"],
        "source_version": int(source["version"]),
        "source_run_audit_sha256": sha256_file(audit_path),
        "correction_sidecar_sha256": correction_sha256,
        "effective_run_audit_canonical_sha256": canonical_json_sha256(effective),
        "correction_count": len(operations),
    }


def load_coatnet_v5_shards(
    shard_directories: Sequence[Path],
    ids: Sequence[str],
    spec: ReducerSpec,
    gold_mask: np.ndarray | None = None,
    attribution_correction_paths: Mapping[int, Path] | None = None,
    attribution_correction_bindings: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(shard_directories) != spec.v5_shard_count:
        raise RuntimeError("CoAtNet V5 shard-count drift")
    pieces: list[dict[str, np.ndarray]] = []
    manifests: list[dict[str, Any]] = []
    quality_scopes: list[np.ndarray] = []
    full_cohort_dicom_read_failures = 0
    full_cohort_affected_indices: set[int] = set()
    correction_paths = dict(attribution_correction_paths or {})
    correction_bindings = dict(attribution_correction_bindings or {})
    if set(correction_paths) != set(correction_bindings):
        raise RuntimeError("CoAtNet V5 correction path/binding keys drift")
    if correction_paths and set(correction_paths) != {1}:
        raise RuntimeError("CoAtNet V5 attribution correction is allowed only for s01")
    for shard_index, directory in enumerate(shard_directories):
        stem = f"coatnet_v5_train_raw_s{shard_index:02d}"
        path = directory / f"{stem}.npz"
        audit_path = directory / "RUN_AUDIT.json"
        audit = _load_audit(audit_path)
        artifact_hash = _verify_audited_artifact(audit, path)
        if int(audit.get("expected_kaggle_version", -1)) != int(
            V5_EXPECTED_KAGGLE_VERSION[shard_index]
        ):
            raise RuntimeError(
                f"CoAtNet V5 expected Kaggle version drift in shard {shard_index}"
            )
        if audit.get("artifact_revision") != V5_EXPECTED_ARTIFACT_REVISION[shard_index]:
            raise RuntimeError(
                f"CoAtNet V5 artifact revision drift in shard {shard_index}"
            )
        correction_manifest = None
        if shard_index in correction_paths:
            audit, correction_manifest = _apply_v5_attribution_correction(
                audit,
                audit_path,
                correction_paths[shard_index],
                shard_index,
                correction_bindings[shard_index],
            )
        piece = _load_npz(
            path,
            {
                "study_instance_uid",
                "targets",
                "original_train_index",
                "probability",
                "success",
                "shard_index",
                "shard_count",
                "scope",
                "checkpoint_sha256",
                "source_notebook_sha256",
                "is_gold58_holdout",
            },
        )
        if piece["targets"].astype(str).tolist() != list(spec.targets):
            raise RuntimeError(f"CoAtNet V5 target drift in shard {shard_index}")
        if int(piece["shard_index"][0]) != shard_index:
            raise RuntimeError(f"CoAtNet V5 shard-index drift in shard {shard_index}")
        if int(piece["shard_count"][0]) != spec.v5_shard_count:
            raise RuntimeError(f"CoAtNet V5 shard-count drift in shard {shard_index}")
        expected_indices = np.array_split(
            np.arange(spec.expected_studies, dtype=np.int32), spec.v5_shard_count
        )[shard_index]
        if not np.array_equal(piece["original_train_index"], expected_indices):
            raise RuntimeError(f"CoAtNet V5 index coverage drift in shard {shard_index}")
        expected_ids = [ids[int(index)] for index in expected_indices]
        if piece["study_instance_uid"].astype(str).tolist() != expected_ids:
            raise RuntimeError(f"CoAtNet V5 UID/order drift in shard {shard_index}")
        if gold_mask is not None and not np.array_equal(
            piece["is_gold58_holdout"].astype(bool), gold_mask[expected_indices]
        ):
            raise RuntimeError(f"CoAtNet V5 Gold-58 lineage drift in shard {shard_index}")
        shape = (len(expected_indices), len(spec.targets))
        _require_probability(
            f"CoAtNet V5 probability shard {shard_index}", piece["probability"], shape
        )
        if not piece["success"].all():
            raise RuntimeError(f"CoAtNet V5 failed study in shard {shard_index}")
        if piece["scope"].astype(str).tolist() != [V5_RAW_SCOPE]:
            raise RuntimeError(f"CoAtNet V5 scope drift in shard {shard_index}")
        if piece["checkpoint_sha256"].astype(str).tolist() != [
            spec.coatnet_v5_checkpoint_sha256
        ]:
            raise RuntimeError(f"CoAtNet V5 checkpoint drift in shard {shard_index}")
        if piece["source_notebook_sha256"].astype(str).tolist() != [
            spec.coatnet_v5_source_sha256
        ]:
            raise RuntimeError(f"CoAtNet V5 source drift in shard {shard_index}")
        if audit.get("checkpoint_sha256") != spec.coatnet_v5_checkpoint_sha256:
            raise RuntimeError(f"CoAtNet V5 audit checkpoint drift in shard {shard_index}")
        if audit.get("source_notebook_sha256") != spec.coatnet_v5_source_sha256:
            raise RuntimeError(f"CoAtNet V5 audit source drift in shard {shard_index}")
        expected_audit = {
            "prediction_scope": V5_RAW_SCOPE,
            "raw_probabilities_before_ranks": True,
            "ranks_computed": False,
            "checkpoint_hash_match": True,
            "source_component_hashes_match_manifest": True,
            "shard_index": shard_index,
            "shard_count": spec.v5_shard_count,
            "expected_train_study_count": spec.expected_studies,
            "source_train_study_count": spec.expected_studies,
            "source_unique_uid_count": spec.expected_studies,
            "source_uid_sequence_hash_match": True,
            "expected_shard_study_count": len(expected_indices),
            "shard_study_count": len(expected_indices),
            "shard_unique_uid_count": len(expected_indices),
            "duplicate_uid_count": 0,
            "target_count": len(spec.targets),
            "target_order": list(spec.targets),
            "prediction_success_count": len(expected_indices),
            "failed_study_count": 0,
            "finite_probability_count": len(expected_indices) * len(spec.targets),
            "expected_probability_count": len(expected_indices) * len(spec.targets),
            "missing_series_metadata_uid_count": 0,
            "worker_count": 2,
            "t4_device_gate": True,
            "gold58_holdout_count": int(
                piece["is_gold58_holdout"].astype(bool).sum()
            ),
            "in_sample_count": int(
                (~piece["is_gold58_holdout"].astype(bool)).sum()
            ),
        }
        for key, expected_value in expected_audit.items():
            if audit.get(key) != expected_value:
                raise RuntimeError(
                    f"CoAtNet V5 audit field drift in shard {shard_index}: "
                    f"{key}={audit.get(key)!r}, expected {expected_value!r}"
                )
        dicom_read_failure_count = int(audit.get("dicom_read_failure_count", -1))
        if dicom_read_failure_count < 0:
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} lacks a DICOM failure count"
            )
        full_cohort_dicom_read_failures += dicom_read_failure_count
        if dicom_read_failure_count == 0:
            if int(audit.get("valid_slice_count_min", 0)) <= 0:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} contains an empty decoded study"
                )
            if "input_quality_scope" in piece:
                quality_scope = piece["input_quality_scope"].astype(str)
                if quality_scope.shape != (len(expected_indices),) or set(
                    quality_scope.tolist()
                ) != {"exact_source_preprocessing"}:
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} zero-skip quality-scope drift"
                    )
            else:
                quality_scope = np.full(
                    len(expected_indices), "exact_source_preprocessing", dtype="U64"
                )
        else:
            required_diagnostic_fields = {
                "dicom_read_count",
                "dicom_read_failure_count",
                "selected_series_count",
                "valid_slice_count",
                "input_quality_scope",
            }
            missing_diagnostic_fields = required_diagnostic_fields.difference(piece)
            if missing_diagnostic_fields:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} lacks attributed diagnostics: "
                    f"{sorted(missing_diagnostic_fields)}"
                )
            per_study_read_count = piece["dicom_read_count"].astype(np.int64)
            per_study_failure_count = piece["dicom_read_failure_count"].astype(np.int64)
            selected_series_count = piece["selected_series_count"].astype(np.int64)
            valid_slice_count = piece["valid_slice_count"].astype(np.int64)
            quality_scope = piece["input_quality_scope"].astype(str)
            for name, values in {
                "dicom_read_count": per_study_read_count,
                "dicom_read_failure_count": per_study_failure_count,
                "selected_series_count": selected_series_count,
                "valid_slice_count": valid_slice_count,
                "input_quality_scope": quality_scope,
            }.items():
                if values.shape != (len(expected_indices),):
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} {name} shape drift"
                    )
            policy = audit.get("dicom_read_failure_policy")
            expected_policy = {
                "behavior": "SOURCE_FAITHFUL_NONE_TO_ZERO",
                "read_retry": False,
                "neighboring_slice_substitution": False,
                "whole_study_probability_fallback": False,
                "max_failures_full_cohort": spec.v5_max_dicom_read_failures_full_cohort,
                "max_failures_per_shard": spec.v5_max_dicom_read_failures_per_shard,
                "max_failures_per_study": spec.v5_max_dicom_read_failures_per_study,
                "max_affected_studies_per_shard": (
                    spec.v5_max_affected_studies_per_shard
                ),
                "max_failure_fraction_per_shard": (
                    spec.v5_max_dicom_read_failure_fraction_per_shard
                ),
            }
            if policy != expected_policy:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} bounded failure-policy drift"
                )
            if audit.get("status") != "PASS_WITH_BOUNDED_SOURCE_PIXEL_SKIPS":
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} bounded status drift"
                )
            if audit.get("dicom_read_failure_policy_gate") is not True:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} bounded failure gate did not pass"
                )
            if audit.get("dicom_read_failure_records_consistent") is not True:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} failure records are inconsistent"
                )
            if audit.get("selected_series_gate_pass") is not True:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} selected-series gate failed"
                )
            if audit.get("valid_slice_gate_pass") is not True:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} valid-slice gate failed"
                )
            if audit.get("all_study_predictions_finite") is not True:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} finite-prediction gate failed"
                )
            events = audit.get("dicom_read_failures")
            if not isinstance(events, list) or len(events) != dicom_read_failure_count:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} failure-event count drift"
                )
            expected_sorted_events = sorted(
                events,
                key=lambda row: (
                    int(row.get("original_train_index", -1)),
                    int(row.get("slot_index", -1)),
                    int(row.get("selected_position", -1)),
                    str(row.get("path_relative_to_train_series", "")),
                ),
            )
            if events != expected_sorted_events:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} failure-event order drift"
                )
            event_counts: dict[int, int] = {}
            for event in events:
                original_index = int(event.get("original_train_index", -1))
                if original_index not in set(expected_indices.astype(int).tolist()):
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event index drift"
                    )
                local_index = original_index - int(expected_indices[0])
                if str(event.get("StudyInstanceUID", "")) != expected_ids[local_index]:
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event UID drift"
                    )
                relative_path = str(event.get("path_relative_to_train_series", ""))
                if not relative_path or relative_path.startswith("/") or ".." in Path(
                    relative_path
                ).parts:
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event path drift"
                    )
                expected_path_hash = hashlib.sha256(relative_path.encode()).hexdigest()
                if event.get("path_relative_sha256") != expected_path_hash:
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event path hash drift"
                    )
                if event.get("stage") != "read_px" or event.get("action") != (
                    "SOURCE_FAITHFUL_NONE_TO_ZERO"
                ):
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event action drift"
                    )
                slot_index = int(event.get("slot_index", -1))
                if not 0 <= slot_index < len(V5_FROZEN_SLOTS):
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event slot drift"
                    )
                expected_plane, expected_fluid, expected_count = V5_FROZEN_SLOTS[
                    slot_index
                ]
                if event.get("slot_plane") != expected_plane:
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event plane drift"
                    )
                if int(event.get("slot_fluid_sensitive", -999)) != expected_fluid:
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event fluid drift"
                    )
                if int(event.get("slot_output_offset", -1)) != (
                    V5_FROZEN_SLOT_OFFSETS[slot_index]
                ):
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event offset drift"
                    )
                if int(event.get("slot_output_count", -1)) != expected_count:
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event slot count drift"
                    )
                affected = event.get("affected_output_indices")
                if not isinstance(affected, list) or not affected:
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event slices missing"
                    )
                slot_offset = V5_FROZEN_SLOT_OFFSETS[slot_index]
                if any(
                    not slot_offset <= int(value) < slot_offset + expected_count
                    for value in affected
                ):
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event slice range drift"
                    )
                if int(event.get("affected_output_slice_count", -1)) != len(affected):
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event slice count drift"
                    )
                file_hash = event.get("file_sha256_when_readable")
                if file_hash is not None and (
                    not isinstance(file_hash, str) or len(file_hash) != 64
                ):
                    raise RuntimeError(
                        f"CoAtNet V5 shard {shard_index} failure-event file hash drift"
                    )
                event_counts[original_index] = event_counts.get(original_index, 0) + 1
                full_cohort_affected_indices.add(original_index)
            observed_event_counts = np.asarray(
                [event_counts.get(int(index), 0) for index in expected_indices],
                dtype=np.int64,
            )
            if not np.array_equal(per_study_failure_count, observed_event_counts):
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} per-study failure attribution drift"
                )
            if int(per_study_read_count.sum()) != int(audit.get("dicom_read_count", -1)):
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} read-count reconciliation drift"
                )
            if int(per_study_failure_count.sum()) != dicom_read_failure_count:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} failure-count reconciliation drift"
                )
            if int(per_study_failure_count.max()) > spec.v5_max_dicom_read_failures_per_study:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} per-study failure budget exceeded"
                )
            if int((per_study_failure_count > 0).sum()) > (
                spec.v5_max_affected_studies_per_shard
            ):
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} affected-study budget exceeded"
                )
            if dicom_read_failure_count > spec.v5_max_dicom_read_failures_per_shard:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} shard failure budget exceeded"
                )
            failure_fraction = dicom_read_failure_count / max(
                int(per_study_read_count.sum()), 1
            )
            if failure_fraction > spec.v5_max_dicom_read_failure_fraction_per_shard:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} failure fraction exceeded"
                )
            if int(selected_series_count.min()) < spec.v5_min_selected_series_per_study:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} selected-series floor drift"
                )
            if int(valid_slice_count.min()) < spec.v5_min_valid_slices_per_study:
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} valid-slice floor drift"
                )
            expected_quality_scope = np.where(
                per_study_failure_count > 0,
                "source_native_zero_fill_attributed",
                "exact_source_preprocessing",
            )
            if not np.array_equal(quality_scope, expected_quality_scope):
                raise RuntimeError(
                    f"CoAtNet V5 shard {shard_index} input-quality scope drift"
                )
        gpu_names = [str(value) for value in audit.get("gpu_names", [])]
        gpu_capabilities = [tuple(value) for value in audit.get("gpu_capabilities", [])]
        if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} runtime is not dual T4: {gpu_names}"
            )
        if len(gpu_capabilities) != 2 or any(
            value < (7, 5) for value in gpu_capabilities
        ):
            raise RuntimeError(
                f"CoAtNet V5 shard {shard_index} CUDA capability drift: "
                f"{gpu_capabilities}"
            )
        pieces.append(piece)
        quality_scopes.append(quality_scope)
        manifests.append(
            {
                "audit": audit,
                "npz_sha256": artifact_hash,
                "attribution_correction": correction_manifest,
            }
        )

    if full_cohort_dicom_read_failures > spec.v5_max_dicom_read_failures_full_cohort:
        raise RuntimeError("CoAtNet V5 full-cohort DICOM failure budget exceeded")
    order = np.concatenate([piece["original_train_index"] for piece in pieces]).astype(int)
    observed_ids = np.concatenate([piece["study_instance_uid"] for piece in pieces]).astype(str)
    probability = np.concatenate([piece["probability"] for piece in pieces], axis=0)
    input_quality_scope = np.concatenate(quality_scopes).astype("U64")
    if not np.array_equal(order, np.arange(spec.expected_studies)):
        raise RuntimeError("CoAtNet V5 shards do not cover train order exactly once")
    if observed_ids.tolist() != list(ids):
        raise RuntimeError("CoAtNet V5 merged UID/order drift")
    # This is deliberately the first rank operation on V5 outputs.
    cohort_rank = rankpct(np.clip(probability, 0.0, 1.0)).astype(np.float32)
    return {
        "study_instance_uid": observed_ids,
        "targets": np.asarray(spec.targets, dtype="U32"),
        "original_train_index": order.astype(np.int32),
        "probability": probability.astype(np.float32),
        "cohort_rank": cohort_rank,
        "input_quality_scope": input_quality_scope,
        "scope": np.asarray([V5_RAW_SCOPE], dtype="U64"),
        "checkpoint_sha256": np.asarray(
            [spec.coatnet_v5_checkpoint_sha256], dtype="U64"
        ),
        "source_notebook_sha256": np.asarray(
            [spec.coatnet_v5_source_sha256], dtype="U64"
        ),
    }, {
        "shards": manifests,
        "rank_computed_after_complete_merge": True,
        "full_cohort_dicom_read_failure_count": full_cohort_dicom_read_failures,
        "full_cohort_dicom_read_failure_budget": (
            spec.v5_max_dicom_read_failures_full_cohort
        ),
        "full_cohort_affected_study_count": len(full_cohort_affected_indices),
        "full_cohort_dicom_read_failure_gate": True,
    }


def coatnet_v45_no_swin(
    primary_rank: np.ndarray,
    legacy_rank: np.ndarray,
    targets: Sequence[str] = TARGET_COLUMNS,
) -> tuple[np.ndarray, dict[str, Any]]:
    primary_rank = np.asarray(primary_rank, dtype=np.float64)
    legacy_rank = np.asarray(legacy_rank, dtype=np.float64)
    if primary_rank.shape != legacy_rank.shape or primary_rank.shape[1] != len(targets):
        raise RuntimeError("CoAtNet V4/V5 rank shape mismatch")
    combined = primary_rank.copy()
    correlations: dict[str, float] = {}
    weights: dict[str, float] = {}
    for target_index, target in enumerate(targets):
        weight = float(V45_COMPLEMENT_WEIGHT[target])
        correlation = float(
            np.corrcoef(primary_rank[:, target_index], legacy_rank[:, target_index])[0, 1]
        )
        correlations[target] = correlation
        if not np.isfinite(correlation):
            weight = 0.0
        elif correlation > 0.992:
            weight *= 0.50
        elif correlation < 0.65:
            weight *= 0.40
        weights[target] = weight
        if weight > 0:
            combined[:, target_index] = (
                (1.0 - weight) * primary_rank[:, target_index]
                + weight * legacy_rank[:, target_index]
            )
    return rankpct(combined).astype(np.float32), {
        "recipe": "frozen_V14_V4.5_CoAtNet_complement_without_optional_Swin",
        "optional_swin_assets": 0,
        "primary_legacy_rank_correlation": correlations,
        "effective_complement_weight": weights,
    }


def directional_final_fusion(
    public_oof_proxy: np.ndarray,
    dinov3_rank_mean: np.ndarray,
    coatnet_rank: np.ndarray,
    targets: Sequence[str] = TARGET_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply exact stage weights to an explicitly non-exact transformer proxy."""

    base_rank = average_rankpct(public_oof_proxy)
    dino3_rank = average_rankpct(dinov3_rank_mean)
    transformer_proxy = 0.55 * base_rank + 0.45 * dino3_rank
    transformer_rank = average_rankpct(transformer_proxy)
    coatnet_rank = average_rankpct(coatnet_rank)
    output = transformer_rank.copy()
    correlations: dict[str, float] = {}
    weights: dict[str, float] = {}
    for target_index, target in enumerate(targets):
        correlation = float(
            np.corrcoef(
                transformer_rank[:, target_index], coatnet_rank[:, target_index]
            )[0, 1]
        )
        weight = float(V45_FINAL_COATNET_WEIGHT[target])
        base_weight = float(V45_BASE_0935_WEIGHT[target])
        if not np.isfinite(correlation):
            weight = base_weight
        elif correlation > 0.992:
            weight = 0.65 * weight + 0.35 * base_weight
        elif correlation < 0.60:
            weight = 0.50 * weight + 0.50 * base_weight
        correlations[target] = correlation
        weights[target] = weight
        output[:, target_index] = (
            (1.0 - weight) * transformer_rank[:, target_index]
            + weight * coatnet_rank[:, target_index]
        )
    output = average_rankpct(output).astype(np.float32)
    return transformer_proxy.astype(np.float32), output, {
        "strength": "directional_only",
        "scope": DIRECTIONAL_SCOPE,
        "exact_stage_weights": {
            "dinov3": 0.45,
            "final_coatnet": weights,
        },
        "transformer_coatnet_rank_correlation": correlations,
        "proxy_substitution": (
            "public_image_oof_mean4 replaces the exact V14 transformer/V18-calibrator "
            "train output; therefore this fusion is not an exact V14 train prediction"
        ),
    }


def build_long_panel(
    base: pd.DataFrame,
    dino: Mapping[str, np.ndarray],
    v4: Mapping[str, np.ndarray],
    v5: Mapping[str, np.ndarray],
    coatnet_rank: np.ndarray,
    transformer_proxy: np.ndarray,
    directional_rank: np.ndarray,
    spec: ReducerSpec,
) -> pd.DataFrame:
    panel = base.copy()
    gold_mask = panel.loc[panel["target"] == spec.targets[0], "is_gold"].astype(bool).to_numpy()
    flatten = lambda values: np.asarray(values).T.reshape(-1)  # target-major panel
    folds = dino["model_fold"].astype(int)
    for model_index, fold in enumerate(folds):
        panel[f"dinov3_fold{fold}_probability"] = flatten(
            dino["probability"][model_index]
        )
    panel["dinov3_mean_probability"] = flatten(dino["mean_probability"])
    panel["dinov3_rank_mean"] = flatten(dino["rank_mean"])
    panel["dinov3_prediction_scope"] = DINO_SCOPE
    panel["coatnet_v4_probability"] = flatten(v4["probability"])
    panel["coatnet_v4_cohort_rank"] = flatten(v4["cohort_rank"])
    panel["coatnet_v4_prediction_scope"] = np.tile(
        v4["prediction_scope"].astype(str), len(spec.targets)
    )
    panel["coatnet_v5_probability"] = flatten(v5["probability"])
    panel["coatnet_v5_cohort_rank"] = flatten(v5["cohort_rank"])
    v5_scope = np.where(gold_mask, "gold58_holdout", "in_sample_full_model")
    panel["coatnet_v5_prediction_scope"] = np.tile(v5_scope, len(spec.targets))
    panel["coatnet_v5_input_quality_scope"] = np.tile(
        v5["input_quality_scope"].astype(str), len(spec.targets)
    )
    panel["coatnet_v45_no_swin_rank"] = flatten(coatnet_rank)
    panel["coatnet_v45_prediction_scope"] = np.tile(v5_scope, len(spec.targets))
    panel["coatnet_v45_input_quality_scope"] = panel[
        "coatnet_v5_input_quality_scope"
    ]
    panel["directional_transformer_proxy"] = flatten(transformer_proxy)
    panel["directional_v45_final_rank"] = flatten(directional_rank)
    panel["directional_v45_prediction_scope"] = DIRECTIONAL_SCOPE
    added_numeric = [
        column
        for column in panel.columns
        if column.startswith(("dinov3_", "coatnet_", "directional_"))
        and pd.api.types.is_numeric_dtype(panel[column])
    ]
    if panel[added_numeric].isna().any().any() or not np.isfinite(
        panel[added_numeric].to_numpy(float)
    ).all():
        raise RuntimeError("reduced panel contains missing/non-finite scores")
    return panel


def write_wide(panel: pd.DataFrame, path: Path, targets: Sequence[str]) -> None:
    columns = [
        column
        for column in panel.columns
        if column.startswith(("dinov3_", "coatnet_", "directional_"))
        and pd.api.types.is_numeric_dtype(panel[column])
    ]
    blocks = []
    for column in columns:
        pivot = panel.pivot(index=ID_COLUMN, columns="target", values=column).reindex(
            columns=targets
        )
        pivot.columns = [f"{column}__{target}" for target in targets]
        blocks.append(pivot)
    pd.concat(blocks, axis=1).reset_index().to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def reduce_train_scores(
    *,
    public_oof_panel: Path,
    public_oof_audit: Path,
    coatnet_v4_npz: Path,
    coatnet_v4_audit: Path,
    dinov3_npz: Path,
    dinov3_audit: Path,
    coatnet_v5_shard_directories: Sequence[Path],
    output_directory: Path,
    coatnet_v5_attribution_corrections: Mapping[int, Path] | None = None,
    coatnet_v5_attribution_correction_bindings: (
        Mapping[int, Mapping[str, Any]] | None
    ) = None,
    spec: ReducerSpec = ReducerSpec(),
) -> dict[str, Any]:
    base, ids, base_audit = load_public_oof_panel(
        public_oof_panel, public_oof_audit, spec
    )
    gold_mask = base.loc[base["target"] == spec.targets[0], "is_gold"].astype(bool).to_numpy()
    v4, v4_manifest = load_coatnet_v4(
        coatnet_v4_npz, coatnet_v4_audit, ids, gold_mask, spec
    )
    dino, dino_manifest = load_dinov3(dinov3_npz, dinov3_audit, ids, spec)
    v5, v5_manifest = load_coatnet_v5_shards(
        coatnet_v5_shard_directories,
        ids,
        spec,
        gold_mask,
        coatnet_v5_attribution_corrections,
        coatnet_v5_attribution_correction_bindings,
    )
    coatnet_rank, coatnet_manifest = coatnet_v45_no_swin(
        v5["cohort_rank"], v4["cohort_rank"], spec.targets
    )
    public_proxy = (
        base.pivot(index=ID_COLUMN, columns="target", values="public_image_oof_mean4")
        .reindex(index=ids, columns=spec.targets)
        .to_numpy(float)
    )
    transformer_proxy, directional_rank, directional_manifest = directional_final_fusion(
        public_proxy, dino["rank_mean"], coatnet_rank, spec.targets
    )
    panel = build_long_panel(
        base,
        dino,
        v4,
        v5,
        coatnet_rank,
        transformer_proxy,
        directional_rank,
        spec,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    long_path = output_directory / "train_model_score_panel_long.csv.gz"
    wide_path = output_directory / "train_model_scores_wide.csv.gz"
    v5_path = output_directory / "coatnet_v5_train_raw_merged.npz"
    fusion_path = output_directory / "train_directional_v45_scores.npz"
    panel.to_csv(
        long_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    write_wide(panel, wide_path, spec.targets)
    write_deterministic_npz(v5_path, **v5)
    write_deterministic_npz(
        fusion_path,
        study_instance_uid=np.asarray(ids, dtype="U64"),
        targets=np.asarray(spec.targets, dtype="U32"),
        coatnet_v45_no_swin_rank=coatnet_rank,
        directional_transformer_proxy=transformer_proxy,
        directional_v45_final_rank=directional_rank,
        scope=np.asarray([DIRECTIONAL_SCOPE], dtype="U96"),
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in (long_path, wide_path, v5_path, fusion_path)
    }
    audit = {
        "success": True,
        "study_count": spec.expected_studies,
        "target_count": len(spec.targets),
        "row_count": len(panel),
        "uid_sequence_sha256": uid_sequence_sha256(ids),
        "inputs": {
            "public_oof_panel": {
                "sha256": spec.public_oof_panel_sha256,
                "audit": base_audit,
            },
            "coatnet_v4": v4_manifest,
            "dinov3": dino_manifest,
            "coatnet_v5": v5_manifest,
        },
        "fusions": {
            "coatnet_v45_no_swin": coatnet_manifest,
            "directional_v45_final": directional_manifest,
        },
        "prediction_scope_boundary": {
            "published_image_columns": "published_oof",
            "dinov3": DINO_SCOPE,
            "coatnet_non_gold": "in_sample_full_model",
            "coatnet_gold58": "gold58_holdout",
            "directional_final": DIRECTIONAL_SCOPE,
        },
        "artifacts": artifacts,
    }
    audit_path = output_directory / "RUN_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit
