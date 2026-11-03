"""Token-preserving append-only construction for the post-freeze panel.

The regular pandas round-trip is deterministic, but it can rewrite decimal
tokens by a few last-place digits.  This module treats every frozen pre-Gold
field as an opaque CSV string and appends only the already-produced post-freeze
Gold columns by exact ``StudyInstanceUID x target`` key.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import io
import os
from pathlib import Path
from typing import Mapping, Sequence

from .report_reliability_gold_gate import (
    CANDIDATE_WEIGHT_STATUS,
    EXTERNAL_REQUIRED_COLUMNS,
)
from .report_reliability_postfreeze_gold import GOLD_APPEND_COLUMNS


KEY_COLUMNS: tuple[str, str] = ("StudyInstanceUID", "target")
SOURCE_ROLES: tuple[str, str, str] = (
    "proxy_decisive_presentation",
    "proxy_abstain_presentation",
    "final_external_predictions",
)


@dataclass(frozen=True)
class TokenExactPanelResult:
    rows: int
    studies: int
    source_role_rows: Mapping[str, int]
    original_complete_field_differences: Mapping[str, int]
    token_exact_pre_gold_field_differences: int
    token_exact_append_field_differences: int


@dataclass(frozen=True)
class GoldAnnotationsResult:
    rows: int
    studies: int
    assessable_rows: int


def _open_csv_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_exact_csv(path: str | Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    source = Path(path)
    with _open_csv_text(source) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{source}: missing or duplicate CSV header")
        header = tuple(reader.fieldnames)
        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{source}: malformed CSV row {index}")
            rows.append({column: str(row[column]) for column in header})
    return header, rows


def _key(row: Mapping[str, str], *, name: str) -> tuple[str, str]:
    value = tuple(row[column] for column in KEY_COLUMNS)
    if any(not item.strip() for item in value):
        raise ValueError(f"{name}: empty cell key")
    return value[0], value[1]


def _index_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    name: str,
) -> dict[tuple[str, str], Mapping[str, str]]:
    result: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = _key(row, name=name)
        if key in result:
            raise ValueError(f"{name}: duplicate cell key {key}")
        result[key] = row
    return result


def _write_deterministic_gzip_csv(
    path: Path,
    *,
    header: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite token-exact panel {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(header),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                compressed.write(buffer.getvalue().encode("utf-8"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_exact_plain_csv(
    path: Path,
    *,
    header: Sequence[str],
    rows: Sequence[Mapping[str, str]],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Gold annotations {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(header),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_gold_annotations_sidecar(
    gold_panel_path: str | Path,
    frozen_external_path: str | Path,
    output_path: str | Path,
    *,
    expected_rows: int = 696,
    expected_studies: int = 58,
    expected_targets_per_study: int = 12,
    append_columns: Sequence[str] = GOLD_APPEND_COLUMNS,
) -> GoldAnnotationsResult:
    """Write exact keys plus the six append-only Gold columns, and nothing else."""

    gold_header, gold_rows = read_exact_csv(gold_panel_path)
    append = tuple(append_columns)
    output_header = (*KEY_COLUMNS, *append)
    missing = sorted(set(output_header).difference(gold_header))
    if missing:
        raise ValueError(f"Gold panel misses annotation columns {missing}")
    gold_index = _index_rows(gold_rows, name="Gold panel")
    if len(gold_rows) != int(expected_rows):
        raise ValueError("Gold annotations row count drift")
    studies = len({key[0] for key in gold_index})
    if studies != int(expected_studies):
        raise ValueError("Gold annotations study count drift")
    per_study: dict[str, int] = {}
    for uid, _ in gold_index:
        per_study[uid] = per_study.get(uid, 0) + 1
    if set(per_study.values()) != {int(expected_targets_per_study)}:
        raise ValueError("Gold annotations are not the expected Cartesian cohort")

    external_header, external_rows = read_exact_csv(frozen_external_path)
    if not set(KEY_COLUMNS).issubset(external_header):
        raise ValueError("frozen external source misses cell keys")
    external_index = _index_rows(external_rows, name="frozen external source")
    if not set(gold_index).issubset(external_index):
        raise ValueError("Gold annotation keys are not covered by frozen external source")

    output_rows = [
        {column: row[column] for column in output_header}
        for row in gold_rows
    ]
    if any(row["gold"] not in {"0", "1"} for row in output_rows):
        raise ValueError("Gold annotations contain a non-binary Gold token")
    if any(
        row["teacher_reference_disagreement"] not in {"", "False", "True"}
        for row in output_rows
    ):
        raise ValueError("Gold annotations contain an invalid disagreement token")
    if any(row["automatic_label_or_weight_action"] != "NONE_FORBIDDEN" for row in output_rows):
        raise ValueError("Gold annotations contain a label/weight action")
    assessable = sum(bool(row["teacher_reference_disagreement"]) for row in output_rows)

    destination = Path(output_path)
    _write_exact_plain_csv(destination, header=output_header, rows=output_rows)
    written_header, written_rows = read_exact_csv(destination)
    if written_header != output_header or written_rows != output_rows:
        raise AssertionError("Gold annotations changed during deterministic write")
    return GoldAnnotationsResult(
        rows=len(written_rows),
        studies=studies,
        assessable_rows=assessable,
    )


def build_token_exact_complete_panel(
    source_paths: Mapping[str, str | Path],
    canonical_complete_path: str | Path,
    output_path: str | Path,
    *,
    expected_rows: int = 52_884,
    expected_studies: int = 4_407,
    pre_gold_columns: Sequence[str] = EXTERNAL_REQUIRED_COLUMNS,
    append_columns: Sequence[str] = GOLD_APPEND_COLUMNS,
) -> TokenExactPanelResult:
    """Build a sidecar with exact source tokens and exact append tokens.

    Row order and append values follow the canonical post-freeze complete
    panel.  Every pre-Gold value comes from one and only one frozen source row.
    """

    if set(source_paths) != set(SOURCE_ROLES):
        raise ValueError("source_paths must contain the exact three frozen roles")
    pre = tuple(pre_gold_columns)
    append = tuple(append_columns)
    if len(set((*pre, *append))) != len(pre) + len(append):
        raise ValueError("token-exact output schema contains duplicate columns")
    if tuple(pre[:2]) != KEY_COLUMNS:
        raise ValueError("token-exact pre-Gold schema must start with exact keys")

    frozen_rows: dict[tuple[str, str], Mapping[str, str]] = {}
    role_counts: dict[str, int] = {}
    for role in SOURCE_ROLES:
        header, rows = read_exact_csv(source_paths[role])
        missing = sorted(set(pre).difference(header))
        if missing:
            raise ValueError(f"{role}: missing pre-Gold columns {missing}")
        role_index = _index_rows(rows, name=role)
        overlap = set(frozen_rows).intersection(role_index)
        if overlap:
            raise ValueError(f"frozen source roles overlap at {next(iter(overlap))}")
        frozen_rows.update(role_index)
        role_counts[role] = len(rows)

    canonical_header, canonical_rows = read_exact_csv(canonical_complete_path)
    expected_header = (*pre, *append)
    if canonical_header != expected_header:
        raise ValueError("canonical complete panel schema drift")
    canonical_index = _index_rows(canonical_rows, name="canonical complete panel")
    if set(canonical_index) != set(frozen_rows):
        raise ValueError("canonical and frozen source cell-key sets differ")
    if len(canonical_rows) != int(expected_rows):
        raise ValueError("token-exact panel row count drift")
    studies = len({key[0] for key in canonical_index})
    if studies != int(expected_studies):
        raise ValueError("token-exact panel study count drift")

    drift = {column: 0 for column in pre}
    output_rows: list[dict[str, str]] = []
    for canonical_row in canonical_rows:
        key = _key(canonical_row, name="canonical complete panel")
        source_row = frozen_rows[key]
        for column in pre:
            drift[column] += int(source_row[column] != canonical_row[column])
        output_rows.append(
            {
                **{column: source_row[column] for column in pre},
                **{column: canonical_row[column] for column in append},
            }
        )

    if any(row["candidate_weight_status"] != CANDIDATE_WEIGHT_STATUS for row in output_rows):
        raise ValueError("frozen candidate weight status drift")
    if any(row["automatic_label_or_weight_action"] != "NONE_FORBIDDEN" for row in output_rows):
        raise ValueError("canonical panel contains a label/weight action")
    if any(row["gold"] not in {"", "0", "1"} for row in output_rows):
        raise ValueError("canonical panel contains a non-binary Gold token")
    if any(
        row["teacher_reference_disagreement"] not in {"", "False", "True"}
        for row in output_rows
    ):
        raise ValueError("canonical panel contains an invalid disagreement token")

    destination = Path(output_path)
    _write_deterministic_gzip_csv(
        destination,
        header=expected_header,
        rows=output_rows,
    )
    written_header, written_rows = read_exact_csv(destination)
    if written_header != expected_header or len(written_rows) != len(output_rows):
        raise AssertionError("token-exact sidecar write/read schema or count drift")
    written_index = _index_rows(written_rows, name="token-exact sidecar")
    pre_differences = 0
    append_differences = 0
    for key, written in written_index.items():
        pre_differences += sum(
            written[column] != frozen_rows[key][column] for column in pre
        )
        append_differences += sum(
            written[column] != canonical_index[key][column] for column in append
        )
    if pre_differences or append_differences:
        raise AssertionError("token-exact sidecar value drift after write/read")

    return TokenExactPanelResult(
        rows=len(written_rows),
        studies=studies,
        source_role_rows=role_counts,
        original_complete_field_differences={
            column: count for column, count in drift.items() if count
        },
        token_exact_pre_gold_field_differences=pre_differences,
        token_exact_append_field_differences=append_differences,
    )


__all__ = [
    "GoldAnnotationsResult",
    "KEY_COLUMNS",
    "SOURCE_ROLES",
    "TokenExactPanelResult",
    "build_gold_annotations_sidecar",
    "build_token_exact_complete_panel",
    "read_exact_csv",
]
