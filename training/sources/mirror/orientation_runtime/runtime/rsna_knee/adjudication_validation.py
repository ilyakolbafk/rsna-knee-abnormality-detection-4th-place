"""Validation and conservative merge for blinded report-only reader passes."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ASSERTIONS = {"positive", "negative", "uncertain", "not_mentioned", "abstain"}
POLARITIES = {"positive", "negative", "mixed", "uncertain", "not_applicable"}
CRITERIA_MATCHES = {"meets", "below", "unclear", "not_assessable"}
EXPECTED_CRITERIA_BY_ASSERTION = {
    "positive": "meets",
    "negative": "below",
    "uncertain": "unclear",
    "not_mentioned": "not_assessable",
}
REQUIRED_RESULT_FIELDS = {
    "case_key",
    "assertion",
    "evidence_spans",
    "section",
    "polarity",
    "uncertainty",
    "severity",
    "anatomy",
    "criteria_match",
    "abstain_reason",
    "confidence",
}
REQUIRED_SPAN_FIELDS = {
    "char_start",
    "char_end",
    "byte_start",
    "byte_end",
    "text",
    "sha256",
}


def validate_exact_span(report: str, span: Mapping[str, Any]) -> None:
    if set(span) != REQUIRED_SPAN_FIELDS:
        raise ValueError(f"Span fields drift: {sorted(set(span) ^ REQUIRED_SPAN_FIELDS)}")
    try:
        char_start = int(span["char_start"])
        char_end = int(span["char_end"])
        byte_start = int(span["byte_start"])
        byte_end = int(span["byte_end"])
        text = str(span["text"])
        claimed_sha = str(span["sha256"])
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("Malformed exact span") from error
    if not (0 <= char_start < char_end <= len(report)):
        raise ValueError("Invalid character span bounds")
    if report[char_start:char_end] != text:
        raise ValueError("Character span does not reproduce exact text")
    report_bytes = report.encode("utf-8")
    if not (0 <= byte_start < byte_end <= len(report_bytes)):
        raise ValueError("Invalid UTF-8 byte span bounds")
    try:
        byte_text = report_bytes[byte_start:byte_end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Byte span is not aligned to UTF-8 boundaries") from error
    if byte_text != text:
        raise ValueError("UTF-8 byte span does not reproduce exact text")
    if byte_start != len(report[:char_start].encode("utf-8")):
        raise ValueError("byte_start differs from the recomputed offset")
    if byte_end != len(report[:char_end].encode("utf-8")):
        raise ValueError("byte_end differs from the recomputed offset")
    actual_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if claimed_sha != actual_sha:
        raise ValueError("Exact span SHA-256 mismatch")


def validate_reader_results(
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    reader_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate exact key coverage, schema fields, values, and every span."""

    case_keys = [str(case["case_key"]) for case in cases]
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("Input cases contain duplicate case_key values")
    result_keys = [str(result.get("case_key", "")) for result in results]
    if len(result_keys) != len(set(result_keys)):
        raise ValueError(f"Reader {reader_id} contains duplicate case_key values")
    if result_keys != case_keys:
        raise ValueError(f"Reader {reader_id} case keys or order differ from input")

    validated: list[dict[str, Any]] = []
    assertion_counts: dict[str, int] = {value: 0 for value in sorted(ASSERTIONS)}
    span_count = 0
    for position, (case, raw_result) in enumerate(zip(cases, results, strict=True)):
        result = dict(raw_result)
        if set(result) != REQUIRED_RESULT_FIELDS:
            raise ValueError(
                f"Reader {reader_id} result {position} field drift: "
                f"{sorted(set(result) ^ REQUIRED_RESULT_FIELDS)}"
            )
        assertion = result["assertion"]
        polarity = result["polarity"]
        criteria = result["criteria_match"]
        spans = result["evidence_spans"]
        confidence = result["confidence"]
        if assertion not in ASSERTIONS:
            raise ValueError(f"Reader {reader_id} result {position} invalid assertion")
        if polarity not in POLARITIES:
            raise ValueError(f"Reader {reader_id} result {position} invalid polarity")
        if criteria not in CRITERIA_MATCHES:
            raise ValueError(f"Reader {reader_id} result {position} invalid criteria")
        expected_criteria = EXPECTED_CRITERIA_BY_ASSERTION.get(str(assertion))
        if expected_criteria is not None and criteria != expected_criteria:
            raise ValueError(
                f"Reader {reader_id} result {position} assertion/criteria mismatch"
            )
        if not isinstance(spans, list):
            raise ValueError(f"Reader {reader_id} result {position} spans are not a list")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"Reader {reader_id} result {position} confidence is not numeric")
        if not np.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"Reader {reader_id} result {position} confidence is out of bounds")
        for field in ("section", "uncertainty"):
            if not isinstance(result[field], str):
                raise ValueError(f"Reader {reader_id} result {position} {field} is not text")
        for field in ("severity", "anatomy", "abstain_reason"):
            if result[field] is not None and not isinstance(result[field], str):
                raise ValueError(f"Reader {reader_id} result {position} {field} has invalid type")

        if assertion in {"positive", "negative", "uncertain"} and not spans:
            raise ValueError(
                f"Reader {reader_id} result {position} semantic decision has no exact span"
            )
        if assertion == "not_mentioned" and spans:
            raise ValueError(
                f"Reader {reader_id} result {position} not_mentioned has evidence spans"
            )
        if assertion == "abstain" and not result["abstain_reason"]:
            raise ValueError(
                f"Reader {reader_id} result {position} abstain has no reason"
            )
        if assertion != "abstain" and result["abstain_reason"] is not None:
            raise ValueError(
                f"Reader {reader_id} result {position} non-abstain has abstain_reason"
            )
        for span in spans:
            if not isinstance(span, Mapping):
                raise ValueError(f"Reader {reader_id} result {position} span is not an object")
            validate_exact_span(str(case["raw_report"]), span)
        assertion_counts[str(assertion)] += 1
        span_count += len(spans)
        validated.append(result)

    diagnostics = {
        "reader_id": reader_id,
        "status": "PASS",
        "cases": len(validated),
        "assertion_counts": assertion_counts,
        "exact_spans": span_count,
        "case_key_order_exact": True,
        "schema_exact": True,
        "all_spans_exact": True,
    }
    return validated, diagnostics


def merge_reader_results(
    cases: Sequence[Mapping[str, Any]],
    reader_a: Sequence[Mapping[str, Any]],
    reader_b: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Freeze a strict agreement decision; disagreement always abstains."""

    if not (len(cases) == len(reader_a) == len(reader_b)):
        raise ValueError("Reader merge length mismatch")
    rows: list[dict[str, Any]] = []
    for case, left, right in zip(cases, reader_a, reader_b, strict=True):
        if left["case_key"] != case["case_key"] or right["case_key"] != case["case_key"]:
            raise ValueError("Reader merge case-key mismatch")
        assertions_agree = left["assertion"] == right["assertion"]
        criteria_agree = left["criteria_match"] == right["criteria_match"]
        if (
            assertions_agree
            and criteria_agree
            and left["assertion"] != "abstain"
        ):
            decision = str(left["assertion"])
            reason = "two_blind_passes_assertion_and_criteria_agree"
        else:
            decision = "abstain"
            if not assertions_agree:
                reason = "reader_assertion_disagreement"
            elif not criteria_agree:
                reason = "reader_criteria_disagreement"
            else:
                reason = "one_or_both_reader_abstain"
        rows.append(
            {
                "case_key": case["case_key"],
                "target": case["target"],
                "rubric_scope": case["rubric_scope"],
                "reader_a_assertion": left["assertion"],
                "reader_b_assertion": right["assertion"],
                "reader_a_criteria_match": left["criteria_match"],
                "reader_b_criteria_match": right["criteria_match"],
                "reader_a_confidence": float(left["confidence"]),
                "reader_b_confidence": float(right["confidence"]),
                "reader_assertions_agree": bool(assertions_agree),
                "reader_criteria_agree": bool(criteria_agree),
                "frozen_report_only_decision": decision,
                "frozen_decision_reason": reason,
                "reader_a_evidence_spans_json": json.dumps(
                    left["evidence_spans"], ensure_ascii=False, separators=(",", ":")
                ),
                "reader_b_evidence_spans_json": json.dumps(
                    right["evidence_spans"], ensure_ascii=False, separators=(",", ":")
                ),
                "same_model_family_two_isolated_passes": True,
                "independent_model_families": False,
                "can_be_called_ground_truth": False,
            }
        )
    return pd.DataFrame(rows)


def read_jsonl(lines: Iterable[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in lines if line.strip()]
