#!/usr/bin/env python3
"""Render frozen 30-study V62MB web prompts from the stress29 protocol."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / ".codex/runs/web-mri-v62mb-stress29-20260926/PROMPT_TEMPLATE_STRESS29.md"
READER = "chatgpt-web-v62mb-train4407"


def batch_prompt(template: str, batch: dict) -> str:
    case_order = batch["case_order"]
    count = len(case_order)
    if count < 1 or count > 30 or len(set(case_order)) != count:
        raise ValueError("invalid case count/order")
    bundles = [f"{b['bundle_id']}.zip" for b in batch["bundles"]]
    if not bundles or len(bundles) > 4 or sum(len(b["case_order"]) for b in batch["bundles"]) != count:
        raise ValueError("invalid bundle coverage")
    flat = [case_id for bundle in batch["bundles"] for case_id in bundle["case_order"]]
    if flat != case_order:
        raise ValueError("bundle order differs from batch order")
    prompt = template
    replacements = {
        "29 blinded knee MRI studies, V62MB web stress test":
            f"{count} blinded knee MRI studies, V62MB web annotation",
        "four ZIP bundles: {BUNDLE_LIST}. Together they contain exactly 29 separate studies.":
            f"{len(bundles)} ZIP bundles: {{BUNDLE_LIST}}. Together they contain exactly {count} separate studies.",
        "cases/G###/": "cases/T####/",
        "The stress test values honest coverage over a full-looking response.":
            "Truthful coverage takes priority over a full-looking response.",
        'reader: "chatgpt-web-v62mb-stress29"':
            f'reader: "{READER}"',
        "exactly 29 entries in that order": f"exactly {count} entries in that order",
        "exactly 29 result entries": f"exactly {count} result entries",
        "G###": "T####",
    }
    for old, new in replacements.items():
        if prompt.count(old) != 1:
            raise ValueError(f"source prompt anchor missing or ambiguous: {old}")
        prompt = prompt.replace(old, new)
    replacements = {
        "{BATCH_ID}": batch["batch_id"],
        "{BUNDLE_LIST}": ", ".join(f"`{name}`" for name in bundles),
        "{CASE_ORDER}": " → ".join(case_order),
        "{CASE_ORDER_JSON}": json.dumps(case_order, ensure_ascii=False),
    }
    for old, new in replacements.items():
        if old not in prompt:
            raise ValueError(f"missing render placeholder {old}")
        prompt = prompt.replace(old, new)
    if "{BATCH_" in prompt or "{CASE_" in prompt or "{BUNDLE_" in prompt:
        raise ValueError("unrendered placeholder")
    return prompt


def main(number: int, output_manifest: Path | None = None) -> None:
    shard = json.loads((HERE / f"SHARD_{number:02d}.json").read_text())
    template = SOURCE.read_text()
    rendered = None
    if output_manifest is not None:
        output = json.loads(output_manifest.read_text())
        if output["shard_id"] != shard["shard_id"] or output["status"] not in {"COMPLETE", "COMPLETE_WITH_RENDER_FAILURES"}:
            raise ValueError("Kaggle output manifest does not match frozen shard")
        rendered = {case["case_id"] for case in output["cases"]}
        if not rendered <= {case["case_id"] for case in shard["cases"]}:
            raise ValueError("Kaggle output contains unknown case IDs")
    folder = HERE / (f"prompts_shard{number:02d}_actual" if rendered is not None else f"prompts_shard{number:02d}")
    folder.mkdir(exist_ok=True)
    receipts = []
    for original_batch in shard["batches"]:
        if rendered is None:
            batch = original_batch
        else:
            batch = {"batch_id": original_batch["batch_id"],
                     "case_order": [case for case in original_batch["case_order"] if case in rendered],
                     "bundles": [
                         {"bundle_id": bundle["bundle_id"],
                          "case_order": [case for case in bundle["case_order"] if case in rendered]}
                         for bundle in original_batch["bundles"]
                         if any(case in rendered for case in bundle["case_order"])
                     ]}
            if not batch["case_order"]:
                continue
        body = batch_prompt(template, batch)
        path = folder / f"{batch['batch_id']}_PROMPT.md"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(body)
        receipts.append({"batch_id": batch["batch_id"], "case_count": len(batch["case_order"]),
                         "bundle_names": [f"{x['bundle_id']}.zip" for x in batch["bundles"]],
                         "path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()})
    manifest = folder / "PROMPT_MANIFEST.json"
    manifest.write_text(json.dumps({"status": "LOCAL_DRAFT_NOT_SENT",
                                    "source_template_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
                                    "reader": READER, "batches": receipts}, indent=2) + "\n")
    print(json.dumps({"shard": shard["shard_id"], "prompt_count": len(receipts),
                      "study_count": sum(x["case_count"] for x in receipts),
                      "manifest": str(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True, choices=range(1, 9))
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()
    main(args.shard, args.output_manifest)
