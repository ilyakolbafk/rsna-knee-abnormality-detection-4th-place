"""Download private Kaggle cache outputs and materialize one verified v8.3 release."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_BUILDER = "935625ae385df67cba32d11aca08a3cce7f3ec27d42313be79edc0236e9c483a"
EXPECTED_POLICY = "knee-laterality-v8.3"
EXPECTED_STATE = "canonical_L_v8_3"
KNOWN_SOURCE_ERRORS = {
    8: "1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035",
    9: "1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680",
}
EXPECTED = {
    0: (254, 254), 1: (306, 306), 2: (254, 254), 3: (272, 273),
    4: (275, 276), 5: (283, 283), 6: (280, 280), 7: (260, 262),
    8: (261, 261), 9: (272, 272), 10: (275, 275), 11: (276, 276),
    12: (262, 263), 13: (263, 263), 14: (271, 272), 15: (281, 282),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def kernel_id(shard):
    return f"ilyakolb/rsna-v83-aux60-cache-s{shard:02d}-20260915"


def download_one(root, role, identifier, file_workers, attempts=8):
    destination = root / "downloads" / role
    marker = destination / "DOWNLOAD_COMPLETE.json"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("fast_kernel_output.py")),
        identifier,
        "--path", str(destination),
        "--workers", str(file_workers),
    ]
    failure = None
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command, capture_output=True, text=True)
        complete = list(destination.rglob("CACHE_COMPLETE.json"))
        # Kaggle can finish the requested downloads and then return non-zero
        # while fetching the optional log.  The component verifier below is
        # the authority, so a complete output tree is sufficient here.
        if len(complete) == 1:
            try:
                component_root, _complete, manifest = check_component(
                    destination, None if role == "gold58" else int(role[1:])
                )
                missing_or_short = [
                    str(component_root / row.cache_file)
                    for row in manifest.itertuples(index=False)
                    if not (component_root / row.cache_file).is_file()
                    or (component_root / row.cache_file).stat().st_size != int(row.bytes)
                ]
                if missing_or_short:
                    raise ValueError(f"missing or size-mismatched NPZ files: {missing_or_short[:3]}")
                failure = None
                break
            except Exception as exc:
                failure = f"component incomplete after attempt {attempt}: {exc}"
        else:
            failure = (
                f"kaggle exit={result.returncode}; CACHE_COMPLETE count={len(complete)}; "
                f"stderr={result.stderr[-500:]}"
            )
        if attempt < attempts:
            time.sleep(min(45, 5 * 2 ** (attempt - 1)))
    if failure is not None:
        raise RuntimeError(f"Unable to download {identifier}: {failure}")
    atomic_json(marker, {"kernel": identifier, "downloaded_at": time.time(), "cache_complete": str(complete[0])})
    return destination


def check_component(directory, shard=None):
    complete_path = next(directory.rglob("CACHE_COMPLETE.json"))
    component_root = complete_path.parent
    complete = json.loads(complete_path.read_text())
    manifest = pd.read_csv(component_root / "CACHE_MANIFEST.csv")
    if complete["status"] != "PASS_PIXEL_CACHE":
        raise ValueError(f"Cache component failed: {complete_path}")
    for key, want in [("builder_sha256", EXPECTED_BUILDER), ("orientation_policy", EXPECTED_POLICY), ("orientation_state", EXPECTED_STATE)]:
        if complete.get(key) != want:
            raise ValueError(f"{complete_path}: {key} mismatch")
    if sha(component_root / "CACHE_MANIFEST.csv") != complete["cache_manifest_sha256"]:
        raise ValueError(f"Manifest hash mismatch: {complete_path}")
    if shard is None:
        if complete["gold_studies"] != 58 or complete["gold_samples"] != 58 or complete["gold_train_mask"] != 0:
            raise ValueError("Gold58 component cardinality/mask failure")
        if complete["train_samples"] != 0 or set(manifest.namespace) != {"gold58"}:
            raise ValueError("Gold58 namespace failure")
        if sha(component_root / "GOLD58_LABELS.csv") != complete["gold_labels_sha256"]:
            raise ValueError("Gold58 label hash mismatch")
    else:
        parents, samples = EXPECTED[shard]
        if complete["parent_shard_id"] != shard or complete["train_parents"] != parents or complete["train_samples"] != samples:
            raise ValueError(f"Shard {shard} cardinality failure")
        if len(manifest) != samples or set(manifest.namespace) != {"train"}:
            raise ValueError(f"Shard {shard} namespace/row failure")
        if sha(component_root / "TRAIN_LABELS.csv") != complete["train_labels_sha256"]:
            raise ValueError(f"Shard {shard} label hash mismatch")
        errors = manifest[manifest.source_errors.astype(int) > 0]
        if shard in KNOWN_SOURCE_ERRORS:
            if int(manifest.source_errors.sum()) != 1 or errors.StudyInstanceUID.tolist() != [KNOWN_SOURCE_ERRORS[shard]]:
                raise ValueError(f"Shard {shard} does not match its frozen one-frame source exception")
        elif int(manifest.source_errors.sum()) != 0:
            raise ValueError(f"Unexpected source decode errors in shard {shard}")
    return component_root, complete, manifest


def verify_file(item):
    path, want = item
    if not path.is_file():
        return str(path), "MISSING", want
    got = sha(path)
    return str(path), got, want


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--download-workers", type=int, default=5)
    parser.add_argument("--file-workers", type=int, default=32)
    parser.add_argument("--hash-workers", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(f"s{i:02d}", kernel_id(i)) for i in range(16)]
    jobs.append(("gold58", "ilyakolb/rsna-v83-aux60-cache-gold58-20260915"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {
            pool.submit(download_one, root, role, identifier, args.file_workers): role
            for role, identifier in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            role = futures[future]
            future.result()
            print(json.dumps({"phase": "download_complete", "role": role}), flush=True)

    train_manifests, train_labels, components = [], [], []
    for shard in range(16):
        directory = root / "downloads" / f"s{shard:02d}"
        component_root, complete, manifest = check_component(directory, shard)
        manifest = manifest.copy()
        manifest["cache_file"] = manifest.cache_file.map(lambda p: str((component_root / p).relative_to(root)))
        train_manifests.append(manifest)
        labels = pd.read_csv(component_root / "TRAIN_LABELS.csv")
        train_labels.append(labels)
        components.append({
            "role": f"s{shard:02d}",
            "kernel": kernel_id(shard),
            "cache_complete_sha256": sha(component_root / "CACHE_COMPLETE.json"),
            "manifest_sha256": complete["cache_manifest_sha256"],
            "cache_bytes": int(complete["cache_bytes"]),
        })
    gold_root, gold_complete, gold_manifest = check_component(root / "downloads" / "gold58", None)
    gold_manifest = gold_manifest.copy()
    gold_manifest["cache_file"] = gold_manifest.cache_file.map(lambda p: str((gold_root / p).relative_to(root)))
    gold_labels = pd.read_csv(gold_root / "GOLD58_LABELS.csv")
    components.append({
        "role": "gold58",
        "kernel": "ilyakolb/rsna-v83-aux60-cache-gold58-20260915",
        "cache_complete_sha256": sha(gold_root / "CACHE_COMPLETE.json"),
        "manifest_sha256": gold_complete["cache_manifest_sha256"],
        "cache_bytes": int(gold_complete["cache_bytes"]),
    })

    train_manifest = pd.concat(train_manifests, ignore_index=True)
    labels = pd.concat(train_labels, ignore_index=True)
    if len(train_manifest) != 4352 or train_manifest.SampleID.nunique() != 4352:
        raise ValueError("Combined train manifest must have 4,352 unique samples")
    if train_manifest.StudyInstanceUID.nunique() != 4345:
        raise ValueError("Combined train manifest must have 4,345 parents")
    if len(labels) != 4352 or labels.SampleID.nunique() != 4352:
        raise ValueError("Combined labels must have 4,352 unique samples")
    if set(train_manifest.SampleID) != set(labels.SampleID):
        raise ValueError("Manifest/label sample mismatch")
    if len(gold_manifest) != 58 or gold_manifest.StudyInstanceUID.nunique() != 58 or len(gold_labels) != 58:
        raise ValueError("Gold58 combined cardinality failure")
    if set(train_manifest.StudyInstanceUID) & set(gold_manifest.StudyInstanceUID):
        raise ValueError("Train/Gold overlap")
    if not ((train_manifest.tokens + train_manifest.padded_tokens) == 64).all():
        raise ValueError("Train token-width drift")
    if not ((gold_manifest.tokens + gold_manifest.padded_tokens) == 64).all():
        raise ValueError("Gold token-width drift")

    items = []
    for manifest in (train_manifest, gold_manifest):
        for row in manifest.itertuples(index=False):
            items.append((root / row.cache_file, row.sha256))
    mismatches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.hash_workers) as pool:
        for path, got, want in pool.map(verify_file, items):
            if got != want:
                mismatches.append({"path": path, "got": got, "want": want})
    if mismatches:
        raise ValueError(f"Cache file hash mismatch: {mismatches[:3]}")

    train_manifest.to_csv(root / "TRAIN_MANIFEST.csv", index=False)
    labels.to_csv(root / "TRAIN_LABELS.csv", index=False)
    gold_manifest.to_csv(root / "GOLD58_MANIFEST.csv", index=False)
    gold_labels.to_csv(root / "GOLD58_LABELS.csv", index=False)
    receipt = {
        "status": "PASS_GLOBAL_AUX60_V83_RELEASE",
        "verified_at": time.time(),
        "builder_sha256": EXPECTED_BUILDER,
        "orientation_policy": EXPECTED_POLICY,
        "orientation_state": EXPECTED_STATE,
        "train_parents": 4345,
        "train_samples": 4352,
        "active_main_label_samples": int(labels[["label_mask__" + t for t in [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
            "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
        ]]].astype(bool).any(axis=1).sum()),
        "gold_studies": 58,
        "gold_train_mask": 0,
        "train_gold_overlap": 0,
        "verified_npz_files": len(items),
        "verified_npz_bytes": int(sum((root / path).stat().st_size for path in pd.concat([train_manifest.cache_file, gold_manifest.cache_file]))),
        "known_source_frame_exceptions": [
            {"study": study, "count": 1, "shard": shard}
            for shard, study in sorted(KNOWN_SOURCE_ERRORS.items())
        ],
        "train_manifest_sha256": sha(root / "TRAIN_MANIFEST.csv"),
        "train_labels_sha256": sha(root / "TRAIN_LABELS.csv"),
        "gold_manifest_sha256": sha(root / "GOLD58_MANIFEST.csv"),
        "gold_labels_sha256": sha(root / "GOLD58_LABELS.csv"),
        "components": components,
    }
    if receipt["active_main_label_samples"] != 4350:
        raise ValueError(f"Expected 4,350 active main-label samples, got {receipt['active_main_label_samples']}")
    atomic_json(root / "GLOBAL_RELEASE_RECEIPT.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
