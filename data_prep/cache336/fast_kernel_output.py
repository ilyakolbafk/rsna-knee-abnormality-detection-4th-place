#!/usr/bin/env python3
"""Download one Kaggle kernel output with bounded concurrent file transfers."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def list_files(kernel: str):
    owner, slug = kernel.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    token = None
    files = []
    log = None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            # Kaggle currently rejects larger pages with HTTP 400.
            request.page_size = 100
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            if response.log:
                log = response.log
            token = response.next_page_token
            if not token:
                break
    return slug, files, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("--path", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument("--read-timeout", type=float, default=25.0)
    args = ap.parse_args()

    root = Path(args.path)
    root.mkdir(parents=True, exist_ok=True)
    slug, files, log = list_files(args.kernel)
    print(json.dumps({"phase": "listed", "kernel": args.kernel, "files": len(files)}), flush=True)

    # A resumed component normally already contains CACHE_MANIFEST.csv.  It is
    # immutable and records the expected byte size for every large cache file.
    # Reuse those files without opening hundreds of signed HTTP streams merely
    # to obtain Content-Length; the final verifier still checks every SHA256.
    manifest_sizes: dict[Path, int] = {}
    manifests = list(root.glob("*/CACHE_MANIFEST.csv"))
    if len(manifests) == 1:
        manifest = manifests[0]
        with manifest.open(newline="") as fh:
            for row in csv.DictReader(fh):
                cache_file = row.get("cache_file")
                size = row.get("bytes")
                if cache_file and size:
                    manifest_sizes[manifest.parent / cache_file] = int(size)
        print(json.dumps({
            "phase": "resume_manifest",
            "kernel": args.kernel,
            "known_cache_files": len(manifest_sizes),
        }), flush=True)

    started = time.monotonic()
    completed = 0
    transferred = 0

    def download(item):
        target = root / item.file_name
        part = target.with_suffix(target.suffix + ".partial")
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest_size = manifest_sizes.get(target)
        if target.is_file() and manifest_size is not None and target.stat().st_size == manifest_size:
            return manifest_size, True
        last_error = None
        for attempt in range(args.attempts):
            try:
                # This route can leave a final TCP stream half-open.  A short
                # read timeout retries the one partial file instead of holding
                # an entire shard for five minutes per attempt.
                with requests.get(item.url, stream=True, timeout=(20, args.read_timeout)) as response:
                    response.raise_for_status()
                    expected = response.headers.get("Content-Length")
                    expected = int(expected) if expected else None
                    if target.is_file() and expected is not None and target.stat().st_size == expected:
                        return target.stat().st_size, True
                    with part.open("wb") as fh:
                        written = 0
                        for block in response.iter_content(1 << 20):
                            if block:
                                fh.write(block)
                                written += len(block)
                    if expected is not None and written != expected:
                        raise RuntimeError(f"short file {item.file_name}: {written}/{expected}")
                    os.replace(part, target)
                    return written, False
            except Exception as exc:
                last_error = exc
                part.unlink(missing_ok=True)
                time.sleep(min(30, 1.7**attempt))
        raise RuntimeError(f"{item.file_name}: {last_error}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, item): item.file_name for item in files}
        for future in as_completed(futures):
            size, reused = future.result()
            completed += 1
            if not reused:
                transferred += size
            elapsed = max(time.monotonic() - started, 1e-6)
            print(json.dumps({
                "phase": "file_complete",
                "kernel": args.kernel,
                "files_done": completed,
                "files_total": len(files),
                "transferred_bytes": transferred,
                "mib_s": round(transferred / elapsed / (1 << 20), 3),
            }), flush=True)

    if log:
        (root / f"{slug}.log").write_text(log)
    print(json.dumps({"status": "PASS", "kernel": args.kernel, "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()
