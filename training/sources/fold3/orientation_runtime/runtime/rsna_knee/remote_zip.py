"""Selective, CRC-verified extraction from Kaggle's remote competition ZIP.

The full RSNA archive is hundreds of gigabytes.  Kaggle's signed object URL
supports HTTP byte ranges, so :class:`HttpRangeReader` makes the object look
seekable to ``zipfile`` and lets us extract only explicitly selected studies.
Signed URLs are kept in memory and are never written to an artifact.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile, ZipInfo

import pandas as pd


DEFAULT_COMPETITION = "rsna-knee-abnormality-detection"
_UID_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def get_kaggle_archive_url(competition: str = DEFAULT_COMPETITION) -> str:
    """Return a fresh signed bulk-download URL using the authenticated SDK."""

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.competitions.types.competition_api_service import (
            ApiDownloadDataFilesRequest,
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Kaggle SDK is required; install the pinned download extra first"
        ) from error

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = competition
        response = kaggle.competitions.competition_api_client.download_data_files(
            request
        )
    if not getattr(response, "url", None):
        raise RuntimeError("Kaggle did not return a competition archive URL")
    return str(response.url)


class HttpRangeReader(io.RawIOBase):
    """Read-only seekable HTTP object backed by validated Range requests."""

    def __init__(
        self,
        url_provider: Callable[[], str],
        *,
        block_size: int = 8 * 1024 * 1024,
        cache_blocks: int = 8,
        timeout_seconds: float = 60.0,
        maximum_attempts: int = 5,
        session: Any | None = None,
    ) -> None:
        super().__init__()
        if block_size <= 0 or cache_blocks <= 0:
            raise ValueError("block_size and cache_blocks must be positive")
        if maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")
        try:
            import requests
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("requests is required for remote ZIP access") from error

        self._requests = requests
        self._url_provider = url_provider
        self._block_size = int(block_size)
        self._cache_blocks = int(cache_blocks)
        self._timeout = float(timeout_seconds)
        self._maximum_attempts = int(maximum_attempts)
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._url = ""
        self._size = 0
        self._position = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self.requests_made = 0
        self.bytes_received = 0
        self.url_refreshes = 0
        self._refresh_url(initial=True)

    @property
    def size(self) -> int:
        return self._size

    def _refresh_url(self, *, initial: bool = False) -> None:
        url = self._url_provider()
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise RuntimeError("URL provider returned an invalid HTTP URL")
        response = self._session.get(
            url,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
            allow_redirects=True,
            timeout=self._timeout,
        )
        self.requests_made += 1
        if response.status_code != 206:
            raise RuntimeError(
                f"Remote archive does not support byte ranges: HTTP {response.status_code}"
            )
        match = _CONTENT_RANGE_PATTERN.match(response.headers.get("Content-Range", ""))
        if match is None or int(match.group(1)) != 0 or int(match.group(2)) != 0:
            raise RuntimeError("Remote archive returned an invalid Content-Range header")
        size = int(match.group(3))
        if self._size and size != self._size:
            raise RuntimeError("Remote archive size changed while refreshing its signed URL")
        if len(response.content) != 1:
            raise RuntimeError("Remote archive returned an invalid one-byte probe")
        self.bytes_received += len(response.content)
        self._url = response.url
        self._size = size
        if not initial:
            self.url_refreshes += 1

    def _fetch_block(self, block_index: int) -> bytes:
        cached = self._cache.get(block_index)
        if cached is not None:
            self._cache.move_to_end(block_index)
            return cached

        start = block_index * self._block_size
        if start >= self._size:
            return b""
        end = min(start + self._block_size, self._size) - 1
        expected = end - start + 1
        last_error: Exception | None = None
        for attempt in range(self._maximum_attempts):
            try:
                response = self._session.get(
                    self._url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                    allow_redirects=True,
                    timeout=self._timeout,
                )
                self.requests_made += 1
                if response.status_code in {401, 403}:
                    self._refresh_url()
                    continue
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Range {start}-{end} failed with HTTP {response.status_code}"
                    )
                match = _CONTENT_RANGE_PATTERN.match(
                    response.headers.get("Content-Range", "")
                )
                if match is None:
                    raise RuntimeError("Range response has no valid Content-Range")
                actual = tuple(int(value) for value in match.groups())
                if actual != (start, end, self._size):
                    raise RuntimeError(
                        f"Unexpected Content-Range {actual}; expected {(start, end, self._size)}"
                    )
                data = response.content
                if len(data) != expected:
                    raise RuntimeError(
                        f"Short Range response: {len(data)} bytes, expected {expected}"
                    )
                self.bytes_received += len(data)
                self._cache[block_index] = data
                self._cache.move_to_end(block_index)
                while len(self._cache) > self._cache_blocks:
                    self._cache.popitem(last=False)
                return data
            except (self._requests.RequestException, RuntimeError) as error:
                last_error = error
                if attempt + 1 < self._maximum_attempts:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"Unable to fetch remote archive range {start}-{end}"
        ) from last_error

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self._position = min(position, self._size)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if size is None or size < 0:
            size = self._size - self._position
        size = min(size, self._size - self._position)
        if size <= 0:
            return b""
        pieces: list[bytes] = []
        remaining = size
        while remaining:
            block_index, within = divmod(self._position, self._block_size)
            block = self._fetch_block(block_index)
            take = min(remaining, len(block) - within)
            if take <= 0:
                raise RuntimeError("Remote archive produced an empty block before EOF")
            pieces.append(block[within : within + take])
            self._position += take
            remaining -= take
        return b"".join(pieces)

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        if not self.closed and self._owns_session:
            self._session.close()
        super().close()


def _validate_study_ids(study_ids: Iterable[str]) -> list[str]:
    normalized = sorted({str(value).strip() for value in study_ids})
    if not normalized:
        raise ValueError("At least one StudyInstanceUID is required")
    invalid = [value for value in normalized if not _UID_PATTERN.fullmatch(value)]
    if invalid:
        raise ValueError(f"Invalid StudyInstanceUID values: {invalid[:3]}")
    return normalized


def _selected_dicom(info: ZipInfo, selected: set[str]) -> tuple[str, str, str] | None:
    path = PurePosixPath(info.filename)
    if (
        info.is_dir()
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 4
        or path.parts[0] != "train_series"
        or path.parts[1] not in selected
        or path.suffix.lower() != ".dcm"
    ):
        return None
    return path.parts[1], path.parts[2], path.parts[3]


def inventory_subset(
    archive: ZipFile,
    study_ids: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build an exact member inventory for selected studies."""

    studies = _validate_study_ids(study_ids)
    selected = set(studies)
    rows: list[dict[str, Any]] = []
    archive_members = 0
    for info in archive.infolist():
        archive_members += 1
        parsed = _selected_dicom(info, selected)
        if parsed is None:
            continue
        study_uid, series_uid, filename = parsed
        rows.append(
            {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "filename": filename,
                "member_path": info.filename,
                "uncompressed_bytes": int(info.file_size),
                "compressed_bytes": int(info.compress_size),
                "crc32": f"{info.CRC:08x}",
                "header_offset": int(info.header_offset),
                "compression_type": int(info.compress_type),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"Selected studies absent from archive: {studies[:5]}")
    frame = frame.sort_values("header_offset").reset_index(drop=True)
    found = set(frame["StudyInstanceUID"])
    missing = sorted(selected.difference(found))
    if missing:
        raise RuntimeError(f"Selected studies absent from archive: {missing[:5]}")
    if frame["member_path"].duplicated().any():
        raise RuntimeError("Remote archive contains duplicate selected member paths")
    summary = {
        "selected_studies": len(studies),
        "selected_series": int(
            frame[["StudyInstanceUID", "SeriesInstanceUID"]]
            .drop_duplicates()
            .shape[0]
        ),
        "selected_dicoms": int(len(frame)),
        "selected_uncompressed_bytes": int(frame["uncompressed_bytes"].sum()),
        "selected_compressed_bytes": int(frame["compressed_bytes"].sum()),
        "archive_members": archive_members,
        "compression_types": {
            str(int(key)): int(value)
            for key, value in frame["compression_type"].value_counts().items()
        },
    }
    return frame, summary


def _safe_destination(root: Path, member_path: str) -> Path:
    relative = PurePosixPath(member_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_path}")
    destination = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    resolved_parent = destination.parent.resolve()
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise ValueError(f"ZIP member escapes destination: {member_path}")
    return destination


def _file_crc32(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def remaining_download_bytes(
    inventory: pd.DataFrame,
    destination_root: str | Path,
    *,
    verify_existing: bool = True,
) -> tuple[int, int]:
    """Return remaining uncompressed bytes and count of verified existing files."""

    root = Path(destination_root)
    remaining = 0
    verified = 0
    for row in inventory.itertuples(index=False):
        destination = _safe_destination(root, row.member_path)
        if destination.is_file() and destination.stat().st_size == row.uncompressed_bytes:
            if not verify_existing or _file_crc32(destination) == int(row.crc32, 16):
                verified += 1
                continue
        remaining += int(row.uncompressed_bytes)
    return remaining, verified


def enforce_disk_reserve(
    destination_root: str | Path,
    *,
    remaining_bytes: int,
    minimum_free_bytes: int,
) -> dict[str, int]:
    root = Path(destination_root)
    probe = root
    while not probe.exists():
        if probe.parent == probe:
            raise RuntimeError(f"Cannot find an existing parent for {root}")
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    projected_free = free - int(remaining_bytes)
    if projected_free < minimum_free_bytes:
        raise RuntimeError(
            "Disk guard refused extraction: "
            f"{remaining_bytes / 2**30:.2f} GiB remain to write, "
            f"{free / 2**30:.2f} GiB are free, and "
            f"{minimum_free_bytes / 2**30:.2f} GiB must remain"
        )
    return {
        "free_bytes_before": int(free),
        "remaining_bytes": int(remaining_bytes),
        "projected_free_bytes": int(projected_free),
        "minimum_free_bytes": int(minimum_free_bytes),
    }


def extract_subset(
    archive: ZipFile,
    inventory: pd.DataFrame,
    destination_root: str | Path,
    *,
    minimum_free_bytes: int,
    verify_existing: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Extract selected DICOMs with CRC checks and atomic finalization."""

    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    remaining, preexisting = remaining_download_bytes(
        inventory, root, verify_existing=verify_existing
    )
    disk = enforce_disk_reserve(
        root,
        remaining_bytes=remaining,
        minimum_free_bytes=minimum_free_bytes,
    )
    started = time.monotonic()
    written_bytes = 0
    written_files = 0
    ordered = inventory.sort_values("header_offset")
    total_files = len(ordered)
    for position, row in enumerate(ordered.itertuples(index=False), start=1):
        destination = _safe_destination(root, row.member_path)
        if destination.is_file() and destination.stat().st_size == row.uncompressed_bytes:
            if not verify_existing or _file_crc32(destination) == int(row.crc32, 16):
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        checksum = 0
        size = 0
        try:
            with archive.open(row.member_path, "r") as source, temporary.open("wb") as sink:
                while chunk := source.read(4 * 1024 * 1024):
                    sink.write(chunk)
                    size += len(chunk)
                    checksum = zlib.crc32(chunk, checksum)
                sink.flush()
                os.fsync(sink.fileno())
        except (BadZipFile, OSError):
            temporary.unlink(missing_ok=True)
            raise
        checksum &= 0xFFFFFFFF
        if size != row.uncompressed_bytes or checksum != int(row.crc32, 16):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"CRC or size mismatch for {row.member_path}")
        os.replace(temporary, destination)
        written_files += 1
        written_bytes += size
        if progress is not None:
            progress(
                {
                    "position": position,
                    "total_files": total_files,
                    "written_files": written_files,
                    "written_bytes": written_bytes,
                    "elapsed_seconds": time.monotonic() - started,
                    "study_uid": row.StudyInstanceUID,
                }
            )

    final_remaining, final_verified = remaining_download_bytes(
        inventory, root, verify_existing=True
    )
    if final_remaining or final_verified != total_files:
        raise RuntimeError(
            f"Final extraction verification failed: {final_verified}/{total_files} files"
        )
    return {
        "files": total_files,
        "preexisting_verified_files": int(preexisting),
        "written_files": int(written_files),
        "written_bytes": int(written_bytes),
        "final_verified_files": int(final_verified),
        "elapsed_seconds": float(time.monotonic() - started),
        "disk_guard": disk,
    }


def open_kaggle_archive(
    competition: str = DEFAULT_COMPETITION,
    *,
    block_size: int = 8 * 1024 * 1024,
    cache_blocks: int = 8,
) -> tuple[HttpRangeReader, ZipFile]:
    provider = lambda: get_kaggle_archive_url(competition)
    reader = HttpRangeReader(
        provider,
        block_size=block_size,
        cache_blocks=cache_blocks,
    )
    try:
        archive = ZipFile(reader)
    except Exception:
        reader.close()
        raise
    return reader, archive
