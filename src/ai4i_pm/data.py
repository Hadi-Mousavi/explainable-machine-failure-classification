"""Official UCI acquisition, safe member reading, and immutable raw bytes."""

import io
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from ai4i_pm.schema import validate_source_schema
from ai4i_pm.utils import json_bytes, preserve_bytes, read_json, sha256_bytes, sha256_file, utc_now

SOURCE_URL = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
)
DOI = "https://doi.org/10.24432/C5HS5C"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
MAX_BYTES = 10_000_000


def extract_csv(payload: bytes, *, archived: bool) -> bytes:
    """Read exactly ai4i2020.csv in memory; never extract paths to disk."""
    if len(payload) > MAX_BYTES:
        raise ValueError("Source exceeds the 10 MB safety limit")
    if not archived:
        return payload
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != "ai4i2020.csv":
            raise ValueError("Expected exactly one archive member named ai4i2020.csv")
        member = members[0]
        if member.is_dir() or member.file_size > MAX_BYTES:
            raise ValueError("Invalid or oversized CSV member")
        return archive.read(member)


def _source_payload(manual: Path | None) -> tuple[bytes, bool]:
    """Read bounded bytes from the official HTTPS endpoint or an explicit import."""
    if manual is not None:
        suffix = manual.suffix.lower()
        if suffix not in {".csv", ".zip"}:
            raise ValueError("Manual import accepts an official UCI .csv or .zip only")
        with manual.open("rb") as stream:
            return stream.read(MAX_BYTES + 1), suffix == ".zip"

    request = Request(ARCHIVE_URL, headers={"User-Agent": "ai4i-pm/0.1 (academic study)"})
    try:
        with urlopen(request, timeout=60) as response:
            final = urlparse(response.url)
            if final.scheme != "https" or final.hostname != "archive.ics.uci.edu":
                raise ValueError("Download redirected outside the official HTTPS UCI host")
            return response.read(MAX_BYTES + 1), True
    except OSError as exc:
        raise RuntimeError(
            "Official UCI download BLOCKED. Download from the documented UCI source, "
            "then run acquire --manual PATH_TO_UCI_ZIP_OR_CSV."
        ) from exc


def _restore_source(
    root: Path, raw_path: Path, archive_path: Path, manual: Path | None
) -> dict[str, Any]:
    """Verify existing files and restore missing bytes against recorded digests."""
    metadata = read_json(root / "results/01_data_exploration/provenance.json")
    has_raw = raw_path.exists()
    expected_archive = metadata["archive_sha256"]
    has_archive = bool(expected_archive and archive_path.exists())
    if has_raw:
        load_raw(root)
    if has_archive and sha256_file(archive_path) != expected_archive:
        raise ValueError("Recorded source archive has a changed checksum")
    need_archive = bool(expected_archive and not has_archive)
    if manual is None and has_raw and not need_archive:
        return metadata

    if manual is None and has_archive:
        payload, archived = _source_payload(archive_path)
    else:
        payload, archived = _source_payload(manual)
    csv = extract_csv(payload, archived=archived)
    if sha256_bytes(csv) != metadata["csv_sha256"]:
        raise ValueError("Supplied source differs from the recorded raw dataset")
    if need_archive and (not archived or sha256_bytes(payload) != expected_archive):
        raise ValueError("Restoring the recorded source requires the matching original UCI ZIP")
    validate_source_schema(pd.read_csv(io.BytesIO(csv)))
    if need_archive:
        preserve_bytes(archive_path, payload)
    if not has_raw:
        preserve_bytes(raw_path, csv)
    return metadata


def acquire(root: Path, manual: Path | None = None) -> dict[str, Any]:
    """Acquire official bytes or restore an original acquisition from its provenance."""
    provenance_path = root / "results/01_data_exploration/provenance.json"
    raw_path = root / "data/raw/ai4i2020.csv"
    archive_path = root / "data/raw/uci_ai4i_2020.zip"
    if provenance_path.exists():
        return _restore_source(root, raw_path, archive_path, manual)

    payload, archived = _source_payload(manual)
    csv = extract_csv(payload, archived=archived)
    frame = pd.read_csv(io.BytesIO(csv))
    validate_source_schema(frame)
    if frame.shape != (10_000, 14):
        raise ValueError(f"Unexpected official source dimensions: {frame.shape}")
    metadata = {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "uci_id": 601,
        "source_url": SOURCE_URL,
        "download_url": ARCHIVE_URL if archived else None,
        "doi": DOI,
        "license": "CC BY 4.0",
        "license_url": LICENSE_URL,
        "attribution": "AI4I 2020 Predictive Maintenance Dataset (2020), UCI Repository",
        "acquired_at_utc": utc_now(),
        "method": "manual_official_source_attested_by_user" if manual else "official_https",
        "source_authenticity": "user-attested" if manual else "downloaded from official UCI HTTPS",
        "csv_file": "data/raw/ai4i2020.csv",
        "csv_sha256": sha256_bytes(csv),
        "archive_file": "data/raw/uci_ai4i_2020.zip" if archived else None,
        "archive_sha256": sha256_bytes(payload) if archived else None,
        "archive_members": ["ai4i2020.csv"] if archived else None,
        "rows": len(frame),
        "columns": list(frame.columns),
        "raw_bytes_modified": False,
        "synthetic": True,
    }
    # Check all existing destinations before creating any output.
    outputs = [(raw_path, csv)]
    if archived:
        outputs.append((archive_path, payload))
    for path, content in outputs:
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite existing raw source: {path}")
    for path, content in outputs:
        preserve_bytes(path, content)
    preserve_bytes(provenance_path, json_bytes(metadata))
    return metadata


def load_raw(root: Path) -> tuple[pd.DataFrame, str]:
    """Require provenance and verify original byte checksum before reading data."""
    metadata = read_json(root / "results/01_data_exploration/provenance.json")
    path = root / "data/raw/ai4i2020.csv"
    checksum = sha256_file(path)
    if checksum != metadata["csv_sha256"]:
        raise ValueError("Raw CSV checksum differs from provenance; analysis refused")
    frame = pd.read_csv(path)
    validate_source_schema(frame)
    return frame, checksum
