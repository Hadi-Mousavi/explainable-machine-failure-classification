"""Offline acquisition safety tests; archive bytes are artificial test fixtures."""

import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from ai4i_pm.data import MAX_BYTES, acquire, extract_csv, load_raw
from ai4i_pm.utils import preserve_bytes, sha256_file


def archive_bytes(names: Sequence[str], content: bytes = b"artificial fixture") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, content)
    return buffer.getvalue()


def test_zip_member_bytes_preserved():
    content = b"exact bytes\r\n"
    assert extract_csv(archive_bytes(["ai4i2020.csv"], content), archived=True) == content


@pytest.mark.parametrize(
    "names",
    [
        ["../ai4i2020.csv"],
        ["/ai4i2020.csv"],
        ["nested/ai4i2020.csv"],
        ["ai4i2020.csv", "extra.txt"],
    ],
)
def test_reject_unexpected_archive_structure(names):
    with pytest.raises(ValueError, match="exactly one archive member"):
        extract_csv(archive_bytes(names), archived=True)


def test_size_limit():
    with pytest.raises(ValueError, match="safety limit"):
        extract_csv(b"x" * (MAX_BYTES + 1), archived=False)


def test_never_overwrite_different_raw_bytes(tmp_path):
    path = tmp_path / "source.csv"
    preserve_bytes(path, b"original")
    preserve_bytes(path, b"original")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        preserve_bytes(path, b"changed")
    assert path.read_bytes() == b"original"


def test_checksum_detects_tampering(research_root):
    frame, checksum = load_raw(research_root)
    raw = research_root / "data/raw/ai4i2020.csv"
    assert len(frame) == 10_000
    assert checksum == sha256_file(raw)
    with raw.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="checksum differs"):
        load_raw(research_root)


def test_manual_acquisition_idempotence_and_conflicts(research_root, tmp_path):
    official_like = research_root / "data/raw/ai4i2020.csv"
    # This tests mechanics only; it does not authenticate fixture provenance.
    destination = tmp_path / "imported_fixture"
    first = acquire(destination, official_like)
    provenance = destination / "results/01_data_exploration/provenance.json"
    original = provenance.read_bytes()
    second = acquire(destination, official_like)
    assert first == second
    assert provenance.read_bytes() == original
    assert (destination / "data/raw/ai4i2020.csv").read_bytes() == official_like.read_bytes()
    changed = tmp_path / "changed.csv"
    changed.write_bytes(official_like.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs"):
        acquire(destination, changed)


def test_acquire_refuses_existing_unrecorded_different_raw(research_root, tmp_path):
    destination = tmp_path / "conflict_fixture"
    raw = destination / "data/raw/ai4i2020.csv"
    preserve_bytes(raw, b"unrelated user file")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        acquire(destination, research_root / "data/raw/ai4i2020.csv")
    assert raw.read_bytes() == b"unrelated user file"


def test_official_download_failure_is_explicit_and_has_no_fallback(tmp_path, monkeypatch):
    def blocked(request, timeout):
        assert request.full_url.startswith("https://archive.ics.uci.edu/")
        raise OSError("ARTIFICIAL TEST: network blocked")

    monkeypatch.setattr("ai4i_pm.data.urlopen", blocked)
    with pytest.raises(RuntimeError, match="BLOCKED.*--manual"):
        acquire(tmp_path)
    assert not (tmp_path / "data/raw/ai4i2020.csv").exists()


def test_manual_zip_checksums_and_archive_tampering(research_root, tmp_path):
    csv = (research_root / "data/raw/ai4i2020.csv").read_bytes()
    supplied = tmp_path / "artificial_fixture.zip"
    supplied.write_bytes(archive_bytes(["ai4i2020.csv"], csv))
    destination = tmp_path / "zip_fixture"
    metadata = acquire(destination, supplied)
    assert metadata["archive_sha256"] == sha256_file(supplied)
    assert metadata["csv_sha256"] == sha256_file(research_root / "data/raw/ai4i2020.csv")
    assert acquire(destination) == metadata
    archive = destination / "data/raw/uci_ai4i_2020.zip"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive.*checksum"):
        acquire(destination)


def test_restore_missing_source_from_recorded_zip(research_root: Path, tmp_path: Path) -> None:
    """A fresh checkout has provenance; a partial checkout may also have the archive."""
    csv = (research_root / "data/raw/ai4i2020.csv").read_bytes()
    supplied = tmp_path / "artificial_fixture.zip"
    supplied.write_bytes(archive_bytes(["ai4i2020.csv"], csv))
    destination = tmp_path / "zip_fixture"
    metadata = acquire(destination, supplied)
    provenance = destination / "results/01_data_exploration/provenance.json"
    original_provenance = provenance.read_bytes()
    archive = destination / "data/raw/uci_ai4i_2020.zip"
    raw = destination / "data/raw/ai4i2020.csv"
    raw.unlink()
    archive.unlink()

    different_archive = tmp_path / "repacked_fixture.zip"
    different_archive.write_bytes(supplied.read_bytes() + b"repacked")
    with pytest.raises(ValueError, match="matching original UCI ZIP"):
        acquire(destination, different_archive)
    assert not archive.exists() and not raw.exists()

    assert acquire(destination, supplied) == metadata
    assert archive.read_bytes() == supplied.read_bytes()
    assert raw.read_bytes() == csv
    raw.unlink()
    assert acquire(destination) == metadata
    assert raw.read_bytes() == csv
    assert provenance.read_bytes() == original_provenance


def test_restore_missing_csv_for_csv_only_provenance(research_root: Path, tmp_path: Path) -> None:
    source = research_root / "data/raw/ai4i2020.csv"
    destination = tmp_path / "csv_fixture"
    metadata = acquire(destination, source)
    raw = destination / "data/raw/ai4i2020.csv"
    raw.unlink()

    assert acquire(destination, source) == metadata
    assert raw.read_bytes() == source.read_bytes()
    assert not (destination / "data/raw/uci_ai4i_2020.zip").exists()
