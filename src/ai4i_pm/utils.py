"""Small deterministic serialization and checksum helpers."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 of exact bytes."""
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a local file without parsing or rewriting it."""
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    """Canonical project JSON; refuse nonfinite numbers."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def preserve_bytes(path: Path, content: bytes) -> None:
    """Create exclusively or verify identical bytes; never overwrite existing content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite different existing content: {path}")
        return
    with path.open("xb") as stream:
        stream.write(content)


def utc_now() -> str:
    """Record actual acquisition/override time with an explicit UTC offset."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def environment_versions() -> dict[str, str]:
    """Record the runtime of a generated research artifact."""
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in ("ai4i-pm", "numpy", "pandas", "scikit-learn")},
    }
