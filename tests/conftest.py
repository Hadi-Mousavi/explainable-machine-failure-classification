"""ARTIFICIAL TEST FIXTURES ONLY; never research observations or experiment results."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai4i_pm.schema import FAILURE_MODES, SOURCE_COLUMNS
from ai4i_pm.utils import sha256_file, write_json


def make_test_frame(n_rows: int = 100) -> pd.DataFrame:
    """Build explicitly artificial, deterministic observations for offline tests."""
    rng = np.random.default_rng(123)
    ids = np.arange(1, n_rows + 1)
    types = np.array(["L", "M", "H"])[ids % 3]
    frame = pd.DataFrame(
        {
            "UDI": ids,
            "Product ID": [f"{kind}{number:05d}" for kind, number in zip(types, ids, strict=True)],
            "Type": types,
            "Air temperature [K]": rng.uniform(295, 305, n_rows),
            "Process temperature [K]": rng.uniform(305, 315, n_rows),
            "Rotational speed [rpm]": rng.integers(1200, 2000, n_rows),
            "Torque [Nm]": rng.uniform(10, 70, n_rows),
            "Tool wear [min]": rng.integers(0, 250, n_rows),
            "Machine failure": (ids % 5 == 0).astype(int),
            **{name: np.zeros(n_rows, dtype=int) for name in FAILURE_MODES},
        }
    )
    return frame.loc[:, list(SOURCE_COLUMNS)]


@pytest.fixture
def source_frame() -> pd.DataFrame:
    """A 100-row artificial schema-valid fixture."""
    return make_test_frame()


@pytest.fixture
def research_root(tmp_path: Path) -> Path:
    """A temporary 10000-row ARTIFICIAL test repository, not the real dataset."""
    raw = tmp_path / "data/raw/ai4i2020.csv"
    raw.parent.mkdir(parents=True)
    make_test_frame(10_000).to_csv(raw, index=False)
    write_json(
        tmp_path / "results/day1/provenance.json",
        {
            "csv_sha256": sha256_file(raw),
            "archive_sha256": None,
            "method": "ARTIFICIAL TEST FIXTURE; NOT RESEARCH DATA",
        },
    )
    config = tmp_path / "configs/day1.toml"
    config.parent.mkdir()
    config.write_text(
        "[split]\nseed = 42\ntest_fraction = 0.2\nvalidation_fraction_of_remainder = 0.25\n",
        encoding="utf-8",
    )
    return tmp_path
