# ABOUTME: Checks the published FD001 deep learning table against the report bibliography.
# ABOUTME: Every row needs a cited source, and the rows shared with Phase 1 must agree.
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_table() -> pd.DataFrame:
    """Read results/dl_published.csv."""
    return pd.read_csv(ROOT / "results" / "dl_published.csv")


def bib_keys() -> set[str]:
    """Return every entry key in report/refs.bib."""
    text = (ROOT / "report" / "refs.bib").read_text()
    return set(re.findall(r"^@\w+\{([^,]+),", text, flags=re.MULTILINE))


def test_every_row_cites_a_key_in_refs_bib():
    assert set(load_table().bib_key) <= bib_keys()


def test_bib_keys_are_unique():
    text = (ROOT / "report" / "refs.bib").read_text()
    keys = re.findall(r"^@\w+\{([^,]+),", text, flags=re.MULTILINE)
    assert len(keys) == len(set(keys))


def test_rows_are_complete_and_plausible():
    table = load_table()
    assert table.method.is_unique
    assert table.rmse.notna().all() and table.rmse.between(5, 50).all()
    assert table.values_from.notna().all()
    assert table.rul_cap.dropna().isin([125, 130]).all()


def test_rows_shared_with_phase_1_match_its_comparison():
    phase1 = pd.read_csv(ROOT / "results" / "test_published_comparison.csv")
    phase1 = phase1[phase1["where"] == "published"].set_index(["method", "source"])
    shared = load_table().set_index(["method", "source"]).join(
        phase1, how="inner", rsuffix="_phase1")
    assert len(shared) == 10
    assert (shared.rmse == shared.rmse_phase1).all()
    assert (shared.nasa_score == shared.nasa_score_phase1).all()
