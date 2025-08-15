import csv
import sys
from pathlib import Path

# Ensure src is on the Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from synth_data import generate_dataset


def test_generate_dataset_with_custom_values(tmp_path):
    out_file = tmp_path / "data.csv"
    generate_dataset(
        n=10,
        out_path=out_file,
        names=["Xavier"],
        products=["MegaWidget"],
        currencies=["GBP"],
        seed=0,
    )

    with out_file.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 10
    text = " ".join(r["input"] + r["text_output"] for r in rows)
    assert "Xavier" in text
    assert "MegaWidget" in text
    assert "GBP" in text

