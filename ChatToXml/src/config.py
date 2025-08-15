from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "sample_data.csv"
MODEL_DIR = ROOT  / "t5-small"
SCHEMA_DIR = ROOT / "schema"
