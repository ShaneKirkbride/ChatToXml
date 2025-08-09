from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "sample_data.csv"
MODEL_DIR = ROOT / "models" / "t5-small-finetuned"
SCHEMA_DIR = ROOT / "schema"
