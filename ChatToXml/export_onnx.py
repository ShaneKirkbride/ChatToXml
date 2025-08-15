from pathlib import Path

from optimum.onnxruntime import ORTModelForSeq2Seq
from transformers import T5Tokenizer

from config import MODEL_DIR

ONNX_DIR = Path(MODEL_DIR).parent / "onnx"


def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    model_id = str(MODEL_DIR)
    tok = T5Tokenizer.from_pretrained(model_id)
    ORTModelForSeq2Seq.from_pretrained(
        model_id, export=True, from_transformers=True, save_dir=str(ONNX_DIR)
    )
    tok.save_pretrained(str(ONNX_DIR))
    print(f"Exported ONNX to {ONNX_DIR}")


if __name__ == "__main__":
    main()
