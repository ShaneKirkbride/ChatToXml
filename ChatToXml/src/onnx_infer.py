import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForSeq2Seq
from transformers import T5Tokenizer

from config import MODEL_DIR, SCHEMA_DIR
from xml_utils import pretty, validate_xml

ONNX_DIR = Path(MODEL_DIR).parent / "onnx"


def load_backend() -> tuple[ORTModelForSeq2Seq, T5Tokenizer]:
    tok = T5Tokenizer.from_pretrained(str(ONNX_DIR))
    model_path = ONNX_DIR / "model-quant.onnx"
    if not model_path.exists():
        model_path = ONNX_DIR / "model.onnx"
    mdl = ORTModelForSeq2Seq.from_pretrained(
        str(ONNX_DIR), file_name=model_path.name
    )
    return mdl, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--schema", choices=["user", "product", "order"], required=True)
    args = ap.parse_args()
    model, tok = load_backend()
    inputs = tok(f"to-xml: {args.prompt}", return_tensors="pt")
    out_ids = model.generate(**inputs, max_length=160, num_beams=4)
    xml = tok.decode(out_ids[0], skip_special_tokens=True)
    xsd = str((SCHEMA_DIR / f"{args.schema}.xsd").resolve())
    ok, err = validate_xml(xml, xsd)
    if not ok:
        print("INVALID:", err, "\n\nXML:\n", xml)
        raise SystemExit(1)
    print(pretty(xml))


if __name__ == "__main__":
    main()
