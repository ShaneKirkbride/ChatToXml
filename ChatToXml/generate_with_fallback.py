import argparse

from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import MODEL_DIR, SCHEMA_DIR
from repair import repair_to_schema
from xml_utils import pretty, validate_xml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--schema", choices=["user", "product", "order"], required=True)
    args = ap.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    inputs = tokenizer(f"to-xml: {args.prompt}", return_tensors="pt")
    output_ids = model.generate(**inputs, max_length=160, num_beams=4)
    xml = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    xsd_path = str((SCHEMA_DIR / f"{args.schema}.xsd").resolve())
    ok, err = validate_xml(xml, xsd_path)
    if not ok:
        repaired = repair_to_schema(args.prompt, args.schema)
        ok2, err2 = validate_xml(repaired, xsd_path)
        if not ok2:
            print("Model output AND repair failed validation.")
            print("Model error:", err)
            print("Repair error:", err2)
            print("\nModel XML:\n", xml)
            print("\nRepaired XML:\n", repaired)
            raise SystemExit(1)
        print(pretty(repaired))
    else:
        print(pretty(xml))


if __name__ == "__main__":
    main()
