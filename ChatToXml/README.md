# Offline XML Generator (Natural Language → Valid XML)

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 0) (One-time, online) prefetch model for offline runs
python - <<'PY'
from transformers import T5Tokenizer, T5ForConditionalGeneration
T5Tokenizer.from_pretrained("t5-small")
T5ForConditionalGeneration.from_pretrained("t5-small")
PY

# 1) Synthetic data
python src/synth_data.py

# 2) Train
python src/train.py

# 3) Generate + validate with fallback
python src/generate_with_fallback.py --prompt "Create a user named Alice with ID 123 and email alice@example.com" --schema user

# 4) Export ONNX
python src/export_onnx.py

# 5) ONNX inference
python src/onnx_infer.py --prompt "Create order 555 for user Alice totaling 123.45" --schema order

# 6) Offline UI
python src/ui.py   # open http://127.0.0.1:7860
# The interface displays training metrics, generation speed, and keeps a
# history of prompts with their XML outputs for easy verification.
```

Note: Don't commit trained weights; they're ignored by .gitignore. Use Releases/artifacts if you need to share binaries internally.

## Creating a Dataset

Training data is expected in a CSV file with two columns:

| input | text_output |
|-------|-------------|
| `Create a user named Alice with ID 123` | `<user><id>123</id><name>Alice</name></user>` |

The repository includes `src/synth_data.py`, which produces a synthetic dataset of
2,500 natural-language/ XML pairs. Run:

```bash
python src/synth_data.py
```

This writes `data/sample_data.csv`. To customize the dataset, modify the
generators in `synth_data.py` or replace the CSV with your own examples following
the same schema.

## Training the Model

The model fine-tunes `t5-small` to translate natural language into XML. The
training script loads the CSV, prefixes each input with `to-xml:` and optimizes
using cross-entropy loss.

```bash
python src/train.py
```

Weights and tokenizer files are saved to the `t5-small/` directory. Adjust the
hyperparameters in `src/train.py` for larger datasets or longer training.

## Prompting the Model

Generation expects the `to-xml:` prefix followed by a clear description of the
desired structure. The helper script handles prefixing, validation against an
XSD schema, and a fallback repair step.

```bash
python src/generate_with_fallback.py --prompt "Create a user named Alice with ID 123 and email alice@example.com" --schema user
```

Direct usage from Python:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

tok = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
inputs = tok("to-xml: Create order 555 for user Alice totaling 123.45", return_tensors="pt")
ids = model.generate(**inputs, max_length=160, num_beams=4)
print(tok.decode(ids[0], skip_special_tokens=True))
```

Prompts should mention all required fields (e.g., IDs, names, totals) so the
model can populate the corresponding XML elements. If the model output fails
schema validation, `repair.py` extracts slots heuristically to produce a valid
document.

