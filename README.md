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
