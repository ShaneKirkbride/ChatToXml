import json
import time
from typing import List

import gradio as gr
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import MODEL_DIR, SCHEMA_DIR
from xml_utils import pretty, validate_xml

ONNX_DIR = Path(MODEL_DIR).parent / "onnx"

def _load_backend():
    """
    Returns:
        (model, tokenizer, backend_tag) where backend_tag is "ONNX" or "Torch".
    """
    if ONNX_DIR.exists():
        # Both tokenizer and ONNX model files are in the same directory
        tok = T5Tokenizer.from_pretrained(str(ONNX_DIR), local_files_only=True)

        # Load ONNX Runtime LM model (auto-detects single or multi-file export)
        mdl = ORTModelForSeq2SeqLM.from_pretrained(
            str(ONNX_DIR),
            local_files_only=True
        )
        return mdl, tok, "ONNX"

    # Fallback: regular PyTorch T5 (online or local cache/folder)
    tok = T5Tokenizer.from_pretrained(str(MODEL_DIR))
    mdl = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    return mdl, tok, "Torch"



def _load_training_metrics() -> str:
    trainer_state = Path(MODEL_DIR) / "trainer_state.json"
    if not trainer_state.exists():
        return "Training metrics not available."
    with trainer_state.open() as f:
        state = json.load(f)
    train_loss = None
    eval_loss = None
    for entry in state.get("log_history", []):
        if "train_loss" in entry:
            train_loss = entry["train_loss"]
        if "eval_loss" in entry:
            eval_loss = entry["eval_loss"]
    metrics = []
    if train_loss is not None:
        metrics.append(f"train_loss: {train_loss:.4f}")
    if eval_loss is not None:
        metrics.append(f"eval_loss: {eval_loss:.4f}")
    return "Training metrics - " + ", ".join(metrics)


MODEL, TOKENIZER, BACKEND = _load_backend()
TRAINING_INFO = _load_training_metrics()


def generate_and_validate(prompt: str, schema: str, history: List[List[str]]):
    if not prompt.strip():
        return "", "Please enter a prompt.", f"(Backend: {BACKEND})", "", history
    start = time.perf_counter()
    inputs = TOKENIZER(f"to-xml: {prompt}", return_tensors="pt")
    out_ids = MODEL.generate(**inputs, max_length=160, num_beams=4)
    xml = TOKENIZER.decode(out_ids[0], skip_special_tokens=True)
    xsd_path = str((SCHEMA_DIR / f"{schema}.xsd").resolve())
    ok, err = validate_xml(xml, xsd_path)
    duration = time.perf_counter() - start
    history.append([prompt, xml])
    return (
        pretty(xml) if ok else xml,
        "VALID ✅" if ok else f"INVALID ❌: {err}",
        f"(Backend: {BACKEND})",
        f"Generation time: {duration:.3f}s",
        history,
    )


with gr.Blocks(title="Offline XML Generator") as app:
    gr.Markdown("# Offline XML Generator")
    gr.Markdown(TRAINING_INFO)
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        schema = gr.Dropdown(choices=["user", "product", "order"], value="user", label="Schema")
        submit = gr.Button("Generate")
    xml_out = gr.Code(label="Generated XML", language="html")
    status = gr.Markdown()
    backend = gr.Markdown()
    perf = gr.Markdown()
    history_state = gr.State([])
    history_view = gr.Dataframe(headers=["Prompt", "XML"], label="History", interactive=False)
    submit.click(
        fn=generate_and_validate,
        inputs=[prompt, schema, history_state],
        outputs=[xml_out, status, backend, perf, history_state, history_view],
    )


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inline=False)
