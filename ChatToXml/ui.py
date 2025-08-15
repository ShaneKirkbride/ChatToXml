import json
import time
from typing import List
import re

import gradio as gr
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import MODEL_DIR, SCHEMA_DIR
from xml_utils import pretty, validate_xml
from synth_data import generate_dataset
from train import main as train_main

ONNX_DIR = Path(MODEL_DIR).parent / "onnx"

# Minimal fixer for common bracket issues like "user>id>123</user>"
_MISSING_LT = re.compile(r'(?<!<)(/?)([A-Za-z_][\w\-.]*)(?=>)')
_CLOSE_SIMPLE = re.compile(r'<([A-Za-z_][\w\-.]*)>([^<]+)/\1>')

def repair_xml(xml_text: str, root_hint: str | None = None) -> str:
    """
    Light-touch repair for missing '<' before tag names and bare close tags.
    Keeps content unchanged beyond necessary fixes.
    """
    if not xml_text:
        return xml_text
    s = _MISSING_LT.sub(r'<\1\2', xml_text)         # "user>id>123</user>" -> "<user><id>123</id></user>"
    s = _CLOSE_SIMPLE.sub(r'<\1>\2</\1>', s)        # "<id>123/id>" -> "<id>123</id>"
    if root_hint:
        stripped = s.strip()
        if not stripped.startswith('<') or not stripped.endswith('>'):
            s = f'<{root_hint}>{s}</{root_hint}>'
    return s

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


def generate_and_validate(prompt: str, schema: str, do_repair: bool, history: List[List[str]]):
    if not prompt.strip():
        return "", "Please enter a prompt.", f"(Backend: {BACKEND})", "", history, history

    start = time.perf_counter()
    task = f"to-xml ({schema}): {prompt}"
    inputs = TOKENIZER(task, return_tensors="pt")

    out_ids = MODEL.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=4,
        length_penalty=0.1,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    xml = TOKENIZER.decode(out_ids[0], skip_special_tokens=True)

    xsd_path = str((SCHEMA_DIR / f"{schema}.xsd").resolve())
    ok, err = validate_xml(xml, xsd_path)

    fixed = xml
    if do_repair and not ok:
        fixed_try = repair_xml(xml, root_hint=schema)
        ok2, err2 = validate_xml(fixed_try, xsd_path)
        if ok2:            
            ok, err = True, "FIXED ✅ (auto-repaired common tag issues)"
        else:
            err = f"INVALID ❌: {err2}"
        fixed = fixed_try

    duration = time.perf_counter() - start
    history.append([prompt, fixed])

    return (
        pretty(fixed) if ok else fixed,
        "VALID ✅" if ok else err,
        f"(Backend: {BACKEND})",
        f"Generation time: {duration:.3f}s",
        history,
        history,
    )


def run_data_generation() -> str:
    path = generate_dataset()
    return f"Synthetic dataset generated at {path}"


def run_training() -> tuple[str, str]:
    train_main()
    global MODEL, TOKENIZER, BACKEND
    MODEL, TOKENIZER, BACKEND = _load_backend()
    metrics = _load_training_metrics()
    return "Training complete", metrics


with gr.Blocks(title="Offline XML Generator") as app:
    gr.Markdown("# Offline XML Generator")
    mode = gr.Radio(["Validation", "Training"], value="Validation", label="Mode")

    with gr.Column() as validation_panel:
        with gr.Row():
            prompt = gr.Textbox(label="Prompt")
            schema = gr.Dropdown(choices=["user", "product", "order"], value="user", label="Schema")
            auto_repair = gr.Checkbox(value=True, label="Auto-repair invalid XML")
            submit = gr.Button("Generate")
        gr.Examples(
            examples=[
                [
                    "Create a user named Alice with ID 123 and email alice@example.com",
                    "user",
                ],
                [
                    "Generate a product with id 42 named Widget priced at 9.99",
                    "product",
                ],
                [
                    "Create order 555 for user Alice totaling 123.45",
                    "order",
                ],
            ],
            inputs=[prompt, schema],
            label="Example Prompts",
        )
        xml_out = gr.Code(label="Generated XML", language="html")
        status = gr.Markdown()
        backend = gr.Markdown()
        perf = gr.Markdown()
        history_state = gr.State([])
        history_view = gr.Dataframe(headers=["Prompt", "XML"], label="History", interactive=False)
        submit.click(
            fn=generate_and_validate,
            inputs=[prompt, schema, auto_repair, history_state],
            outputs=[xml_out, status, backend, perf, history_state, history_view],
        )

    with gr.Column(visible=False) as training_panel:
        metrics_md = gr.Markdown(TRAINING_INFO)
        gen_btn = gr.Button("Generate Synthetic Data")
        train_btn = gr.Button("Train Model")
        train_status = gr.Markdown()
        gen_btn.click(fn=run_data_generation, outputs=train_status)
        train_btn.click(fn=run_training, outputs=[train_status, metrics_md])

    def _switch_mode(m: str):
        return (gr.update(visible=m == "Validation"), gr.update(visible=m == "Training"))

    mode.change(_switch_mode, inputs=mode, outputs=[validation_panel, training_panel])


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inline=False)
