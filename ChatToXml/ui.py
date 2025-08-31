import json
import time
from typing import List
import re

import gradio as gr
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
import plotly.graph_objects as go

from config import MODEL_DIR, SCHEMA_DIR
from xml_utils import pretty, validate_xml
from synth_data import generate_dataset
from train import main as train_main
from file_store import (
    create_zip_file,
    read_zip_file,
    update_zip_file,
    delete_zip_file,
    extract_zip_file,
)

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



def _load_training_metrics() -> tuple[str, go.Figure]:
    trainer_state = Path(MODEL_DIR) / "trainer_state.json"
    if not trainer_state.exists():
        return "Training metrics not available.", go.Figure()
    with trainer_state.open() as f:
        state = json.load(f)

    train_epochs: list[float] = []
    train_losses: list[float] = []
    eval_epochs: list[float] = []
    eval_losses: list[float] = []
    eval_accs: list[float] = []

    for entry in state.get("log_history", []):
        if "loss" in entry and "epoch" in entry:
            train_epochs.append(entry["epoch"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "epoch" in entry:
            eval_epochs.append(entry["epoch"])
            eval_losses.append(entry["eval_loss"])
            if "eval_accuracy" in entry:
                eval_accs.append(entry["eval_accuracy"])

    metrics = []
    if train_losses:
        metrics.append(f"train_loss: {train_losses[-1]:.4f}")
    if eval_losses:
        metrics.append(f"eval_loss: {eval_losses[-1]:.4f}")
    if eval_accs:
        metrics.append(f"eval_accuracy: {eval_accs[-1]:.4f}")

    fig = go.Figure()
    if train_epochs:
        fig.add_trace(
            go.Scatter(x=train_epochs, y=train_losses, mode="lines+markers", name="Train Loss")
        )
    if eval_epochs:
        fig.add_trace(
            go.Scatter(x=eval_epochs, y=eval_losses, mode="lines+markers", name="Validation Loss")
        )
    if eval_epochs and eval_accs:
        fig.add_trace(
            go.Scatter(
                x=eval_epochs,
                y=eval_accs,
                mode="lines+markers",
                name="Validation Accuracy",
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(title="Accuracy", overlaying="y", side="right")
        )
    fig.update_layout(title="Training Metrics", xaxis_title="Epoch", yaxis_title="Loss")

    return "Training metrics - " + ", ".join(metrics), fig


MODEL, TOKENIZER, BACKEND = _load_backend()
TRAINING_INFO, TRAINING_FIG = _load_training_metrics()


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


def run_training() -> tuple[str, str, go.Figure]:
    train_main()
    global MODEL, TOKENIZER, BACKEND
    MODEL, TOKENIZER, BACKEND = _load_backend()
    metrics, fig = _load_training_metrics()
    return "Training complete", metrics, fig


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
        with gr.Row():
            xml_out = gr.Code(label="Generated XML", language="html")
            with gr.Column():
                zip_file = gr.File(label="ZIP File", file_types=[".zip"], type="filepath")
                unzip_btn = gr.Button("Unzip")
                unzip_status = gr.Markdown()
                file_name = gr.Textbox(label="Zip Name", value="archive.zip")
                with gr.Row():
                    create_btn = gr.Button("Create")
                    read_btn = gr.Button("Read")
                with gr.Row():
                    update_btn = gr.Button("Update")
                    delete_btn = gr.Button("Delete")
                file_status = gr.Markdown()
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
        create_btn.click(
            fn=create_zip_file, inputs=[file_name, zip_file], outputs=file_status
        )
        read_btn.click(
            fn=read_zip_file, inputs=[file_name], outputs=[xml_out, file_status]
        )
        update_btn.click(
            fn=update_zip_file, inputs=[file_name, zip_file], outputs=file_status
        )
        delete_btn.click(
            fn=delete_zip_file, inputs=[file_name], outputs=file_status
        )
        unzip_btn.click(
            fn=extract_zip_file, inputs=[zip_file], outputs=unzip_status
        )

    with gr.Column(visible=False) as training_panel:
        metrics_md = gr.Markdown(TRAINING_INFO)
        metrics_plot = gr.Plot(value=TRAINING_FIG)
        gen_btn = gr.Button("Generate Synthetic Data")
        train_btn = gr.Button("Train Model")
        train_status = gr.Markdown()
        gen_btn.click(fn=run_data_generation, outputs=train_status)
        train_btn.click(fn=run_training, outputs=[train_status, metrics_md, metrics_plot])

    def _switch_mode(m: str):
        return (gr.update(visible=m == "Validation"), gr.update(visible=m == "Training"))

    mode.change(_switch_mode, inputs=mode, outputs=[validation_panel, training_panel])


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inline=False)
