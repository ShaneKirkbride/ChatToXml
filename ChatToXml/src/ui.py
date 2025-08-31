import json
import time
from typing import List
from pathlib import Path

import gradio as gr
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
import plotly.graph_objects as go

from config import MODEL_DIR, SCHEMA_DIR
from xml_utils import pretty, validate_xml
from synth_data import generate_dataset
from train import main as train_main

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
            go.Scatter(x=eval_epochs, y=eval_accs, mode="lines+markers", name="Validation Accuracy", yaxis="y2")
        )
        fig.update_layout(yaxis2=dict(title="Accuracy", overlaying="y", side="right"))
    fig.update_layout(title="Training Metrics", xaxis_title="Epoch", yaxis_title="Loss")

    return "Training metrics - " + ", ".join(metrics), fig


MODEL, TOKENIZER, BACKEND = _load_backend()
TRAINING_INFO, TRAINING_FIG = _load_training_metrics()


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
        xml_out = gr.Code(label="Generated XML", language="html", interactive=True)
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
