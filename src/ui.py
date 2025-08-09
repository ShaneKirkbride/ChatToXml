import gradio as gr
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import MODEL_DIR, SCHEMA_DIR
from xml_utils import pretty, validate_xml

ONNX_DIR = Path(MODEL_DIR).parent / "onnx"


def _load_backend():
    if ONNX_DIR.exists():
        tok = T5Tokenizer.from_pretrained(str(ONNX_DIR))
        model_path = ONNX_DIR / "model-quant.onnx"
        if not model_path.exists():
            model_path = ONNX_DIR / "model.onnx"
        mdl = ORTModelForSeq2Seq.from_pretrained(str(ONNX_DIR), file_name=model_path.name)
        return mdl, tok, "ONNX"
    tok = T5Tokenizer.from_pretrained(str(MODEL_DIR))
    mdl = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR))
    return mdl, tok, "Torch"


MODEL, TOKENIZER, BACKEND = _load_backend()


def generate_and_validate(prompt: str, schema: str):
    if not prompt.strip():
        return "", "Please enter a prompt.", f"(Backend: {BACKEND})"
    inputs = TOKENIZER(f"to-xml: {prompt}", return_tensors="pt")
    out_ids = MODEL.generate(**inputs, max_length=160, num_beams=4)
    xml = TOKENIZER.decode(out_ids[0], skip_special_tokens=True)
    xsd_path = str((SCHEMA_DIR / f"{schema}.xsd").resolve())
    ok, err = validate_xml(xml, xsd_path)
    return (
        pretty(xml) if ok else xml,
        "VALID ✅" if ok else f"INVALID ❌: {err}",
        f"(Backend: {BACKEND})",
    )


with gr.Blocks(title="Offline XML Generator") as app:
    gr.Markdown("# Offline XML Generator")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        schema = gr.Dropdown(choices=["user", "product", "order"], value="user", label="Schema")
        submit = gr.Button("Generate")
    xml_out = gr.Code(label="Generated XML", language="xml")
    status = gr.Markdown()
    backend = gr.Markdown()
    submit.click(fn=generate_and_validate, inputs=[prompt, schema], outputs=[xml_out, status, backend])


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inline=False)
