from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Single folder containing both ONNX model and tokenizer
model_dir = "./t5-small"

# Load tokenizer from the same folder
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

# Load ONNX Runtime model from the same folder
model = ORTModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)

# Prepare input
input_text = "translate English to German: Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
outputs = model.generate(**inputs)

# Decode output
print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
