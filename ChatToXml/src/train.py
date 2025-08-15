import os
import shutil
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
from config import DATA_CSV, MODEL_DIR

MODEL_NAME = "t5-small"

def preprocess_fn(tokenizer: T5Tokenizer):
    def _fn(examples):
        inputs = [f"to-xml: {x}" for x in examples["input"]]
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
        # as_target_tokenizer is deprecated; use target tokenization directly if you like
        labels = tokenizer(examples["text_output"], padding="max_length", truncation=True, max_length=160)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _fn

def main() -> None:
    # Use a NEW output dir per run to avoid file-lock conflicts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_out = Path(f"{MODEL_DIR}_run_{ts}")

    dataset = load_dataset("csv", data_files=str(DATA_CSV))["train"].train_test_split(test_size=0.1, seed=42)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenized = dataset.map(preprocess_fn(tokenizer), batched=True, remove_columns=["input", "text_output"])
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)

    args = TrainingArguments(
        output_dir=str(tmp_out),              # <<< write to temp dir
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-4,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        fp16=False,
        report_to=[],
        save_safetensors=True,               # set False if .safetensors still gets locked
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    tmp_out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(tmp_out)
    tokenizer.save_pretrained(tmp_out)

    # At this point, make sure your inference app is NOT holding files in MODEL_DIR.
    # Then atomically replace MODEL_DIR with tmp_out.
    model_dir = Path(MODEL_DIR)
    if model_dir.exists():
        # best-effort remove; on Windows you might need to stop the app first
        shutil.rmtree(model_dir)
    shutil.move(str(tmp_out), str(model_dir))

if __name__ == "__main__":
    main()
