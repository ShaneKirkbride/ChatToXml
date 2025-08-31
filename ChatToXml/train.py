import os
import shutil
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import numpy as np
import evaluate

from config import DATA_CSV, MODEL_DIR

MODEL_NAME = "t5-small"


def preprocess_fn(tokenizer: T5Tokenizer):
    def _fn(examples):
        # Input prompt
        inputs = [f"to-xml: {x}" for x in examples["input"]]
        model_inputs = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        # Targets (use text_target to avoid deprecated as_target_tokenizer)
        labels = tokenizer(
            text_target=examples["text_output"],
            padding="max_length",
            truncation=True,
            max_length=160,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _fn


def main() -> None:
    # New output dir per run to avoid file-lock conflicts on Windows
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_out = Path(f"{MODEL_DIR}_run_{ts}")

    dataset = load_dataset("csv", data_files=str(DATA_CSV))["train"].train_test_split(
        test_size=0.2, seed=42
    )

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # --- Add characters that are crucial for XML emission ---
    # (kept as normal tokens so the model can freely generate them)
    added = tokenizer.add_tokens(["<", ">", "/"], special_tokens=False)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    tokenized = dataset.map(
        preprocess_fn(tokenizer),
        batched=True,
        remove_columns=["input", "text_output"],
    )

    # Collator handles label padding to -100 automatically for seq2seq
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"accuracy": accuracy.compute(predictions=decoded_preds, references=decoded_labels)["accuracy"]}

    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_out),           # write to temp dir
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
        save_safetensors=True,             # set False if you still hit file locks
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    tmp_out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(tmp_out)
    tokenizer.save_pretrained(tmp_out)

    # Atomically replace MODEL_DIR with the new run
    model_dir = Path(MODEL_DIR)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.move(str(tmp_out), str(model_dir))


if __name__ == "__main__":
    main()
