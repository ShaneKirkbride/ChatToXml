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
        model_inputs = tokenizer(
            inputs, padding="max_length", truncation=True, max_length=128
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["text_output"],
                padding="max_length",
                truncation=True,
                max_length=160,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _fn


def main() -> None:
    dataset = load_dataset("csv", data_files=str(DATA_CSV))["train"].train_test_split(
        test_size=0.1, seed=42
    )
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    tokenized = dataset.map(
        preprocess_fn(tokenizer), batched=True, remove_columns=["input", "text_output"]
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
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
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()
