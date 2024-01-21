import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, load_metric
import numpy as np
import nltk


def load_and_shrink_t5_model(model_name):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")
    # Load the original T5 model (cnicu/t5-small-booksum)
    original_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Create a new T5 configuration for the shrunk model
    new_config = T5Config.from_pretrained(
        model_name, num_decoder_layers=3, num_encoder_layers=3
    )

    shrunk_model = T5ForConditionalGeneration(new_config).to(device)

    # Copy selected layers from the original model
    for i, layer_idx in enumerate([0, 2, 4]):
        shrunk_model.encoder.block[i] = original_model.encoder.block[layer_idx]

    for i, layer_idx in enumerate([1, 3, 5]):
        shrunk_model.decoder.block[i] = original_model.decoder.block[layer_idx]

    return shrunk_model


def compute_metrics(eval_pred, tokenizer):
    rouge = load_metric("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    return rouge.compute(predictions=decoded_preds, references=decoded_labels)


def load_and_tokenize_dataset(tokenizer):
    train_dataset = load_dataset("kmfoda/booksum")
    train_dataset = train_dataset.select_columns(
        [
            "summary_length",
            "summary_text",
            "chapter",
        ]
    )
    train_dataset = train_dataset.filter(
        lambda x: x["chapter"] is not None and x["summary_text"] is not None
    )

    def tokenize_function(example):
        inputs = tokenizer(
            example["chapter"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        targets = tokenizer(
            example["summary_text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": targets.input_ids,
        }

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.remove_columns(
        ["chapter", "summary_text", "summary_length"]
    )

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return train_dataset


def fine_tune_model(model):
    tokenizer = T5Tokenizer.from_pretrained("cnicu/t5-small-booksum")

    tokenized_datasets = load_and_tokenize_dataset(tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=8,
        learning_rate=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        # evaluation_strategy="epoch",
        # eval_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        # compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()


if __name__ == "__main__":
    shrunk_model = load_and_shrink_t5_model("cnicu/t5-small-booksum")

    fine_tune_model(shrunk_model)
    shrunk_model.save_pretrained("sft-t5-small")
