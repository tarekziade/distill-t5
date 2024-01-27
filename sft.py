"""
Shrink and Fine-tune a T5 model for summarization.

1. remove 50% of the decoder layers
2. fine-tune the shrinked model with various datasets
"""
import os
import functools
import re

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset
import nltk

nltk.download("punkt")


# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

print(f"Using device: {device}")


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def print_model(model):
    size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    size = "%.2fM" % size
    encoder = model.config.num_layers
    decoder = model.config.num_decoder_layers

    print(
        f"This model has {encoder} encoder layers and {decoder} decoders layers.\n"
        f"Its size is {size} parameters shrinking in half"
    )


def shrink_layer(model, layer_name, new_size=None):
    if layer_name == "encoder":
        config_name = "num_layers"
    else:
        config_name = f"num_{layer_name}_layers"

    current_size = getattr(model.config, config_name)

    if new_size is None:
        new_size = int(current_size / 2)

    if current_size != new_size:
        layers_to_remove = [i for i in range(1, new_size * 2, 2)]

        for i in reversed(layers_to_remove):
            del getattr(model, layer_name).block[i]

        setattr(model.config, config_name, new_size)


def load_and_shrink_t5_model(
    model_name, num_decoder_layers=None, num_encoder_layers=None
):
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    print_model(model)
    print("Shrinking model...")
    shrink_layer(model, "decoder", num_decoder_layers)
    shrink_layer(model, "encoder", num_encoder_layers)
    print_model(model)
    return model


def load_and_tokenize_dataset(name, tokenizer):
    def sentence_aware_truncation(text, max_length):
        """Truncate at the sentence level so it fits in the max_length tokens"""
        sentences = nltk.tokenize.sent_tokenize(text)
        truncated_text = ""
        for sentence in sentences:
            tokens = tokenizer.encode(
                truncated_text + sentence, add_special_tokens=False
            )
            if len(tokens) > max_length:
                break
            truncated_text += sentence + " "

        return truncated_text.strip()

    def tokenize_function(example, source_field, target_field):
        inputs = tokenizer(
            [
                sentence_aware_truncation("summarize: " + clean_text(text), 512)
                for text in example[source_field]
            ],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

        targets = tokenizer(
            [
                sentence_aware_truncation(clean_text(summary), 200)
                for summary in example[target_field]
            ],
            padding="max_length",
            truncation=True,
            max_length=200,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": targets.input_ids,
        }

    def prepare_dataset(dataset, source_field, summary_field):
        dataset = dataset.filter(
            lambda x: x[source_field] is not None and x[summary_field] is not None
        )

        dataset = dataset.map(
            functools.partial(
                tokenize_function, source_field=source_field, target_field=summary_field
            ),
            batched=True,
        )

        dataset = dataset.select_columns(
            [
                "input_ids",
                "attention_mask",
                "labels",
            ]
        )
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    train_dataset = load_dataset(name)
    train_dataset = prepare_dataset(train_dataset, "content", "headline")
    return train_dataset.shuffle()


def fine_tune_model(dataset_name, model, tokenizer):
    tokenized_datasets = load_and_tokenize_dataset(dataset_name, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        run_name="shrink-and-finetune",
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # eval_steps=10,
        # save_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()
    trainer.eval_dataset = tokenized_datasets["test"]
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "shrink-and-fine-tune"  # name your W&B project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
    model_name = tokenizer_name = "JulesBelveze/t5-small-headline-generator"
    dataset_name = "JulesBelveze/tldr_news"

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    shrunk_model = load_and_shrink_t5_model(model_name)

    try:
        fine_tune_model(dataset_name, shrunk_model, tokenizer)
    finally:
        target_dir = f"{model_name.split('/')[-1]}-sft-{shrunk_model.config.num_layers}-{shrunk_model.config.num_decoder_layers}"

        shrunk_model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)
