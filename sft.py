import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset, load_metric
import numpy as np
import nltk


TEST_DATA = """\
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
"""


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def load_and_shrink_t5_model(model_name):
    shrunk_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    for i in reversed([1, 3, 5]):
        del shrunk_model.decoder.block[i]

    shrunk_model.config.num_decoder_layers = 3
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
            ["summarize: " + text for text in example["chapter"]],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        targets = tokenizer(
            example["summary_text"],
            padding=True,
            truncation=True,
            max_length=200,
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
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        learning_rate=3e-4,
        # lr_scheduler_type="constant",
        per_device_train_batch_size=8,
        # per_device_eval_batch_size=1,
        warmup_steps=100,
        weight_decay=0.0001,
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
        # do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # eval_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        # compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()


def test_model(prefix, model):
    tokenizer = T5Tokenizer.from_pretrained("cnicu/t5-small-booksum")

    model.eval()

    input_ids = tokenizer.encode(
        "summarize: " + TEST_DATA,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    ).to(device)

    generated_ids = model.generate(input_ids, max_length=120)[0]
    print(
        prefix
        + "\n"
        + tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            remove_invalid_values=True,
        )
    )


if __name__ == "__main__":
    shrunk_model = load_and_shrink_t5_model("cnicu/t5-small-booksum")
    test_model("After Shrink, before fine-tuning: ", shrunk_model)
    try:
        fine_tune_model(shrunk_model)
    finally:
        test_model("After Shrink and fine-tuning ", shrunk_model)
        shrunk_model.save_pretrained("sft-t5-small")
