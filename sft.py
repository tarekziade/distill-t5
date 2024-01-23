import functools
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from datasets import (
    load_dataset,
    load_metric,
    concatenate_datasets,
    Dataset,
    DatasetDict,
)
import numpy as np
import nltk
import re

nltk.download("punkt")


TEST_DATA = """\
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
"""


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_shrink_t5_model(model_name, tokenizer):
    shrunk_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    test_model("Before Shrink, before fine-tuning: ", shrunk_model, tokenizer)

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
    def sentence_aware_truncation(text, max_length):
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

    def tokenize_function(example, source_field="chapter", target_field="summary_text"):
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

        dataset = train_dataset.map(tokenize_function, batched=True)
        dataset = dataset.map(
            functools.partial(
                tokenize_function, source_field=source_field, target_field=summary_field
            ),
            batched=True,
        )

        dataset.select_columns(
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

    print("Loading Booksum")
    train_dataset = load_dataset("kmfoda/booksum")
    train_dataset = prepare_dataset(train_dataset, "chapter", "summary_text")

    # wikipedia
    print("Loading Wikipedia")
    train_dataset2 = load_dataset("tarekziade/wikipedia-topics", split="train[:10%]")
    train_dataset2 = prepare_dataset(train_dataset2, "text", "summary")
    train_dataset["train"] = concatenate_datasets(
        [train_dataset["train"], train_dataset2]
    )

    # xsum
    print("Loading xsum")
    train_dataset3 = load_dataset("EdinburghNLP/xsum", split="train[:5%]")
    train_dataset3 = prepare_dataset(train_dataset3, "document", "summary")
    train_dataset["train"] = concatenate_datasets(
        [train_dataset["train"], train_dataset3]
    )

    return train_dataset.shuffle()


def fine_tune_model(model, tokenizer):
    tokenized_datasets = load_and_tokenize_dataset(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        learning_rate=3e-4,
        # lr_scheduler_type="constant",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        # compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    trainer.train()

    trainer.eval_dataset = tokenized_datasets["test"]

    metrics = trainer.evaluate()
    print(metrics)


def test_model(prefix, model, tokenizer):
    model.eval()

    input_ids = tokenizer.encode(
        "summarize: " + clean_text(TEST_DATA),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        add_special_tokens=False,
    ).to(device)

    generated_ids = model.generate(
        input_ids, max_length=120, no_repeat_ngram_size=0, num_beams=4, top_k=50
    )[0]
    print(
        "*********************** "
        + prefix
        + ": \n"
        + tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            remove_invalid_values=True,
        )
        + "***********************"
        + "\n\n"
    )


if __name__ == "__main__":
    model_name = "cnicu/t5-small-booksum"
    # model_name = "Alred/t5-small-finetuned-summarization-cnn"
    # model_name = "Falconsai/text_summarization"

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    shrunk_model = load_and_shrink_t5_model(model_name, tokenizer)

    test_model("After Shrink, before fine-tuning: ", shrunk_model, tokenizer)
    try:
        fine_tune_model(shrunk_model, tokenizer)
    finally:
        test_model("After Shrink and fine-tuning ", shrunk_model, tokenizer)
        shrunk_model.save_pretrained(model_name.split("/")[-1] + "-sft")
