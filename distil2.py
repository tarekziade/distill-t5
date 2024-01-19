import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import log_softmax, softmax

from torch import tensor

mps_device = torch.device("mps")


TEST_DATA = """\
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
"""
teacher_model = T5ForConditionalGeneration.from_pretrained("cnicu/t5-small-booksum")
teacher_model.to(mps_device)

teacher_model.eval()

tokenizer = T5Tokenizer.from_pretrained("cnicu/t5-small-booksum")

train_dataset = load_dataset("kmfoda/booksum", split="train[:1%]")
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


def test_model(prefix, model):
    model.eval()

    input_ids = tokenizer.encode(
        "summarize: " + TEST_DATA,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    ).to(mps_device)

    generated_ids = model.generate(input_ids, max_length=120)[0]
    print(
        prefix
        + " "
        + tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            remove_invalid_values=True,
        )
    )


def tokenize_function(example):
    inputs = tokenizer(
        ["summarize: " + item for item in example["chapter"]],
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
        "input_ids": inputs.input_ids.squeeze(),
        "attention_mask": inputs.attention_mask.squeeze(),
        "labels": targets.input_ids.squeeze(),
    }


train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(
    ["chapter", "summary_text", "summary_length"]
)


print(train_dataset)

config = teacher_model.config
config.num_layers = 2
# config.d_model = 128
# config.d_ff = 512
# config.d_kv = 64

student_model = T5ForConditionalGeneration(config)
student_model.to(mps_device)

test_model("Before distillation: ", student_model)

student_model.train()


# Hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 1
temperature = 2


def collate_fn(batch):
    input_ids = tensor([item["input_ids"] for item in batch]).to(mps_device)
    attention_mask = tensor([item["attention_mask"] for item in batch]).to(mps_device)
    labels = tensor([item["labels"] for item in batch]).to(mps_device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)


# Training Loop
for epoch in range(num_epochs):
    loss_value = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Loss {loss_value}")

    for batch in progress_bar:
        optimizer.zero_grad()

        # Forward pass through the teacher model
        with torch.no_grad():
            teacher_outputs = teacher_model(**batch)

        # Forward pass through the student model
        student_outputs = student_model(**batch)
        assert student_outputs.logits.size() == teacher_outputs.logits.size()

        student_log_probs = log_softmax(student_outputs.logits / temperature, dim=-1)
        teacher_probs = softmax(teacher_outputs.logits / temperature, dim=-1)

        # Calculate loss
        loss = torch.nn.functional.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean"
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss_value=loss.item())


student_model.save_pretrained("distilled-t5-small")

test_model("After distillation ", student_model)
