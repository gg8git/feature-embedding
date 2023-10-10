from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from format_dataset import get_dataset

model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# [IMPORTANT] fix data collator for text regression specific scenario
def t5_data_collator(examples):
    inputs = tokenizer(
        [f"translate English to French: {example['text']}" for example in examples],  # Add task-specific prefix
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,  # Set max length as needed
    )

    labels = tokenizer(examples, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": labels.input_ids,
    }

# training presets
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    overwrite_output_dir=True
)

# training setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=get_dataset(),
    data_collator=t5_data_collator
)

# training model
trainer.train()
trainer.save_model()

