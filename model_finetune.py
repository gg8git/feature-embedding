from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from compile_dataset import get_dataset
from format_dataset import tokenize_data

model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# [IMPORTANT] fix data collator for text regression specific scenario
def t5_data_collator(examples):
    inputs = [example["input_ids"] for example in examples]
    attention_mask = [example["attention_mask"] for example in examples]
    labels = [example["labels"] for example in examples]

    return {
        "input_ids": inputs,
        "attention_mask": attention_mask,
        "labels": labels
    }

# training presets
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=3
)

# training setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenize_data(get_dataset()),
    data_collator=t5_data_collator
)

# training model
trainer.train()
trainer.save_model()

