# Finetune the T5 transformer model based on the compiled dataset.

from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer
from compile_dataset import get_valence_dataset
from tokenization import tokenize_data

def model_finetuning():
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # data collator to organize data for trainer
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
        train_dataset=tokenize_data(get_valence_dataset()),
        data_collator=t5_data_collator
    )

    # training model
    trainer.train()
    trainer.save_model()

    return model

