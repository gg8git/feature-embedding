from transformers import T5Tokenizer
import tensorflow as tf

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize_data(data):
    tokenized_data = []

    # tokenize dataset
    for example in data:
        input_text = example["input"]
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128) # find out what max_length does
        tokenized_data.append({"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": example["target"]}) # find out what first two params are

    # convert to tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant([item["input_ids"] for item in tokenized_data]),
        tf.constant([item["attention_mask"] for item in tokenized_data]),
        tf.constant([item["labels"] for item in tokenized_data]))
    )

    return dataset