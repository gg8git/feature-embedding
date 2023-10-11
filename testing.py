from transformers import T5Tokenizer
from model_finetune import model_finetuning

def test(prompt):
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    model = model_finetuning()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(input_ids)
    res = float(tokenizer.decode(output[0], skip_special_tokens=True))

    return res