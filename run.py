from compile_dataset import compile_dataset
from test import test

compile_dataset()

test_prompt = ""
res = test(test_prompt)

print(res)