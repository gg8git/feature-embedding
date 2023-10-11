from compile_dataset import compile_dataset
from testing import test

compile_dataset()

test_prompt = ""
res = test(test_prompt)

print(res)