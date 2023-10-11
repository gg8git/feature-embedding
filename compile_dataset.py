# Compile overall dataset through importing texts and creating the annotated collection of texts.

from annotation import valence_annotation

# [IMPORTANT] import texts
texts = []
data = []

# compile dataset with annotations
def compile_dataset():
    for text in texts:
        data.append({"input": text, "target": valence_annotation(text)})

# return dataset to model finetuning program
def get_valence_dataset():
    return data