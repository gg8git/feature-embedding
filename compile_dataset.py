from annotation import valence_annotation

# [IMPORTANT] import texts
texts = []
data = []

def compile_dataset():
    for text in texts:
        data.append({"input": text, "target": valence_annotation(text)})

def get_valence_dataset():
    return data