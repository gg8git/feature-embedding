from annotation import valence_annotation

# [IMPORTANT] import texts
texts = []

def get_dataset(texts):
    data = []

    for text in texts:
        data.append({"input": text, "target": valence_annotation(text)})
    
    return data