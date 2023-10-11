# Define functions to annotate texts to identify a range of features.

import openai
from keys import OPENAI_API_KEY

# gpt-3 prompt generation to generate targets
def valence_annotation(text):
    prompt = f'''Respond with the valence (Positive/Negative) of the following text:

    {text}

    Valence (Positive/Negative):'''

    res = openai.Completion.create(model='text-davinci-003', prompt=prompt, max_tokens=20, temperature=0.5,
                                api_key=OPENAI_API_KEY)
    
    valence = res['choices'][0]['text']
    
    if valence == ' Positive':
        return 1
    return 0