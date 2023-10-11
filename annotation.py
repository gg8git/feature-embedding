import openai
from keys import OPENAI_API_KEY

# [IMPORTANT] prompt generation to generate targets
def valence_annotation(text):
    prompt = f'''

    '''
    res = openai.Completion.create(model='text-davinci-003', prompt=prompt, max_tokens=10, temperature=0,
                                api_key=OPENAI_API_KEY)
    synbio_classification = res['choices'][0]['text']
    if synbio_classification == ' Yes':
        return True
    return False

def is_synbio(desc):
    prompt = f'''
    Based on the following description of a company, classify whether this startup could potentially be considered a synthetic biology startup.

    {desc}

    Yes/No:
    '''
    res = openai.Completion.create(model='text-davinci-003', prompt=prompt, max_tokens=10, temperature=0,
                                api_key=OPENAI_API_KEY)
    synbio_classification = res['choices'][0]['text']
    if synbio_classification == ' Yes':
        return True
    return False