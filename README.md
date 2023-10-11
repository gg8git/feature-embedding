# Feature Embedding Project

**Goal:**
The goal of this project is to create a text-to-number regression model that evaluates each prompt text on a variety of features, creating a mapping from each text into feature embeddings.

**Features:**
The features we will aim to analyze in this project include:
1. Valence
2. Complexity
3. Format
4. Bias
We will initially assess texts for these features based on the Open-AI GPT-3.5 model.

**Steps:**
In order to accomplish our goal, we will take the following steps:
1. Annotate a variety of texts for the desired features using Open-AI's GPT-3.5 model.
2. For each feature, finetune a T5 transformer model to map prompt text to a number between 0 and 1 (where 0 represents not having the feature and 1 represents having the feature).
3. Collect the models to create an aggregate mapping from the prompt texts to the feature embeddings.