import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
import torch

#for not seing a warning message
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from keras.models import model_from_json

import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')



def pipeline(query:str, ) -> str:
    """
    This function is the pipeline for the entire project. It takes in a query and finds the most relevant document.
    and gives it to the OpenAI API to generate a answer
    :param semantic_search_model:
    :param query:
    :return:
    """
    # 1. Preprocess the query
    embedding = get_text_embedding(query)
    # 2. Semantic Search
    best_ctx = semantic_search_model(embedding, method='ann')
    # 3. Answer Generation
    answer = answer_generation(query, best_ctx)
    # 4. Return the answer
    return answer

def get_text_embedding(text, model_name='bert-base-uncased'):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input text and convert to PyTorch tensors
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Get output from pre-trained model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract last layer of output (CLS token) as the text embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding

def semantic_search_model(embedding:np.ndarray, method:str = 'ann') -> str:
    """
    This function takes in a query embedding and finds the most relevant document by using the ANN
    :param embedding:
    :return:
    """
    # load the dataset
    df = pd.read_pickle('./Data_Generation/df_pickle/question-context-20-emb.pkl')
    # load the embeddings
    # make the df only contain the unique contexts
    df = df.drop_duplicates(subset=['context'])
    context_embeddings = np.stack(df['context_embedding'].to_numpy())
    # Concatonate the query embedding ontop of each element in the context_embeddings, so each row is the query embedding and the context embedding
    model_input = np.concatenate((np.tile(embedding, (len(context_embeddings), 1)), context_embeddings),
                                 axis=1)

    if method == 'ann':
        with open('ANN_resamp.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("ANN/ANN_resamp.h5")
        print("Loaded model from disk")

        # Predict the most relevant context
        prediction = loaded_model.predict(model_input)
        # Get the 2 most relevant contexts
        index = np.argsort(prediction, axis=0)[-2:]
        # Get the context
        context1 = df.iloc[index[0][0]]['context']
        context2 = df.iloc[index[1][0]]['context']
        context = context1+ context2
    elif method == 'cs':
        # Should simply be the cosine similarity
        x = np.array(embedding)
        y = context_embeddings
        cos_sim = np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y, axis=1))
        # Get the 2 most relevant contexts
        index = np.argsort(cos_sim, axis=0)[-2:]
        # Get the context
        context1 = df.iloc[index[0]]['context']
        context2 = df.iloc[index[1]]['context']
        context = context1 + context2

    return context

def answer_generation(query:str, context:str) -> str:
    """
    This function takes in a query and a context and uses the OpenAI API to generate an answer
    :param query:
    :param context:
    :return:
    """
    print(f'CONTEXT: ```{context}``` QUESTION: ```{query}``` ANSWER:')
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system",
                 "content": "You are a Teachers Assistant and you should answer the QUESTION using the information given in the CONTEXT, if the CONTEXT is unrelated, you should ignore it."},
                {"role": "user", "content": f'CONTEXT: ```{context}``` QUESTION: ```{query}``` ANSWER:'},
            ]
        )
    except Exception as e:
        print("OPENAI_ERROR:", str(e))

    return completion.choices[0]


if __name__ == "__main__":
    # Read in the data
    answer = pipeline("Which linkage function uses the eucladian distance? Does both maximum and average linkage use the eucladian distance?")
    print(answer)