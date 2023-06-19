# Learning with ChatGPT project

## Aim of the project

This repository is made for a project, exploring using LLM's such as ChatGPT for creating an AI Teaching Assistant.\
This includes all elements needed for a pipeline for that task:

- Data generation
- Semantic search
- Full pipeline

## Requirements

Requirements for running this project can be seen [here](requirements.txt).
For running most files an OpenAI API key is also needed. A .env file can be created with an API-key defined for running the scripts.

## Structure of repository

This repository contains multiple scripts, Jupyter Notebooks and other files.\
\
We encourage to view the script [final pipeline](pipeline.ipynb) for a script of the entire final pipeline from input query to output answer\
\
The [Data_generation](Data_Generation) folder contains scripts for splitting PDF documents into paragraphs. A
folder [df_pickle](Data_Generation/df_pickle) is also found here which contains raw datasets that can be loaded into
other scripts for further processing. The final raw dataset
is [final_02450_emb.pkl](Data_Generation/df_pickle/final_02450_emb.pkl) both containing questions paragraphs and
associated embeddings.\
\
The Notebook [Similarity network](Similarity%20network.ipynb) contains code for training Semantic search models. Weights
and structure of trained models are saved in the folder [ANN](ANN).\
\
By running the script [Interface](Interface.py) an interactive interface of using the pipeline can be viewed and used.
