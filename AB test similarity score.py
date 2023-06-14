import random
from entire_pipeline import*
import gradio as gr

# a/b test for similarity score

def get_context(query: str, method: str = 'cs', n_contexts: int = 5):
    """
    This function is the pipeline for the entire project. It takes in a query and finds the most relevant document.
    and gives it to the OpenAI API to generate a answer
    :param n_contexts: The number of contexts to return
    :param semantic_search_model: The semantic search model to use
    :param query: The query to search for
    :return:
    """
    # 1. Preprocess the query
    embedding = get_text_embedding(query)
    # 2. Semantic Search
    best_ctx = semantic_search_model(embedding, method, n_contexts)
    # 4. Return best context
    return best_ctx

def chatbot(input):
    if input:
        context_cs = get_context(input, method='cs')
        #context_wcs = get_context(input, method='weighted_cs')
        context_ann = get_context(input, method='ann')
        return context_cs, context_ann


questions = [
    "What is the purpose of regularization in machine learning?",
    "What is the difference between supervised and unsupervised learning?",
    "What are the key steps in building a machine learning model?"
   # "What is the role of feature engineering in machine learning?",
  #  "What is the concept of overfitting in machine learning?",
 #   "How does gradient descent work in training a neural network?",
    #"What is the trade-off between bias and variance in machine learning?",
   # "What is the purpose of cross-validation in machine learning?",
  #  "What are some common evaluation metrics used in machine learning?",
 #   "How can you handle missing data in a machine learning dataset?"
]


outputs = gr.outputs.Textbox()
inputs = []
responses = []  # Track selected responses
for question in questions:

    #uses the pipeline to genrerate context paragraphs
    context_cs, context_ann = chatbot(question)

    #Just for testing without using the api
    #answer_pipeline_cs, answer_pipeline_wcs, answer_pipeline_ann, answer_chatgpt = "1", "2", "3", "4"
    input_choices = [context_cs, context_ann]
    #shuffles the answers
    shuffled_choices = sorted(input_choices, key=lambda k: random.random())
    inputs.append(gr.inputs.Radio(shuffled_choices, label=question, type="value"))
    responses.append(input_choices)  # Add choices to selected_responses

def evaluate_responses(*preferred_responses):
    # counnt how many times each answer was chosen
    count_cs = 0
    #count_wcs = 0
    count_ann = 0
    for i, respons in enumerate(preferred_responses):
        if respons == responses[i][0]:
            count_cs += 1
        elif respons == responses[i][1]:
            count_ann += 1

    summary = f"You preferred {count_cs} context paragraphs from cosine similarity and {count_ann} context paragraphs from the ANN."
    return summary


interface = gr.Interface(fn=evaluate_responses, inputs=inputs, outputs=outputs, title="AI Chatbot Evaluation",
                         description="Evaluate responses from pipeline and ChatGPT",
                         theme="compact", allow_flagging="never")

interface.launch()