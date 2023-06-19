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
        #context_ann = get_context(input, method='ann')
        return context_cs


path = "C:/Users/farim/PycharmProjects/ChatGPT_Tutor_Project/Documents/"
with open(path+'questions.txt') as f:
    lines = f.readlines()

# select questions to use. Use index in lines to choose.
questions = [q[:-1] for q in lines[31:61]]

outputs = gr.outputs.Textbox()
inputs = []
responses = []  # Track selected responses
for question in questions:

    #uses the pipeline to genrerate context paragraphs
    context_cs = chatbot(question)

    #Just for testing without using the api
    #answer_pipeline_cs, answer_pipeline_wcs, answer_pipeline_ann, answer_chatgpt = "1", "2", "3", "4"

    inputs.append(gr.inputs.Checkbox(label=(question+":"+context_cs)))
    #responses.append(input_choices)  # Add choices to selected_responses

def evaluate_responses(*preferred_responses):
    print(preferred_responses)
    # count how many times context was found relevant
    relevant = sum(preferred_responses)
    not_relevant = len(preferred_responses)-relevant

    summary = f" You found {relevant} of the context paragraphs relevant and {not_relevant} of the context paragraphs not relevant."
    return summary


interface = gr.Interface(fn=evaluate_responses, inputs=inputs, outputs=outputs, title="AI Chatbot Evaluation",
                         description="Evaluate responses from pipeline and ChatGPT",
                         theme="compact", allow_flagging="never")

interface.launch()