import random
from entire_pipeline import*
import gradio as gr

# a/b test
def chatbot(input):
    if input:
        answer_pipeline, best_context = pipeline(input, method='cs')
        answer_chatgpt = answer_generation(input, pipeline_mode=False)
        return best_context, answer_pipeline, answer_chatgpt


with open('message1.txt') as f:
    lines = f.readlines()

# select questions to use. Use index in lines to choose.
questions = [q[:-1] for q in lines[:56]]


def pipeline(query: str, method: str = 'cs', n_contexts: int = 5, chatgpt_prompt = "As a teacher's assistant for a machine learning course, your role is to assist students by answering their questions. You have over 20 years of experience and are regarded as one of the best in the field. You will be presented with a question and given several context paragraphs from the lecture notes. Your task is to answer the question as comprehensively and accurately as possible  Please thoroughly read the context paragraph and determine if any information is relevant to answer the question. Your answer should be based on your own high level of expertise and only use the information in the context paragraph if any is relevant. In case the question does not have any direct reference in the lecture notes, please state that the specific information is not mentioned in the lecture notes and proceed to answer the question based on your knowledge. You may give a long answer if the question requires it."):
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
    # 3. Answer Generation
    answer = answer_generation(query, best_ctx, chatgpt_prompt)
    # 4. Return the answer
    return answer, best_ctx


outputs = gr.outputs.Textbox()
inputs = []
responses = []  # Track selected responses
for question in questions:

    #uses the pipeline to genrerate responses
    best_context, answer_pipeline, answer_chatgpt = chatbot(question)

    #Just for testing without using the api
    #best_context, answer_pipeline, answer_chatgpt = "1", "2", "3"
    input_choices = [answer_pipeline, answer_chatgpt]
    #shuffles the answers
    shuffled_choices = sorted(input_choices, key=lambda k: random.random())
    inputs.append(gr.inputs.Radio(shuffled_choices, label=question, type="value"))
    responses.append(input_choices)  # Add choices to selected_responses

def evaluate_responses(*preferred_responses):
    # count how many times each answer was chosen
    count_pipeline = 0
    count_chatgpt = 0
    for i, respons in enumerate(preferred_responses):
        if respons == responses[i][0]:
            count_pipeline += 1
        elif respons == responses[i][1]:
            count_chatgpt += 1

    summary = f"You preferred {count_pipeline} response(s) from the pipeline and {count_chatgpt} response(s) from ChatGPT."
    return summary


interface = gr.Interface(fn=evaluate_responses, inputs=inputs, outputs=outputs, title="AI Chatbot Evaluation",
                         description="Evaluate responses from pipeline and ChatGPT",
                         theme="compact", allow_flagging="never")

interface.launch()