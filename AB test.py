import random
from entire_pipeline import*
import gradio as gr

# a/b test
def chatbot(input):
    if input:
        answer_pipeline, best_context = pipeline(input, method='cs')
        answer_chatgpt = answer_generation(input, pipeline_mode=False)
        return best_context, answer_pipeline, answer_chatgpt


questions = [
    "What is the purpose of regularization in machine learning?",
    "What is the difference between supervised and unsupervised learning?",
    "What are the key steps in building a machine learning model?",
    "What is the role of feature engineering in machine learning?",
    "What is the concept of overfitting in machine learning?",
    "How does gradient descent work in training a neural network?",
    "What is the trade-off between bias and variance in machine learning?",
    "What is the purpose of cross-validation in machine learning?",
    "What are some common evaluation metrics used in machine learning?",
    "How can you handle missing data in a machine learning dataset?",
]


def pipeline(query: str, method: str = 'cs', n_contexts: int = 5, chatgpt_prompt = "You are a Teachers Assistant and you should answer the QUESTION using the information given in the CONTEXT, if the CONTEXT is unrelated, you should ignore it."):
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