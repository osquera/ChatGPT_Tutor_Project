import random
from entire_pipeline import*
import gradio as gr

# a/b test for similarity score
def chatbot(input):
    if input:
        answer_pipeline_cs, _ = pipeline(input, method='cs')
        answer_pipeline_wcs, _ = pipeline(input, method='weighted_cs')
        answer_pipeline_ann, _ = pipeline(input, method='ann')
        answer_chatgpt = answer_generation(input, pipeline_mode=False)
        return answer_pipeline_cs, answer_pipeline_wcs, answer_pipeline_ann, answer_chatgpt


questions = [
    "What is the purpose of regularization in machine learning?",
    "What is the difference between supervised and unsupervised learning?",
    "What are the key steps in building a machine learning model?",
    "What is the role of feature engineering in machine learning?"
    #"What is the concept of overfitting in machine learning?",
    #"How does gradient descent work in training a neural network?",
    #"What is the trade-off between bias and variance in machine learning?",
    #"What is the purpose of cross-validation in machine learning?",
    #"What are some common evaluation metrics used in machine learning?",
   # "How can you handle missing data in a machine learning dataset?",
]

outputs = gr.outputs.Textbox()
inputs = []
responses = []  # Track selected responses
for question in questions:

    #uses the pipeline to genrerate responses
    answer_pipeline_cs, answer_pipeline_wcs, answer_pipeline_ann, answer_chatgpt = chatbot(question)

    #Just for testing without using the api
    #best_context, answer_pipeline, answer_chatgpt = "1", "2", "3"
    input_choices = [answer_pipeline_cs, answer_pipeline_wcs, answer_pipeline_ann, answer_chatgpt]
    #shuffles the answers
    shuffled_choices = sorted(input_choices, key=lambda k: random.random())
    inputs.append(gr.inputs.Radio(shuffled_choices, label=question, type="value"))
    responses.append(input_choices)  # Add choices to selected_responses

def evaluate_responses(*preferred_responses):
    # counnt how many times each answer was chosen
    count_cs = 0
    count_wcs = 0
    count_ann = 0
    count_chatgpt = 0
    for i, respons in enumerate(preferred_responses):
        if respons == responses[i][0]:
            count_cs += 1
        elif respons == responses[i][1]:
            count_wcs += 1

        elif respons == responses[i][2]:
            count_ann += 1

        elif respons == responses[i][3]:
            count_chatgpt += 1
    summary = f"{count_cs} of normal cosine, {count_wcs} of weighted cosine, {count_ann} of ANN and {count_chatgpt} of ChatGPT responses."
    return summary


interface = gr.Interface(fn=evaluate_responses, inputs=inputs, outputs=outputs, title="AI Chatbot Evaluation",
                         description="Evaluate responses from pipeline and ChatGPT",
                         theme="compact", allow_flagging="never")

interface.launch()