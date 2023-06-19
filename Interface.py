from entire_pipeline import*
import gradio as gr


#Interface
def chatbot(*input):
    if input:
        answer_pipeline, best_context = pipeline(input[2], method=input[0],n_contexts=input[1],chatgpt_prompt = input[3])
        answer_chatgpt = answer_generation(input[2], pipeline_mode=False)
        return best_context, answer_pipeline, answer_chatgpt

sim_measure = gr.inputs.Radio(['cs','weighted_cs','ann'], label='Similarity measure',default = 'cs' )
n_contexts = gr.Slider(1, 5, step=1, label="Number of context paragraphs", value = 5)
user_input = gr.inputs.Textbox(lines=7, label="User question")
chatgpt_prompt = gr.inputs.Textbox(lines=7, label="ChatGPT promt", default = "As a teacher's assistant for a machine learning course, your role is to assist students by answering their questions. You have over 20 years of experience and are regarded as one of the best in the field. You will be presented with a question and given several context paragraphs from the lecture notes. Your task is to answer the question as comprehensively and accurately as possible  Please thoroughly read the context paragraph and determine if any information is relevant to answer the question. Your answer should be based on your own high level of expertise and only use the information in the context paragraph if any is relevant. In case the question does not have any direct reference in the lecture notes, please state that the specific information is not mentioned in the lecture notes and proceed to answer the question based on your knowledge. You may give a long answer if the question requires it.")


inputs = [sim_measure, n_contexts, user_input, chatgpt_prompt]
outputs = [gr.outputs.Textbox(label="Most similar context"),gr.outputs.Textbox(label="Pipeline reply"), gr.outputs.Textbox(label="ChatGPT reply")]

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=False)


