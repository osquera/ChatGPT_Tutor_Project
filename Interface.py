from entire_pipeline import*
import gradio as gr


#Interface
def chatbot(*input):
    if input:
        answer_pipeline, best_context = pipeline(input[2], method=input[0],n_contexts=input[1],chatgpt_prompt = input[3])
        answer_chatgpt = answer_generation(input[2], pipeline_mode=False)
        return best_context, answer_pipeline, answer_chatgpt

sim_measure = gr.inputs.Radio(['cs','weighted_cs','ann'], label='Similarity measure',default = 'cs' )
n_contexts = gr.Slider(1, 5, step=1, label="Number of context paragraphs", default = 5)
user_input = gr.inputs.Textbox(lines=7, label="User question")
chatgpt_prompt = gr.inputs.Textbox(lines=7, label="ChatGPT promt", default = "You are a Teachers Assistant and you should answer the QUESTION using the information given in the CONTEXT, if the CONTEXT is unrelated, you should ignore it.")


inputs = [sim_measure, n_contexts, user_input, chatgpt_prompt]
outputs = [gr.outputs.Textbox(label="Most similar context"),gr.outputs.Textbox(label="Pipeline reply"), gr.outputs.Textbox(label="ChatGPT reply")]

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=False)


