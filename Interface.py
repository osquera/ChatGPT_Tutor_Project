from entire_pipeline import*
import gradio as gr


#Interface
def chatbot(*input):
    if input:

        answer_pipeline, best_context = pipeline(input[1], method=input[0])
        answer_chatgpt = answer_generation(input, pipeline_mode=False)
        return best_context, answer_pipeline, answer_chatgpt

inputs = [gr.inputs.Radio(['cs','weighted_cs','ann'], label='Similarity measure'),gr.inputs.Textbox(lines=7, label="User question")]
outputs = [gr.outputs.Textbox(label="Most similar context"),gr.outputs.Textbox(label="Pipeline reply"), gr.outputs.Textbox(label="ChatGPT reply")]

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=False)




