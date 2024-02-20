import gradio as gr
from musiclib import musicgen
def gen(piano_only):
    return musicgen(piano_only=piano_only)
with gr.Blocks() as demo:
    piano_only = gr.Checkbox(label="Piano Only")
    synth = gr.Button("Synthesize")
    fileout = gr.Textbox(interactive=False)
    synth.click(gen, inputs=[piano_only], outputs=[fileout])
demo.queue().launch()