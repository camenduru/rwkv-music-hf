import gradio as gr
from musiclib import musicgen
import spaces
@spaces.GPU(enable_queue=True)
def gen(piano_only):
    for item in musicgen(piano_only=piano_only):
        yield item
with gr.Blocks() as demo:
    piano_only = gr.Checkbox(label="Piano Only")
    synth = gr.Button("Synthesize")
    fileout = gr.Textbox(interactive=False)
    synth.click(gen, inputs=[piano_only], outputs=[fileout])
demo.queue().launch()