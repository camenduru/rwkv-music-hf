import gradio as gr
from musiclib import musicgen
from io import BytesIO
import midi_util
from midi_util import VocabConfig

def gen(piano_only):
    midi = ''
    for item in musicgen(piano_only=piano_only):
        midi = item
        yield item, None
    bio = BytesIO()
    cfg = VocabConfig.from_json('./vocab_config.json')
    text = midi.strip()
    mid = midi_util.convert_str_to_midi(cfg, text)
    mid.save(bio)
    return midi, bio.getvalue()
with gr.Blocks() as demo:
    piano_only = gr.Checkbox(label="Piano Only")
    synth = gr.Button("Synthesize")
    txtout = gr.Textbox(interactive=False)
    fileout = gr.File(interactive=False)
    synth.click(gen, inputs=[piano_only], outputs=[txtout, fileout])
demo.queue().launch()