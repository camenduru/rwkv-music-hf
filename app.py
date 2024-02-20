import gradio as gr
from musiclib import musicgen
from io import BytesIO
import midi_util
from midi_util import VocabConfig
def gen(piano_only, length):
    midi = ''
    for item in musicgen(piano_only=piano_only, length=length):
        midi = item
        yield item, None
    bio = BytesIO()
    cfg = VocabConfig.from_json('./vocab_config.json')
    text = midi.strip()
    mid = midi_util.convert_str_to_midi(cfg, text)
    mid.save(file=bio)
    yield midi, bio.getvalue()
with gr.Blocks() as demo:
    piano_only = gr.Checkbox(label="Piano Only")
    length = gr.Slider(label="Length (in tokens)", minimum=4, maximum=4096, step=1, value=4096)
    synth = gr.Button("Synthesize")
    txtout = gr.Textbox(interactive=False, label="MIDI Length")
    fileout = gr.File(interactive=False, label="MIDI File")
    synth.click(gen, inputs=[piano_only, length], outputs=[txtout, fileout])
demo.queue().launch()