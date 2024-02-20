import gradio as gr
from musiclib import musicgen
from io import BytesIO
import midi_util
from midi_util import VocabConfig
from midi2audio import FluidSynth
fs = FluidSynth()
def gen(piano_only, length):
    midi = ''
    for item in musicgen(piano_only=piano_only, length=length):
        midi = item
        yield item, None, None
    bio = BytesIO()
    cfg = VocabConfig.from_json('./vocab_config.json')
    text = midi.strip()
    mid = midi_util.convert_str_to_midi(cfg, text)
    mid.save(file=bio)
    audio = BytesIO()
    fs.midi_to_audio(bio.getvalue(), audio)
    return midi, bio.getvalue(), audio.getvalue()
with gr.Blocks() as demo:
    piano_only = gr.Checkbox(label="Piano Only")
    length = gr.Slider(label="Length", minimum=4, maximum=4096, step=1, value=4096)
    synth = gr.Button("Synthesize")
    txtout = gr.Textbox(interactive=False)
    fileout = gr.File(interactive=False)
    audioout = gr.Audio(interactive=False)
    synth.click(gen, inputs=[piano_only, length], outputs=[txtout, fileout, audioout])
demo.queue().launch()