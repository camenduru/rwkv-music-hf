import gradio as gr
import numpy as np
from musiclib import musicgen
from io import BytesIO
import midi_util
from midi_util import VocabConfig
import tempfile
from glob import glob
import librosa
from midi2audio import FluidSynth
fs = FluidSynth()

def gen(piano_only, piano_seed, length):
    midi = ''
    if piano_seed:
        ccc = 'v:5b:3 v:5b:2 t125 t125 t125 t106 pi:43:5 t24 pi:4a:7 t15 pi:4f:7 t17 pi:56:7 t18 pi:54:7 t125 t49 pi:51:7 t117 pi:4d:7 t125 t125 t111 pi:37:7 t14 pi:3e:6 t15 pi:43:6 t12 pi:4a:7 t17 pi:48:7 t125 t60 pi:45:7 t121 pi:41:7 t125 t117 s:46:5 s:52:5 f:46:5 f:52:5 t121 s:45:5 s:46:0 s:51:5 s:52:0 f:45:5 f:46:0 f:51:5 f:52:0 t121 s:41:5 s:45:0 s:4d:5 s:51:0 f:41:5 f:45:0 f:4d:5 f:51:0 t102 pi:37:0 pi:3e:0 pi:41:0 pi:43:0 pi:45:0 pi:48:0 pi:4a:0 pi:4d:0 pi:4f:0 pi:51:0 pi:54:0 pi:56:0 t19 s:3e:5 s:41:0 s:4a:5 s:4d:0 f:3e:5 f:41:0 f:4a:5 f:4d:0 t121 v:3a:5 t121 v:39:7 t15 v:3a:0 t106 v:35:8 t10 v:39:0 t111 v:30:8 v:35:0 t125 t117 v:32:8 t10 v:30:0 t125 t125 t103 v:5b:0 v:5b:0 t9 pi:4a:7'
    else:
        ccc = '<pad>'
    for item in musicgen(ccc, piano_only=piano_only, length=length):
        midi = item
        yield item, None, None
    bio = BytesIO()
    cfg = VocabConfig.from_json('./vocab_config.json')
    text = midi.strip()
    mid = midi_util.convert_str_to_midi(cfg, text)
    with tempfile.NamedTemporaryFile(suffix='.midi', delete=False) as tmp, tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as aud:
        mid.save(tmp.name)
        fs.midi_to_audio(tmp.name, aud.name)
        yield midi, tmp.name, aud.name
with gr.Blocks() as demo:
    gr.Markdown("# RWKV 4 Music (MIDI)\n\nThis demo uses the RWKV 4 MIDI model available [here](https://huggingface.co/BlinkDL/rwkv-4-music/blob/main/RWKV-4-MIDI-560M-v1-20230717-ctx4096.pth). Details may be found [here](https://huggingface.co/BlinkDL/rwkv-4-music). The music generation code may be found [here](https://github.com/BlinkDL/ChatRWKV/tree/main/music). The MIDI Tokenizer may be found [here](https://github.com/briansemrau/MIDI-LLM-tokenizer).\n\nNot sure how to play MIDI files? I recommend using the open source [VLC Media Player](https://www.videolan.org/vlc/) with can play MIDI files using FluidSynth.")
    piano_only = gr.Checkbox(label="Piano Only")
    piano_seed = gr.Checkbox(label="Use Piano Melody Seed")
    length = gr.Slider(label="Max Length (in tokens)", minimum=4, maximum=4096, step=1, value=512, info="The audio may still be shorter than this")
    synth = gr.Button("Synthesize")
    txtout = gr.Textbox(interactive=False, label="MIDI Tokens")
    fileout = gr.File(interactive=False, label="MIDI File", type="binary")
    audioout = gr.Audio(interactive=False, label="Audio")
    synth.click(gen, inputs=[piano_only, piano_seed, length], outputs=[txtout, fileout, audioout])
demo.queue(api_open=False).launch(show_api=False)