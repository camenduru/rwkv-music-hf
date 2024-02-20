import gradio as gr
from musiclib import musicgen
from io import BytesIO
import midi_util
from midi_util import VocabConfig
import tempfile
from glob import glob
import librosa
from midi2audio import FluidSynth
fs = FluidSynth()
def trim_silence(filename, threshold=0.1, duration=0.5):
    y, sr = librosa.load(filename, sr=None)
    silent_frames = librosa.effects.split(y, top_db=threshold)
    total_duration = librosa.get_duration(y=y, sr=sr)
    duration_samples = int(duration * sr)
    trimmed_audio = []
    for start, end in silent_frames:
        if end - start > duration_samples:
            trimmed_audio.extend(y[start:end])
    trimmed_audio = np.array(trimmed_audio)
    return trimmed_audio, sr
def gen(piano_only, length):
    midi = ''
    for item in musicgen(piano_only=piano_only, length=length):
        midi = item
        yield item, None, None
    bio = BytesIO()
    cfg = VocabConfig.from_json('./vocab_config.json')
    text = midi.strip()
    mid = midi_util.convert_str_to_midi(cfg, text)
    with tempfile.NamedTemporaryFile(suffix='.midi', delete=False) as tmp, tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as aud:
        mid.save(tmp.name)
        fs.midi_to_audio(tmp.name, aud.name)
        trim_silence(aud.name)
        yield midi, tmp.name, aud.name
with gr.Blocks() as demo:
    gr.Markdown("# RWKV 4 Music (MIDI)\n\nThis demo uses the RWKV 4 MIDI model available [here](https://huggingface.co/BlinkDL/rwkv-4-music/blob/main/RWKV-4-MIDI-560M-v1-20230717-ctx4096.pth). Details may be found [here](https://huggingface.co/BlinkDL/rwkv-4-music). The music generation code may be found [here](https://github.com/BlinkDL/ChatRWKV/tree/main/music). The MIDI Tokenizer may be found [here](https://github.com/briansemrau/MIDI-LLM-tokenizer).\n\nNot sure how to play MIDI files? I recommend using the open source [VLC Media Player](https://www.videolan.org/vlc/) with can play MIDI files using FluidSynth.")
    piano_only = gr.Checkbox(label="Piano Only")
    length = gr.Slider(label="Max Length (in tokens)", minimum=4, maximum=4096, step=1, value=512, info="The audio may still be shorter than this")
    synth = gr.Button("Synthesize")
    txtout = gr.Textbox(interactive=False, label="MIDI Tokens")
    fileout = gr.File(interactive=False, label="MIDI File", type="binary")
    audioout = gr.Audio(interactive=False, label="Audio")
    synth.click(gen, inputs=[piano_only, length], outputs=[txtout, fileout, audioout])
demo.queue(api_open=False).launch(show_api=False)