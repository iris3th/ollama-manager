# Voice conversation setup

The app does not bundle speech models. It calls tools installed on the host, so
the Rust binary stays small and you can choose the models you like.

## Host tools

| Purpose | Default tool | Notes |
| --- | --- | --- |
| Record microphone | `parec`, `pw-record`, or `arecord` | Auto-detected; `parec` is preferred on Debian, `arecord` comes from `alsa-utils`. |
| Speech-to-text | Parakeet Redux (ONNX) or `whisper-cli` | Parakeet runs through `python3` and ONNX Runtime; Whisper needs a `ggml-*.bin` model. |
| Text-to-speech | `piper` | Plus an `.onnx` voice, e.g. `en_US-lessac-medium`. |

The app uses these local models for the whole conversation:
**Parakeet Redux** or **Whisper** (speech → text) → **Ollama** (text → reply) →
**Piper** (reply → speech).

On Arch/CachyOS something like:

```bash
sudo pacman -S alsa-utils
# whisper.cpp and piper are available from the AUR / extra, depending on your setup
paru -S whisper.cpp piper-tts
```

On Debian/Ubuntu, the minimum is:

```bash
sudo apt update
sudo apt install alsa-utils libasound2-plugins
```

Whisper.cpp is usually not packaged on Debian, so build it from source. Piper
can be installed with `pipx`. See
[README.md](README.md#debian--ubuntu-setup) for the full Debian instructions.

Then download a Whisper model and a Piper voice, for example:

```bash
mkdir -p ~/.local/share/whisper ~/.local/share/piper
# whisper.cpp model (tiny.en is fastest, base.en is more accurate)
curl -L -o ~/.local/share/whisper/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Piper voice
curl -L -o ~/.local/share/piper/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o ~/.local/share/piper/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## Parakeet Redux (default speech-to-text)

Parakeet Redux is a compact ONNX model that runs on the CPU through ONNX
Runtime. It needs `python3` plus `onnxruntime` and `numpy`.

```bash
sudo apt install python3 python3-venv

# A dedicated virtualenv keeps onnxruntime out of the system Python.
python3 -m venv ~/.local/share/parakeet-venv
~/.local/share/parakeet-venv/bin/pip install --upgrade pip
~/.local/share/parakeet-venv/bin/pip install "onnxruntime>=1.22" numpy
```

Do not use `pip3 install` or `sudo pip3 install` for the system Python: Debian
marks it as externally managed (PEP 668) and the install fails with
`error: externally-managed-environment`. Always use the venv's `pip` path.

Download the model files (about 416 MB) in one of two ways.

**Hugging Face CLI installed in the venv:**

```bash
~/.local/share/parakeet-venv/bin/pip install huggingface_hub

mkdir -p ~/.local/share/parakeet
~/.local/share/parakeet-venv/bin/huggingface-cli download \
  eschmidbauer/parakeet-redux-onnx \
  --local-dir ~/.local/share/parakeet
```

If that command is not found, newer `huggingface_hub` versions use `hf`:

```bash
~/.local/share/parakeet-venv/bin/hf download \
  eschmidbauer/parakeet-redux-onnx \
  --local-dir ~/.local/share/parakeet
```

**Plain `curl`, without any Hugging Face tooling:**

```bash
mkdir -p ~/.local/share/parakeet
cd ~/.local/share/parakeet

for file in \
  transcribe.py config.json vocab.txt \
  preprocessor.onnx vad-model.onnx decoder_joint-model.onnx encoder-model.onnx
do
  curl -L --fail --retry 3 \
    -o "$file" \
    "https://huggingface.co/eschmidbauer/parakeet-redux-onnx/resolve/main/$file"
done
```

Test it:

```bash
~/.local/share/parakeet-venv/bin/python \
  ~/.local/share/parakeet/transcribe.py /tmp/voice-test.wav
```

The app looks for a virtualenv in `<parakeet folder>/venv/bin/python` or
`~/.local/share/parakeet-venv/bin/python`, and falls back to `python3`. That way
ONNX Runtime can live in a virtualenv without activating it first. To use a
different interpreter, set a custom **Speech-to-text command** with `{python}`:

```text
/path/to/python {parakeet_script} --model-dir {parakeet_dir} {wav}
```

## Using it

1. Open **Settings → Voice**.
2. Check the **Detected tools** line.
3. Choose the **Speech-to-text engine** (**Parakeet Redux** by default, **Auto**,
   or **Whisper**) and set the matching model path. Set the Piper voice path
   too.
4. **Speak assistant replies with Piper** is enabled by default. You can turn it
   off here or with the **Speak replies** checkbox in the chat bar.
   Optionally also enable **Send transcription automatically** — speak, stop,
   and the message is sent without pressing Send.
5. For a hands-free loop, enable **Voice conversation mode**. The app then
   listens, transcribes with the selected engine, sends the text to Ollama,
   speaks the reply with Piper, and starts listening again. If a transcript is
   ready while the previous answer is still generating, it is queued and sent
   as soon as that answer finishes; the app does not start listening again and
   lose the question. Recording is disabled while a transcript is being
   processed.
6. Enable **Voice only — speak replies without showing the text in the chat**
   if you do not want the assistant's answer written on screen. The reply is
   still stored for conversation context and is spoken normally.

In push-to-talk mode, click **🎤 Speak**, talk, then click
**■ Stop & transcribe**. The transcript appears in the message box (or is sent
immediately if auto-send is enabled). **🔊 Speak reply** speaks the last answer
again. While audio is playing, **🔇 Stop speaking** cuts it off immediately.

While a reply is being generated, **■ Stop** stops it. That question/answer pair
is marked “⏹ stopped” and excluded from later requests, so the model does not
answer the stopped question again when you ask something new.

## Command templates

Commands run through `sh -c`, so pipes work. These placeholders are expanded and
shell-quoted:

- `{wav}` — temporary recording file
- `{model}` — configured Whisper or Piper model path
- `{lang}` — configured language
- `{parakeet_dir}` — folder containing the Parakeet ONNX export
- `{parakeet_script}` — path to Parakeet's `transcribe.py`
- `{python}` — interpreter used for Parakeet (virtualenv if found)

Text-to-speech commands receive the text on **stdin**.

A custom speech-to-text command overrides the engine selector. When the field is
empty, the app uses the selected engine:

```text
# Whisper
whisper-cli -m {model} -f {wav} -l {lang} -nt

# Parakeet Redux
{python} {parakeet_script} --model-dir {parakeet_dir} {wav}
```

When the TTS command is empty, the app auto-detects Piper and plays its output
with `paplay`, `pw-play`, or `aplay` (in that order). A manual Piper command
should look like:

```text
piper --model {model} --output_file /tmp/piper.wav && paplay /tmp/piper.wav
```

Avoid piping Piper's raw output directly into `aplay` on PipeWire systems;
`aplay` may fail with an ALSA device error even though `paplay`/`pw-play` work.
