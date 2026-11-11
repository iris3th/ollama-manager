# Ollama Manager

A small desktop chat client for a local [Ollama](https://ollama.com/) server,
written in Rust with `egui`/`eframe`.

It supports:

- browsing, downloading, and deleting Ollama models
- local chat history stored in `chats/`
- reusable system prompts stored in `Prompts/`
- **voice conversation**: record your question, transcribe it locally with
  Parakeet Redux (ONNX Runtime) or Whisper, send it to Ollama, and hear the
  answer through Piper

The speech features call tools installed on the host. See
[SPEECH.md](SPEECH.md) for the full voice configuration and command templates.

## Build and run

Install Rust, then from this directory:

```bash
cargo build --release
./target/release/ollama-manager
```

`build.sh` builds the same binary in a Rust container:

```bash
./build.sh
./target/release/ollama-manager
```

Run the binary from the directory where you want the `chats/` and `Prompts/`
folders to live. Settings are stored in
`~/.config/ollama-manager/settings.json`.

## Debian / Ubuntu setup

Minimum packages for microphone input and audio playback:

```bash
sudo apt update
sudo apt install alsa-utils libasound2-plugins pulseaudio-utils
```

That gives you `arecord`, `aplay`, `parec`, and `paplay`. Piper provides the
voice; there is no synthetic fallback.

### Parakeet Redux speech-to-text (default)

Parakeet Redux is a small, fast ONNX model that runs through ONNX Runtime and
usually transcribes more accurately than Whisper `base.en` on a CPU. It is the
default speech-to-text engine. It needs `python3` and a downloaded model folder.

#### 1. Install the Python runtime

```bash
sudo apt install python3 python3-venv

# A dedicated virtualenv keeps onnxruntime out of the system Python.
python3 -m venv ~/.local/share/parakeet-venv
~/.local/share/parakeet-venv/bin/pip install --upgrade pip
~/.local/share/parakeet-venv/bin/pip install "onnxruntime>=1.22" numpy
```

Use the venv's `pip` as shown above. Do **not** run `pip3 install` or
`sudo pip3 install` against the system Python: Debian marks it as
“externally managed” (PEP 668) and the install fails with
`error: externally-managed-environment`. That is expected, not a bug in the
instructions.

#### 2. Download the model (about 416 MB)

Pick one of these two options. Option A is simpler; Option B needs no Python
packaging tools at all.

**Option A — Hugging Face CLI inside the venv**

Install the CLI into the same virtualenv (this works even when system `pip3`
refuses):

```bash
~/.local/share/parakeet-venv/bin/pip install huggingface_hub
```

Then run it with the venv's full path:

```bash
mkdir -p ~/.local/share/parakeet
~/.local/share/parakeet-venv/bin/huggingface-cli download \
  eschmidbauer/parakeet-redux-onnx \
  --local-dir ~/.local/share/parakeet
```

Newer versions of `huggingface_hub` ship the command as `hf` instead. If
`huggingface-cli` says “command not found”, use:

```bash
~/.local/share/parakeet-venv/bin/hf download \
  eschmidbauer/parakeet-redux-onnx \
  --local-dir ~/.local/share/parakeet
```

There is no need to activate the venv or install `pipx`. If the CLI is still
missing, check what the install created:

```bash
ls ~/.local/share/parakeet-venv/bin | grep -E 'hf|hugging'
```

**Option B — plain `curl` (no Hugging Face CLI)**

This downloads the same files directly and works with only `curl`:

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

Expected files and rough sizes:

| File | Size |
| --- | --- |
| `transcribe.py` | 18 KB |
| `config.json` | < 1 KB |
| `vocab.txt` | 92 KB |
| `preprocessor.onnx` | 1.2 MB |
| `vad-model.onnx` | 18 MB |
| `decoder_joint-model.onnx` | 70 MB |
| `encoder-model.onnx` | 328 MB |

#### 3. Test it directly

```bash
~/.local/share/parakeet-venv/bin/python \
  ~/.local/share/parakeet/transcribe.py /tmp/voice-test.wav
```

The app automatically uses a virtualenv at
`~/.local/share/parakeet-venv/bin/python` or
`<Parakeet folder>/venv/bin/python`, and falls back to `python3`. No activation
is needed.

Then open **Settings → Voice** and set **Parakeet folder** to
`~/.local/share/parakeet`. The line below the field tells you whether the files
were found. To use Whisper instead, change **Speech-to-text engine** to
**Whisper (whisper.cpp)**.

### Whisper speech-to-text (alternative)

Debian usually does **not** ship a `whisper.cpp` package, so `apt-cache search
whisper.cpp` returning nothing is expected. Build it from source:

```bash
sudo apt install build-essential cmake git
git clone --depth 1 https://github.com/ggml-org/whisper.cpp
cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build whisper.cpp/build -j"$(nproc)"
sudo install -m755 whisper.cpp/build/bin/whisper-cli /usr/local/bin/
```

Verify it is installed:

```bash
whisper-cli --help
```

Download a model:

```bash
mkdir -p ~/.local/share/whisper
curl -L -o ~/.local/share/whisper/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

### Piper text-to-speech (optional, better voice)

```bash
sudo apt install pipx
pipx install piper-tts
pipx ensurepath
```

Download a voice:

```bash
mkdir -p ~/.local/share/piper
curl -L -o ~/.local/share/piper/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o ~/.local/share/piper/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Then open **Settings → Voice** and set:

- **Speech-to-text engine** → **Parakeet Redux** (default)
- **Parakeet folder** → `~/.local/share/parakeet`
- **Whisper model** (only if you use Whisper) → `~/.local/share/whisper/ggml-base.en.bin`
- **Piper voice** → `~/.local/share/piper/en_US-lessac-medium.onnx`
- **Whisper language** → e.g. `en`

The **Detected tools** line should show `parec` or `pw-record`, `piper`, and
`paplay` or `pw-play`. The line under **Parakeet folder** confirms whether
`transcribe.py` and `encoder-model.onnx` were found.

## Using voice conversation

The voice pipeline uses local models:

**Parakeet Redux** or **Whisper** (speech → text) → **Ollama** (text → reply) → **Piper** (reply → speech)

No screen reader or system dictation is used. The app loads your local speech
model and Piper voice, sends the transcript to Ollama, and speaks the model's
answer.

### Conversation mode

1. Open **Settings → Voice** and choose the speech-to-text engine, model paths,
   and Piper voice.
2. Click **🎙 Voice conversation** in the chat bar (or enable it in Settings).
3. The app starts listening. Speak, then click **■ Stop & send**.
4. It transcribes with Parakeet or Whisper, sends the text to Ollama, speaks the
   reply with Piper, then automatically starts listening for your next turn.
5. Click the toggle again to stop the conversation.

You do not need to type anything. Everything goes through the language model.
If the engine detects no speech (for example, silence), the app simply starts
listening again instead of ending the conversation. The mode is remembered
across restarts; if it was left on, the app resumes listening at startup.

If a transcript is ready while the previous answer is still being generated,
the app keeps it and sends it as soon as that answer finishes. It does not start
listening again while a transcript is waiting, so the question cannot be lost.
Recording is also disabled while a transcript is being processed.

### Spoken replies without text

If you do not want the assistant's answer written on screen, enable
**Voice only — speak replies without showing the text in the chat** in
**Settings → Voice**, or tick **Hide text** in the chat bar. The reply is still
kept in the conversation history so the model remembers it, but the chat shows
only a “🔊 Reply spoken aloud” placeholder. Errors are still shown.

This requires **Speak replies** (or conversation mode), and the app turns
speaking on automatically when you enable it.

### Stopping speech

While a reply is being spoken, the chat bar shows **🔇 Stop speaking**. Click
it to cut the audio immediately. In conversation mode the app then goes back to
listening; otherwise it just stops the playback and waits for you.

Starting a new reply, or clicking **🔊 Speak reply** again, also replaces the
current playback. Recording is disabled while speech is playing so the
microphone does not pick up the assistant's own voice.

### Stopping generation

While a reply is being generated, **■ Stop** appears next to the message box.
Clicking it stops the model and marks that question/answer pair as stopped. The
pair stays visible in the chat with a “⏹ stopped — not used in later replies”
note, but it is left out of later requests. The next question is therefore
answered on its own instead of the model trying to answer the stopped one again.
Sending a new message also stops any reply that is still being spoken.

Conversations saved by an older build are cleaned up automatically: an
interrupted exchange with an empty answer is also skipped. If an old chat
contains a *partial* answer from before this build, start a new chat once so the
model no longer sees that partial text.

### Push-to-talk

If you prefer manual control:

1. Click **🎤 Speak**.
2. Talk, then click **■ Stop & transcribe**.
3. The transcript appears in the message box, or is sent immediately if
   **Send transcription automatically** is enabled.
4. Replies are spoken automatically because **Speak replies** is enabled by
   default. Use the checkbox to turn it off, or **🔊 Speak reply** to repeat the
   last answer.
5. While audio is playing, **🔇 Stop speaking** stops it immediately.

## Testing speech tools

Test recording and transcription from a terminal:

```bash
# Record 5 seconds from the default microphone
arecord -d 5 -f S16_LE -r 16000 -c 1 /tmp/voice-test.wav

# Parakeet Redux (the app finds this venv automatically)
~/.local/share/parakeet-venv/bin/python \
  ~/.local/share/parakeet/transcribe.py /tmp/voice-test.wav

# or Whisper
whisper-cli -m ~/.local/share/whisper/ggml-base.en.bin \
  -f /tmp/voice-test.wav -l en -nt
```

If `arecord` fails with an ALSA device error, use PipeWire/PulseAudio
recording instead. `parec` is the most reliable on Debian:

```bash
# Stop with Ctrl+C
parec --rate=16000 --channels=1 --format=s16le --file-format=wav /tmp/voice-test.wav
```

If your `pw-record` supports `--container`, you can also use:

```bash
pw-record --rate 16000 --channels 1 --format s16 --container wav /tmp/voice-test.wav
```

The app auto-detects `parec`, then `pw-record` (only if it supports
`--container`), then `arecord` when the **Recording command** field is empty.

Test text-to-speech with Piper. Synthesize to a WAV file first and play it with
`paplay` or `pw-play`:

```bash
echo "hello from piper" | \
  piper --model ~/.local/share/piper/en_US-lessac-medium.onnx \
        --output_file /tmp/piper-test.wav

paplay /tmp/piper-test.wav
```

If `paplay` does not produce sound, try:

```bash
pw-play /tmp/piper-test.wav
```

Both are normally installed on a Debian desktop with PipeWire/PulseAudio:

```bash
sudo apt install pulseaudio-utils   # provides paplay
sudo apt install pipewire-bin       # provides pw-play
```

The app uses `paplay` / `pw-play` automatically for Piper audio.

Check the output device with:

```bash
aplay -l
speaker-test -c 2 -t sine -l 1
```

## Troubleshooting

**No Voice section in Settings**

Scroll down inside the Settings tab. Also make sure you are running the newly
built binary:

```bash
strings target/release/ollama-manager | grep -m1 "Speech-to-text command"
```

If nothing is printed, the binary is stale — rebuild and run
`./target/release/ollama-manager`.

**Recording fails / no microphone**

```bash
arecord -l
```

Make sure your user can access the audio device. On desktop Debian,
PipeWire or PulseAudio should provide the ALSA compatibility layer; install
`libasound2-plugins` if `arecord` cannot find a device.

If `arecord` fails, clear the **Recording command** field in
**Settings → Voice** so the app uses `parec` (or a compatible `pw-record`)
instead. You can test those directly with the commands in
**Testing speech tools** above.

**Transcription fails**

Check the **Speech-to-text engine** in **Settings → Voice**:

- **Parakeet Redux**: `transcribe.py`, `config.json`, `vocab.txt`,
  `preprocessor.onnx`, `vad-model.onnx`, `decoder_joint-model.onnx`, and
  `encoder-model.onnx` must exist in the **Parakeet folder**, and
  `onnxruntime` must be installed for the Python that the app uses. If the line
  under the folder says “Parakeet not ready”, download the files as shown in
  **Parakeet Redux speech-to-text** above.
- **Whisper**: `whisper-cli` must be on `PATH` and the model path must exist. If
  your binary is named differently, edit the **Speech-to-text command**
  template.

If the engine is set to **Auto**, Parakeet is used when its files are present,
otherwise Whisper.

**A later voice question is transcribed but not sent**

The app queues a transcript while the previous reply is still generating and
sends it when that reply finishes. It will not start listening again while a
transcript is waiting. If a question still seems to disappear, run the app with
diagnostics enabled and reproduce the problem:

```bash
OLLAMA_MANAGER_DEBUG=1 ./target/release/ollama-manager
```

The log is written to `/tmp/ollama-manager-debug.log`. It records the transcript,
whether the question was sent or queued, the chat request messages, and the
voice-loop state. Only these events are logged; the file is not written unless
`OLLAMA_MANAGER_DEBUG` is set.

If the log says `The model returned an empty reply`, the question reached Ollama
but the model produced no text. Try another model or a shorter question. The app
no longer re-speaks an older answer in that case; it shows the empty-reply
status and starts listening again.

If installing `onnxruntime` fails with `error: externally-managed-environment`,
you used the system Python. Install it into the venv instead and let the app
find that venv automatically:

```bash
~/.local/share/parakeet-venv/bin/pip install "onnxruntime>=1.22" numpy
```

If you see:

```text
Speech-to-text failed: error: 1 input file(s) failed
```

the recorder wrote a WAV file that the speech engine could not read. The app
prints more detail after the message, for example:

```text
Speech-to-text failed (recording is 48044 bytes, WAV): error: failed to read audio file '...' | error: 1 input file(s) failed
```

The app also repairs the WAV header before transcription. `parec` writes a
placeholder header when it starts and only fills in the sizes when it exits
cleanly; if it is killed, the engine can reject the file. Rebuild the latest
package if you see this message.

The last failed recording is kept for inspection at:

```text
/tmp/ollama-manager-last-failed.wav
```

Test it directly:

```bash
file /tmp/ollama-manager-last-failed.wav

~/.local/share/parakeet-venv/bin/python \
  ~/.local/share/parakeet/transcribe.py \
  /tmp/ollama-manager-last-failed.wav

whisper-cli -m ~/.local/share/whisper/ggml-base.en.bin \
  -f /tmp/ollama-manager-last-failed.wav -l en -nt
```

If that works, the app should work too after rebuilding. If it fails with
`failed to read audio data as wav`, the file is corrupt — check your recorder
and microphone.

**TTS fails or is silent**

Run the test in **Settings → Voice → Test voice**. The result is shown in the
status line at the bottom of the sidebar and inside the Voice section. The app
uses Piper and does not fall back to a robotic synthesizer.

Test the tools outside the app as shown in **Testing speech tools** above.
The app auto-detects `paplay`, `pw-play`, then `aplay` for Piper, so it should
work on PipeWire/PulseAudio even when plain `aplay` fails.

If you set a custom TTS command, use a real WAV file and a player that works on
your system, for example:

```text
piper --model {model} --output_file /tmp/piper.wav && paplay /tmp/piper.wav
```
