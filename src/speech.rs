//! Speech-to-text and text-to-speech helpers.
//!
//! The app deliberately shells out to well-known host tools instead of
//! bundling large models into the Rust binary:
//!
//!   * recording: `parec` / `pw-record` / `arecord`
//!   * STT:       Parakeet Redux ONNX (via `python3 transcribe.py`) or
//!                `whisper-cli` with a local Whisper `ggml-*.bin` model
//!   * TTS:       `piper` with a local Piper `.onnx` voice model
//!
//! Every command is a template. The following placeholders are expanded
//! and shell-quoted before the command is run through `sh -c`:
//!
//!   {wav}             recorded WAV file
//!   {model}           configured STT/TTS model path
//!   {lang}            configured language
//!   {parakeet_dir}    folder containing the Parakeet ONNX export
//!   {parakeet_script} path to Parakeet's `transcribe.py`
//!
//! TTS commands receive the text to speak on stdin, so a command such as
//! `piper --model {model} --output-raw | aplay -q -r 22050 -f S16_LE -t raw -`
//! works without any temporary files.

use crate::SpeechSettings;

use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicI32, Ordering},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Returned by [`speak`] when playback was stopped on purpose.
pub(crate) const STOPPED: &str = "speech stopped";

/// PID of the text-to-speech process group currently playing, or 0.
static SPEECH_PID: AtomicI32 = AtomicI32::new(0);
/// PID that the user asked to stop, so `speak` can report it as intentional.
static SPEECH_STOPPED_PID: AtomicI32 = AtomicI32::new(0);

/// Clears the current speech PID when playback ends, on any return path.
struct SpeechPidGuard {
    pid: i32,
}

impl SpeechPidGuard {
    fn new(pid: i32) -> Self {
        SPEECH_PID.store(pid, Ordering::SeqCst);
        Self { pid }
    }
}

impl Drop for SpeechPidGuard {
    fn drop(&mut self) {
        let _ = SPEECH_PID.compare_exchange(
            self.pid,
            0,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }
}

/// A recording currently in progress.
pub(crate) struct Recording {
    child: Child,
    path: PathBuf,
    started: Instant,
}

impl Recording {
    /// Start the recorder configured in `settings`.
    pub fn start(settings: &SpeechSettings) -> Result<Self, String> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        let path = std::env::temp_dir().join(format!(
            "ollama-manager-recording-{stamp}.wav"
        ));

        let configured = settings.record_command.trim();
        let template = if configured.is_empty()
            || configured == DEFAULT_RECORD_COMMAND
        {
            // Treat the old hard-coded default as "auto" too, so existing
            // settings files also benefit from PipeWire/PulseAudio capture.
            record_command()
        } else {
            configured.to_owned()
        };

        let command = render_command(&template, &[("wav", &path.to_string_lossy())]);

        let mut process = shell_command(&command);

        // Put the recorder in its own process group so stopping it also stops
        // any child process it spawned (e.g. a shell wrapper around `parec`).
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            process.process_group(0);
        }

        let mut child = process
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Could not start recorder: {e}"))?;

        // Catch missing devices / bad commands immediately instead of
        // showing a recording indicator forever.
        thread::sleep(Duration::from_millis(200));

        if let Ok(Some(status)) = child.try_wait() {
            let mut error = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut error);
            }
            let _ = fs::remove_file(&path);

            return Err(format!(
                "Recorder exited immediately ({status}): {}",
                error.trim()
            ));
        }

        Ok(Self {
            child,
            path,
            started: Instant::now(),
        })
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// `false` means the recorder exited on its own (device unplugged, etc.).
    pub fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Stop the recorder, finalise the WAV header and return the file.
    pub fn stop(mut self) -> Result<PathBuf, String> {
        stop_child(&mut self.child);

        let status = self
            .child
            .wait()
            .map_err(|e| format!("Could not wait for recorder: {e}"))?;

        let size = fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);

        if !self.path.exists() || size < 1024 {
            let _ = fs::remove_file(&self.path);
            return Err(format!(
                "Recording was empty ({status}). Check your microphone and the record command."
            ));
        }

        // `parec` writes the WAV header when it starts and only fills in the
        // RIFF/data sizes when it exits cleanly. Stopping it with SIGINT leaves
        // placeholder sizes, which makes Whisper reject the file with
        // "error: 1 input file(s) failed". Repair the sizes from the real file.
        if let Err(e) = finalize_wav(&self.path) {
            let _ = fs::remove_file(&self.path);
            return Err(format!("Could not read the recording: {e}"));
        }

        // Drop would delete the file, so hand ownership to the caller.
        let path = self.path.clone();
        self.path = PathBuf::new();
        Ok(path)
    }

    pub fn cancel(mut self) {
        #[cfg(unix)]
        kill_recorder(&self.child);

        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            #[cfg(unix)]
            kill_recorder(&self.child);

            let _ = self.child.kill();
            let _ = self.child.wait();
        }

        if !self.path.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub(crate) const DEFAULT_STT_COMMAND: &str =
    "whisper-cli -m {model} -f {wav} -l {lang} -nt";

/// Folder containing the Parakeet Redux ONNX export and `transcribe.py`.
pub(crate) fn parakeet_dir(settings: &SpeechSettings) -> PathBuf {
    let configured = settings.parakeet_dir.trim();

    if configured.is_empty() {
        PathBuf::from(expand_tilde("~/.local/share/parakeet"))
    } else {
        PathBuf::from(expand_tilde(configured))
    }
}

fn command_on_path(name: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths)
                .any(|directory| directory.join(name).is_file())
        })
        .unwrap_or(false)
}

/// Python interpreter used to run Parakeet.
///
/// ONNX Runtime is often installed in a virtualenv, so prefer one next to the
/// model files or the conventional `~/.local/share/parakeet-venv`.
pub(crate) fn parakeet_python(settings: &SpeechSettings) -> String {
    let candidates = [
        parakeet_dir(settings).join("venv/bin/python"),
        PathBuf::from(expand_tilde("~/.local/share/parakeet-venv/bin/python")),
    ];

    for candidate in candidates {
        if candidate.is_file() {
            return candidate.to_string_lossy().into_owned();
        }
    }

    "python3".to_owned()
}

/// Files that must be present in the Parakeet folder.
const PARAKEET_REQUIRED: &[&str] = &[
    "transcribe.py",
    "encoder-model.onnx",
    "decoder_joint-model.onnx",
    "preprocessor.onnx",
    "vocab.txt",
];

/// True when the Parakeet Redux ONNX files and Python runtime are present.
pub(crate) fn parakeet_ready(settings: &SpeechSettings) -> bool {
    let directory = parakeet_dir(settings);
    let python = parakeet_python(settings);

    (Path::new(&python).is_file() || command_on_path("python3"))
        && PARAKEET_REQUIRED
            .iter()
            .all(|name| directory.join(name).is_file())
}

/// Which engine `transcribe` will use with the current settings.
pub(crate) fn active_stt_engine(settings: &SpeechSettings) -> &'static str {
    match settings.stt_engine.as_str() {
        "whisper" => "Whisper",
        "parakeet" => "Parakeet Redux",
        _ if parakeet_ready(settings) => "Parakeet Redux",
        _ => "Whisper",
    }
}

/// Choose the speech-to-text command template for the current settings.
fn stt_template(settings: &SpeechSettings) -> Result<String, String> {
    let configured = settings.stt_command.trim();
    // A saved template only wins when the user actually changed it. The old
    // default is treated as "automatic", so existing settings files do not
    // force Whisper after Parakeet has been installed.
    let use_custom = !configured.is_empty() && configured != DEFAULT_STT_COMMAND;

    let template = if use_custom {
        configured.to_owned()
    } else if active_stt_engine(settings) == "Parakeet Redux" {
        if !parakeet_ready(settings) {
            return Err(format!(
                "Parakeet Redux is not set up. Expected transcribe.py and the ONNX model files in {}. See SPEECH.md for download instructions.",
                parakeet_dir(settings).display()
            ));
        }

        "{python} {parakeet_script} --model-dir {parakeet_dir} {wav}".to_owned()
    } else {
        DEFAULT_STT_COMMAND.to_owned()
    };

    if template.contains("{model}") && settings.stt_model.trim().is_empty() {
        return Err(
            "Set a Whisper model path in Settings → Voice before using speech-to-text."
                .to_owned(),
        );
    }

    Ok(template)
}

/// Transcribe a WAV file to plain text.
pub(crate) fn transcribe(
    wav: &Path,
    settings: &SpeechSettings,
) -> Result<String, String> {
    let template = stt_template(settings)?;

    let directory = parakeet_dir(settings);
    let python = parakeet_python(settings);

    let command = render_command(
        &template,
        &[
            ("wav", &wav.to_string_lossy()),
            ("model", &settings.stt_model),
            ("lang", &settings.language),
            ("parakeet_dir", &directory.to_string_lossy()),
            (
                "parakeet_script",
                &directory.join("transcribe.py").to_string_lossy(),
            ),
            ("python", &python),
        ],
    );

    let output = shell_command(&command)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("Could not run speech-to-text: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let lines: Vec<&str> = stderr
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();

        // The final line is usually the generic "N input file(s) failed";
        // the useful cause is just above it.
        let details = lines
            .iter()
            .rev()
            .take(4)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join(" | ");

        return Err(format!(
            "Speech-to-text failed ({}){}",
            describe_audio(wav),
            if details.is_empty() {
                String::new()
            } else {
                format!(": {details}")
            }
        ));
    }

    let text = clean_transcript(&String::from_utf8_lossy(&output.stdout));

    if text.is_empty() {
        Err("No speech detected.".to_owned())
    } else {
        Ok(text)
    }
}

/// Speak `text` using the configured TTS command.
pub(crate) fn speak(
    text: &str,
    settings: &SpeechSettings,
) -> Result<(), String> {
    let text = clean_for_speech(text);

    if text.trim().is_empty() {
        return Ok(());
    }

    let commands = resolve_tts_commands(settings);

    if commands.is_empty() {
        let model = expand_tilde(&settings.tts_model);

        if command_exists("piper") && !Path::new(&model).exists() {
            return Err(format!(
                "Piper is installed, but the voice file was not found at {model}. Set the correct path in Settings → Voice."
            ));
        }

        return Err(
            "No Piper voice model found. Install piper and set the .onnx voice path in Settings → Voice."
                .to_owned(),
        );
    }

    let mut errors = Vec::new();

    for template in commands {
        let command = render_command(&template, &[("model", &settings.tts_model)]);

        match run_tts_command(&command, &text) {
            Ok(()) => return Ok(()),
            Err(e) if e == STOPPED => return Err(e),
            Err(e) => errors.push(e),
        }
    }

    Err(format!("Text-to-speech failed: {}", errors.join(" | ")))
}

/// Stop the text-to-speech playback that is currently running, if any.
pub(crate) fn stop_speaking() {
    let pid = SPEECH_PID.swap(0, Ordering::SeqCst);

    if pid <= 0 {
        return;
    }

    SPEECH_STOPPED_PID.store(pid, Ordering::SeqCst);

    #[cfg(unix)]
    {
        // The TTS command runs in its own process group (`bash`, `piper`,
        // `paplay`, …). Kill the whole group so audio stops immediately.
        let _ = Command::new("kill")
            .arg("-KILL")
            .arg(format!("-{pid}"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

fn run_tts_command(command: &str, text: &str) -> Result<(), String> {
    // `bash` + pipefail so a failing `piper` cannot be hidden by a
    // successful (but silent) `aplay` at the end of a pipeline.
    let mut process = tts_shell_command(command);

    // A separate process group lets `stop_speaking` kill the whole pipeline.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        process.process_group(0);
    }

    SPEECH_STOPPED_PID.store(0, Ordering::SeqCst);

    let mut child = process
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Could not start text-to-speech: {e}"))?;

    let pid = child.id() as i32;
    let _pid_guard = SpeechPidGuard::new(pid);

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(text.as_bytes())
            .map_err(|e| format!("Could not send text to TTS: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Text-to-speech failed: {e}"))?;

    // Only report "stopped" if this exact playback was the one stopped.
    let stopped = SPEECH_STOPPED_PID
        .compare_exchange(pid, 0, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok();

    if stopped {
        return Err(STOPPED.to_owned());
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "{}",
            stderr.trim().lines().last().unwrap_or("unknown error")
        ));
    }

    Ok(())
}

/// Names of tools found on this host, for the settings screen.
pub(crate) fn detected_tools() -> Vec<&'static str> {
    [
        "arecord",
        "parec",
        "pw-record",
        "python3",
        "whisper-cli",
        "whisper",
        "piper",
        "espeak-ng",
        "espeak",
        "spd-say",
        "aplay",
        "paplay",
        "pw-play",
    ]
    .into_iter()
    .filter(|name| command_exists(name))
    .collect()
}

const DEFAULT_RECORD_COMMAND: &str =
    "arecord -q -f S16_LE -r 16000 -c 1 -t wav {wav}";

fn record_command() -> String {
    // `parec` is the most portable option on Debian with PipeWire/PulseAudio:
    // it can always write a proper WAV file.
    if command_exists("parec") {
        return "parec --rate=16000 --channels=1 --format=s16le --file-format=wav {wav}"
            .to_owned();
    }

    // Some `pw-record` builds are compiled without libsndfile support and do
    // not understand `--container`. Only use it when that flag exists.
    if command_exists("pw-record") && command_supports_flag("pw-record", "--container")
    {
        return "pw-record --rate 16000 --channels 1 --format s16 --container wav {wav}"
            .to_owned();
    }

    DEFAULT_RECORD_COMMAND.to_owned()
}

fn command_supports_flag(name: &str, flag: &str) -> bool {
    match Command::new(name).arg("--help").output() {
        Ok(output) => {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            text.contains(flag)
        }
        Err(_) => false,
    }
}

fn resolve_tts_commands(settings: &SpeechSettings) -> Vec<String> {
    if !settings.tts_command.trim().is_empty() {
        return vec![settings.tts_command.clone()];
    }

    let mut commands = Vec::new();
    let model = expand_tilde(&settings.tts_model);

    if command_exists("piper")
        && !model.trim().is_empty()
        && Path::new(&model).exists()
    {
        // Synthesize to a real WAV first, then let the desktop's normal
        // player handle it. This avoids raw-audio/sample-rate problems and
        // works with PulseAudio/PipeWire where plain `aplay` may fail.
        let player = if command_exists("paplay") {
            "paplay"
        } else if command_exists("pw-play") {
            "pw-play"
        } else {
            "aplay -q"
        };

        commands.push(format!(
            "tmp=$(mktemp --suffix=.wav); piper --model {{model}} --output_file \"$tmp\" && {player} \"$tmp\"; status=$?; rm -f \"$tmp\"; exit $status"
        ));
    }

    commands
}

fn render_command(template: &str, vars: &[(&str, &str)]) -> String {
    let mut command = template.to_owned();

    for (key, value) in vars {
        let value = shell_quote(&expand_tilde(value));
        command = command.replace(&format!("{{{key}}}"), &value);
    }

    command
}

fn expand_tilde(value: &str) -> String {
    if let Some(rest) = value.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().into_owned();
        }
    }

    value.to_owned()
}

fn shell_quote(value: &str) -> String {
    let safe = !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:".contains(c));

    if safe {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn shell_command(command: &str) -> Command {
    #[cfg(unix)]
    {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command);
        cmd
    }

    #[cfg(windows)]
    {
        let mut cmd = Command::new("cmd");
        cmd.arg("/C").arg(command);
        cmd
    }
}

fn tts_shell_command(command: &str) -> Command {
    let mut cmd = Command::new("bash");
    cmd.arg("-c").arg(format!("set -o pipefail; {command}"));
    cmd
}

fn command_exists(name: &str) -> bool {
    shell_command(&format!("command -v {name} >/dev/null 2>&1"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn stop_child(child: &mut Child) {
    // SIGINT asks the recorder to stop. Not every recorder finalises its WAV
    // header on the way out, so `finalize_wav` repairs the file afterwards.
    // Fall back to SIGKILL if it does not exit promptly.
    #[cfg(unix)]
    {
        signal_recorder(child, "-INT");

        for _ in 0..40 {
            if !matches!(child.try_wait(), Ok(None)) {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }

        signal_recorder(child, "-KILL");
    }

    let _ = child.kill();
}

#[cfg(unix)]
fn signal_recorder(child: &Child, signal: &str) {
    let _ = Command::new("kill")
        .arg(signal)
        .arg(format!("-{}", child.id()))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(unix)]
fn kill_recorder(child: &Child) {
    signal_recorder(child, "-KILL");
}

/// Short description of a recording, used in error messages.
fn describe_audio(path: &Path) -> String {
    let Ok(meta) = fs::metadata(path) else {
        return "recording file is missing".to_owned();
    };

    if meta.len() == 0 {
        return "recording is empty".to_owned();
    }

    let mut head = [0u8; 12];
    let kind = match fs::File::open(path).and_then(|mut file| file.read_exact(&mut head)) {
        Ok(()) if &head[0..4] == b"RIFF" && &head[8..12] == b"WAVE" => "WAV",
        Ok(()) => "unknown format",
        Err(_) => "unreadable",
    };

    format!("recording is {} bytes, {}", meta.len(), kind)
}

/// Make sure a finished recording is a complete, readable WAV file.
///
/// Recorders such as `parec` write a placeholder WAV header when they start and
/// only fill in the RIFF/data sizes when they exit cleanly. The app stops them
/// with SIGINT, so those sizes are often left as `0xFFFFFFFF`. Whisper then
/// rejects the file with `error: 1 input file(s) failed`. Rewrite the sizes
/// from the real file length. If the file has no WAV header at all, assume raw
/// 16 kHz mono 16-bit PCM and prepend one.
fn finalize_wav(path: &Path) -> Result<(), String> {
    let data = fs::read(path).map_err(|e| e.to_string())?;
    let len = data.len();

    let looks_like_wav =
        len >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE";

    if !looks_like_wav {
        return write_raw_as_wav(path, &data);
    }

    // Walk the RIFF chunks to find the audio payload.
    let mut pos = 12usize;
    let mut data_offset = None;

    while pos.saturating_add(8) <= len {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;

        if id == b"data" {
            data_offset = Some(pos + 8);
            break;
        }

        pos = pos.saturating_add(8).saturating_add(size + (size & 1));
    }

    let Some(data_offset) = data_offset else {
        return Err(
            "the recording is a WAV file without an audio data chunk"
                .to_owned(),
        );
    };

    let audio_len = len.saturating_sub(data_offset) as u32;

    if audio_len < 1600 {
        return Err(
            "the recording contained no audio (hold the button while you speak)"
                .to_owned(),
        );
    }

    let mut fixed = data;
    let riff_size = (len as u32).saturating_sub(8);
    fixed[4..8].copy_from_slice(&riff_size.to_le_bytes());

    let size_pos = data_offset - 4;
    fixed[size_pos..size_pos + 4].copy_from_slice(&audio_len.to_le_bytes());

    fs::write(path, fixed).map_err(|e| e.to_string())
}

fn write_raw_as_wav(path: &Path, pcm: &[u8]) -> Result<(), String> {
    if pcm.len() < 1600 {
        return Err(
            "the recording contained no audio (hold the button while you speak)"
                .to_owned(),
        );
    }

    let mut wav = wav_header(pcm.len() as u32);
    wav.extend_from_slice(pcm);

    fs::write(path, wav).map_err(|e| e.to_string())
}

fn wav_header(data_len: u32) -> Vec<u8> {
    let channels: u16 = 1;
    let sample_rate: u32 = 16_000;
    let bits: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * (bits as u32 / 8);
    let block_align = channels * (bits / 8);

    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&(36 + data_len).to_le_bytes());
    header.extend_from_slice(b"WAVE");
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes()); // PCM
    header.extend_from_slice(&channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&block_align.to_le_bytes());
    header.extend_from_slice(&bits.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data_len.to_le_bytes());
    header
}

fn clean_transcript(raw: &str) -> String {
    raw.lines()
        .map(str::trim)
        .map(strip_timestamp)
        .filter(|line| {
            !line.is_empty()
                && *line != "[BLANK_AUDIO]"
                && *line != "(silence)"
                && *line != "[silence]"
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_owned()
}

fn strip_timestamp(line: &str) -> &str {
    let trimmed = line.trim_start();

    if !trimmed.starts_with('[') {
        return line;
    }

    match trimmed.find(']') {
        Some(end) => trimmed[end + 1..].trim_start(),
        None => line,
    }
}

/// Strip Markdown and fenced code blocks so the assistant does not read
/// syntax out loud.
fn clean_for_speech(text: &str) -> String {
    let mut output = String::new();
    let mut in_code = false;

    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
            continue;
        }

        if in_code {
            continue;
        }

        let cleaned = line
            .trim_start_matches(|c: char| {
                c == '#' || c == '>' || c == '-' || c == '*' || c == '+'
            })
            .trim();

        if !cleaned.is_empty() {
            output.push_str(cleaned);
            output.push('\n');
        }
    }

    output
        .replace("**", "")
        .replace("__", "")
        .replace('`', "")
        .replace("~~", "")
        .trim()
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_wav(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ollama-manager-test-{}-{name}.wav",
            std::process::id()
        ))
    }

    #[test]
    fn repairs_placeholder_wav_sizes() {
        let path = temp_wav("repair");
        let pcm = vec![1u8; 3200];

        let mut data = wav_header(pcm.len() as u32);
        data[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        data[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&pcm);

        fs::write(&path, &data).unwrap();
        finalize_wav(&path).unwrap();

        let fixed = fs::read(&path).unwrap();
        let riff_size = u32::from_le_bytes(fixed[4..8].try_into().unwrap());
        let data_size = u32::from_le_bytes(fixed[40..44].try_into().unwrap());

        assert_eq!(riff_size, fixed.len() as u32 - 8);
        assert_eq!(data_size, pcm.len() as u32);
        assert_eq!(&fixed[44..], &pcm[..]);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn wraps_raw_pcm_in_a_wav_header() {
        let path = temp_wav("raw");
        let pcm = vec![2u8; 3200];
        fs::write(&path, &pcm).unwrap();

        finalize_wav(&path).unwrap();

        let fixed = fs::read(&path).unwrap();
        assert_eq!(&fixed[0..4], b"RIFF");
        assert_eq!(&fixed[8..12], b"WAVE");
        assert_eq!(fixed.len(), 44 + pcm.len());
        assert_eq!(&fixed[44..], &pcm[..]);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_silent_recordings() {
        let path = temp_wav("silent");
        let data = wav_header(0);
        fs::write(&path, &data).unwrap();

        assert!(finalize_wav(&path).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn stop_speaking_interrupts_playback() {
        let mut settings = crate::SpeechSettings::default();
        settings.tts_command = "sleep 30".to_owned();

        let handle = thread::spawn(move || speak("hello", &settings));
        thread::sleep(Duration::from_millis(400));
        stop_speaking();

        let result = handle.join().unwrap();
        assert_eq!(result, Err(STOPPED.to_owned()));
    }

    #[test]
    fn normal_playback_is_not_reported_as_stopped() {
        let mut settings = crate::SpeechSettings::default();
        settings.tts_command = "cat >/dev/null".to_owned();

        assert_eq!(speak("hello", &settings), Ok(()));
    }

    #[test]
    fn old_default_command_is_treated_as_automatic() {
        let mut settings = crate::SpeechSettings::default();
        settings.stt_engine = "whisper".to_owned();
        settings.stt_command = DEFAULT_STT_COMMAND.to_owned();

        assert_eq!(stt_template(&settings).unwrap(), DEFAULT_STT_COMMAND);
    }

    #[test]
    fn parakeet_engine_uses_transcribe_script() {
        let mut settings = crate::SpeechSettings::default();
        settings.stt_engine = "parakeet".to_owned();
        settings.stt_model = "/tmp/model.bin".to_owned();

        if !command_on_path("python3") {
            return;
        }

        let dir = std::env::temp_dir().join(format!(
            "ollama-manager-parakeet-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for name in PARAKEET_REQUIRED {
            fs::write(dir.join(name), "").unwrap();
        }

        settings.parakeet_dir = dir.to_string_lossy().into_owned();

        assert!(parakeet_ready(&settings));
        assert_eq!(active_stt_engine(&settings), "Parakeet Redux");
        assert_eq!(
            stt_template(&settings).unwrap(),
            "{python} {parakeet_script} --model-dir {parakeet_dir} {wav}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn custom_command_overrides_the_engine() {
        let mut settings = crate::SpeechSettings::default();
        settings.stt_engine = "parakeet".to_owned();
        settings.stt_command = "my-stt {wav}".to_owned();

        assert_eq!(stt_template(&settings).unwrap(), "my-stt {wav}");
    }

    #[test]
    fn parakeet_transcription_runs_the_script() {
        if !command_on_path("python3") {
            return;
        }

        let dir = std::env::temp_dir().join(format!(
            "ollama-manager-parakeet-run-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for name in PARAKEET_REQUIRED {
            fs::write(dir.join(name), "").unwrap();
        }
        fs::write(
            dir.join("transcribe.py"),
            "print('hello from parakeet')\n",
        )
        .unwrap();

        let wav = temp_wav("parakeet-input");
        fs::write(&wav, wav_header(3200)).unwrap();

        let mut settings = crate::SpeechSettings::default();
        settings.stt_engine = "parakeet".to_owned();
        settings.parakeet_dir = dir.to_string_lossy().into_owned();

        assert_eq!(transcribe(&wav, &settings), Ok("hello from parakeet".to_owned()));

        let _ = fs::remove_file(&wav);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn real_parakeet_model_transcribes_a_sample() {
        let settings = crate::SpeechSettings::default();

        if !parakeet_ready(&settings) {
            return;
        }

        let sample = Path::new("/usr/share/sounds/alsa/Front_Center.wav");

        if !sample.exists() {
            return;
        }

        let text = transcribe(sample, &settings).unwrap();
        assert!(!text.trim().is_empty());
    }
}
