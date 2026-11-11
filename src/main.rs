mod render;
mod speech;
mod storage;

use render::render_message;
use speech::Recording;
use storage::{format_bytes, load_settings, save_settings};

use eframe::egui;
use base64::Engine as _;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufRead, BufReader, ErrorKind, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Deserialize)]
struct Model {
    name: String,
    #[serde(default)]
    size: u64,
    #[serde(default, rename = "modified_at")]
    modified_at: String,
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<Model>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Settings {
    host: String,
    port: String,
    #[serde(default)]
    speech: SpeechSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            host: "http://localhost".to_owned(),
            port: "11434".to_owned(),
            speech: SpeechSettings::default(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SpeechSettings {
    /// Template for the recorder. `{wav}` is the output file.
    pub record_command: String,
    /// Template for speech-to-text. `{wav}`, `{model}` and `{lang}` are expanded.
    pub stt_command: String,
    /// Path to a whisper.cpp ggml model.
    pub stt_model: String,
    /// Speech-to-text engine: "auto", "whisper", or "parakeet".
    pub stt_engine: String,
    /// Folder containing Parakeet ONNX files and `transcribe.py`.
    pub parakeet_dir: String,
    pub language: String,
    /// Template for text-to-speech. Text is written to stdin.
    /// `{model}` is expanded.
    pub tts_command: String,
    /// Path to a Piper `.onnx` voice.
    pub tts_model: String,
    /// Send the transcription as soon as it is ready.
    pub auto_send: bool,
    /// Speak assistant replies with the Piper voice model automatically.
    pub auto_speak: bool,
    /// Continuous voice conversation: record, transcribe, send, speak, repeat.
    pub conversation_mode: bool,
    /// Speak replies but do not display the assistant text in the chat.
    #[serde(default)]
    pub hide_reply_text: bool,
}

impl Default for SpeechSettings {
    fn default() -> Self {
        Self {
            // Empty means: auto-detect pw-record, parec, then arecord.
            record_command: String::new(),
            // Empty means: use the selected speech-to-text engine.
            stt_command: String::new(),
            stt_model: "~/.local/share/whisper/ggml-base.en.bin".to_owned(),
            stt_engine: "parakeet".to_owned(),
            parakeet_dir: "~/.local/share/parakeet".to_owned(),
            language: "en".to_owned(),
            tts_command: String::new(),
            tts_model: "~/.local/share/piper/en_US-lessac-medium.onnx"
                .to_owned(),
            auto_send: true,
            // Speak answers out loud by default; can be toggled in Settings → Voice
            // or with the Speak replies checkbox in the chat bar.
            auto_speak: true,
            conversation_mode: false,
            hide_reply_text: false,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct ChatMessage {
    role: String,
    content: String,
    /// A stopped exchange. It stays visible, but is not sent to the model
    /// again, so a stopped question is never answered later.
    #[serde(default, skip_serializing_if = "is_false")]
    stopped: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Messages sent to Ollama for a new request.
///
/// Stopped exchanges are kept in the visible conversation but excluded here,
/// so the model does not answer a question the user already stopped.
fn messages_for_request(messages: &[ChatMessage]) -> Vec<ChatMessage> {
    // Old chats saved before stopped exchanges were tracked can contain an
    // empty assistant message from a stopped generation. Treat such an
    // exchange as stopped as well, otherwise the model sees the old question
    // and tries to answer it together with the new one.
    let mut excluded = vec![false; messages.len()];

    for (index, message) in messages.iter().enumerate() {
        if message.role == "assistant" && message.content.trim().is_empty() {
            excluded[index] = true;

            if index > 0 && messages[index - 1].role == "user" {
                excluded[index - 1] = true;
            }
        }
    }

    messages
        .iter()
        .enumerate()
        .filter(|(index, message)| !message.stopped && !excluded[*index])
        .map(|(_, message)| message.clone())
        .collect()
}

/// Mark the unfinished question/answer pair as stopped.
///
/// The messages stay visible, but `messages_for_request` filters them out so
/// the model is never asked to answer that question again.
fn mark_last_exchange_stopped(messages: &mut [ChatMessage]) {
    let Some(index) = messages
        .iter()
        .rposition(|message| message.role == "assistant")
    else {
        return;
    };

    messages[index].stopped = true;

    if index > 0 && messages[index - 1].role == "user" {
        messages[index - 1].stopped = true;
    }
}

/// Append a diagnostic line when `OLLAMA_MANAGER_DEBUG` is set.
///
/// The log is written to `$TMPDIR/ollama-manager-debug.log` (usually
/// `/tmp/ollama-manager-debug.log`). It is only used for troubleshooting the
/// voice loop; normal runs do not touch the file.
fn debug_log(message: impl AsRef<str>) {
    if std::env::var_os("OLLAMA_MANAGER_DEBUG").is_none() {
        return;
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    let path = std::env::temp_dir().join("ollama-manager-debug.log");

    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "[{stamp}] {}", message.as_ref());
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Conversation {
    title: String,
    folder: String,
    messages: Vec<ChatMessage>,
}

impl Default for Conversation {
    fn default() -> Self {
        Self {
            title: "New conversation".to_owned(),
            folder: "General".to_owned(),
            messages: Vec::new(),
        }
    }
}

#[derive(Clone)]
struct ChatEntry {
    path: PathBuf,
    title: String,
    folder: String,
}
#[derive(Clone)]
struct SavedPrompt {
    name: String,
    content: String,
}

enum Event {
    Models(Vec<Model>),
    Message(String),
    Busy(bool),
    RefreshRequested,
    ChatToken(u64, String),
    ChatFinished(u64, Result<(), String>),
    ChatStopped(u64),
    Transcription(Result<String, String>),
    SpeechFinished(u64, Result<(), String>),
    AttachmentReady(Result<PendingAttachment, String>),
}

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Models,
    Chat,
    Settings,
}

// Small vector-style letter marks are used instead of emoji so they render
// consistently with egui's built-in fonts on Linux and Windows.
fn workspace_nav_button(ui: &mut egui::Ui, active_tab: &mut Tab, tab: Tab, label: &str, mark: &str, tint: egui::Color32) {
    let selected = *active_tab == tab;
    let fill = if selected {
        egui::Color32::from_rgb(226, 202, 164)
    } else {
        egui::Color32::from_rgb(237, 228, 212)
    };
    let stroke = if selected {
        egui::Stroke::new(1.0, egui::Color32::from_rgb(174, 119, 53))
    } else {
        egui::Stroke::new(1.0, egui::Color32::from_rgb(199, 178, 145))
    };
    let response = egui::Frame::new()
        .fill(fill)
        .stroke(stroke)
        .corner_radius(8.0)
        .inner_margin(egui::Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.set_width((ui.available_width() - 20.0).max(0.0));
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(25.0, 25.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 6.0, tint.gamma_multiply(if selected { 0.30 } else { 0.18 }));
                ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, mark,
                    egui::FontId::proportional(14.0), tint);
                ui.add_space(5.0);
                ui.label(
                    egui::RichText::new(label)
                        .strong()
                        .color(egui::Color32::from_rgb(116, 98, 78)),
                );
            });
        }).response;
    let response = response.interact(egui::Sense::click());
    if response.clicked() {
        *active_tab = tab;
    }
    response.on_hover_cursor(egui::CursorIcon::PointingHand);
}

#[derive(Clone)]
struct PendingAttachment {
    name: String,
    size: usize,
    text: Option<String>,
    image_base64: Option<String>,
    note: Option<String>,
}

fn prepare_dropped_file(name: String, bytes: Vec<u8>) -> Result<PendingAttachment, String> {
    const MAX_FILE_SIZE: usize = 20 * 1024 * 1024;
    if bytes.len() > MAX_FILE_SIZE {
        return Err(format!("{} is larger than the 20 MB attachment limit.", name));
    }
    let extension = std::path::Path::new(&name)
        .extension().and_then(|v| v.to_str()).unwrap_or("").to_ascii_lowercase();
    let size = bytes.len();
    let image = matches!(extension.as_str(), "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp");
    let text_like = matches!(extension.as_str(),
        "txt" | "md" | "markdown" | "log" | "csv" | "tsv" | "json" | "jsonl" |
        "xml" | "yaml" | "yml" | "toml" | "ini" | "conf" | "rs" | "py" | "js" |
        "ts" | "tsx" | "jsx" | "html" | "css" | "sh" | "bash" | "sql" | "c" |
        "h" | "cpp" | "java" | "go" | "rb" | "php" | "swift" | "kt" | "diff" | "patch");
    let (text, image_base64, note) = if image {
        (None, Some(base64::engine::general_purpose::STANDARD.encode(&bytes)), None)
    } else if text_like || std::str::from_utf8(&bytes).is_ok() {
        match String::from_utf8(bytes) {
            Ok(value) => (Some(value), None, None),
            Err(_) => (None, None, Some("Could not decode this file as UTF-8; filename only will be attached.".to_owned())),
        }
    } else {
        (None, None, Some("This file type is attached by name only; its contents are not readable by the app yet.".to_owned()))
    };
    Ok(PendingAttachment { name, size, text, image_base64, note })
}

struct OllamaManager {
    settings: Settings,
    model_name: String,
    models: Vec<Model>,
    status: String,
    connected: bool,
    busy: bool,

    receiver: Receiver<Event>,
    sender: Sender<Event>,

    active_tab: Tab,

    chat_model: String,
    chat_entries: Vec<ChatEntry>,
    current_chat: Conversation,
    current_chat_path: Option<PathBuf>,
    selected_folder: String,
    folder_input: String,
    draft: String,
    draft_has_focus: bool,
    pending_attachments: Vec<PendingAttachment>,
    prompts: Vec<SavedPrompt>,
    prompt_name: String,
    prompt_content: String,
    selected_prompt: String,

    chat_cancel: Option<Arc<AtomicBool>>,
    chat_generation: u64,
    /// A chat request is in flight. Kept separate from `busy` so model
    /// operations cannot make the voice loop think the chat is finished.
    chat_active: bool,
    /// A transcription is running in the background.
    transcribing: bool,
    /// A transcript is ready but the previous chat reply is still finishing.
    pending_send: bool,
    recording: Option<Recording>,
    speech_tools: Vec<String>,
    speaking: bool,
    speech_generation: u64,
}

impl Default for OllamaManager {
    fn default() -> Self {
        let mut settings = load_settings();

        // Older settings files stored the built-in Whisper template. Treat it
        // as the automatic default so the engine selector takes effect.
        if settings.speech.stt_command.trim() == speech::DEFAULT_STT_COMMAND {
            settings.speech.stt_command.clear();
        }

        let (sender, receiver) = mpsc::channel();

        let mut app = Self {
            settings,
            model_name: "qwen3:4b".to_owned(),
            models: Vec::new(),
            status: "Not connected yet".to_owned(),
            connected: false,
            busy: false,
            receiver,
            sender,
            active_tab: Tab::Chat,

            chat_model: String::new(),
            chat_entries: Vec::new(),
            current_chat: Conversation::default(),
            current_chat_path: None,
            selected_folder: "General".to_owned(),
            folder_input: String::new(),
            draft: String::new(),
            draft_has_focus: false,
            pending_attachments: Vec::new(),

            prompts: Vec::new(),
            prompt_name: String::new(),
            prompt_content: String::new(),
            selected_prompt: String::new(),

            chat_cancel: None,
            chat_generation: 0,
            chat_active: false,
            transcribing: false,
            pending_send: false,
            recording: None,
            speech_tools: speech::detected_tools()
                .into_iter()
                .map(str::to_owned)
                .collect(),
            speaking: false,
            speech_generation: 0,
        };
        app.reload_chats();
        app.reload_prompts();
        app.refresh_models();

        // If voice conversation mode was left enabled, resume listening.
        app.maybe_start_voice_turn();

        app
    }
}

impl OllamaManager {
    fn base_url(&self) -> String {
        format!(
            "{}:{}",
            self.settings.host.trim_end_matches('/'),
            self.settings.port.trim()
        )
    }

    fn chats_dir() -> PathBuf {
        PathBuf::from("chats")
    }

fn prompts_dir() -> PathBuf {
    PathBuf::from("Prompts")
}

fn reload_prompts(&mut self) {
    self.prompts.clear();

    let directory = Self::prompts_dir();

    if let Err(e) = fs::create_dir_all(&directory) {
        self.status = format!("Could not create Prompts directory: {e}");
        return;
    }

    if let Ok(entries) = fs::read_dir(&directory) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }

            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Untitled")
                .replace('-', " ");

            if let Ok(content) = fs::read_to_string(&path) {
                self.prompts.push(SavedPrompt {
                    name,
                    content,
                });
            }
        }
    }

    self.prompts.sort_by(|a, b| {
        a.name.to_lowercase().cmp(&b.name.to_lowercase())
    });
}

fn save_prompt(&mut self) {
    let name = self.prompt_name.trim().to_owned();

    if name.is_empty() {
        self.status = "Enter a name for the prompt".to_owned();
        return;
    }

    if self.prompt_content.trim().is_empty() {
        self.status = "Enter some prompt text".to_owned();
        return;
    }

    let filename = Self::safe_name(&name);
    let path = Self::prompts_dir().join(format!("{filename}.md"));

    if let Err(e) = fs::create_dir_all(Self::prompts_dir()) {
        self.status = format!("Could not create Prompts directory: {e}");
        return;
    }

    match fs::write(&path, &self.prompt_content) {
        Ok(()) => {
            self.reload_prompts();

            self.selected_prompt = name.clone();
            self.status = format!("Prompt saved: {name}");
        }
        Err(e) => {
            self.status = format!("Could not save prompt: {e}");
        }
    }
}
    fn safe_name(name: &str) -> String {
        let result: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else if c.is_whitespace() {
                    '-'
                } else {
                    '_'
                }
            })
            .collect();

        if result.is_empty() {
            "General".to_owned()
        } else {
            result
        }
    }

    fn reload_chats(&mut self) {
        self.chat_entries.clear();

        let root = Self::chats_dir();

        if let Err(e) = fs::create_dir_all(&root) {
            self.status = format!("Could not create chats directory: {e}");
            return;
        }

        if let Ok(folders) = fs::read_dir(&root) {
            for folder in folders.flatten() {
                if !folder.path().is_dir() {
                    continue;
                }

                let folder_name = folder.file_name().to_string_lossy().to_string();

                if let Ok(files) = fs::read_dir(folder.path()) {
                    for file in files.flatten() {
                        let path = file.path();

                        if path.extension().and_then(|s| s.to_str()) != Some("json") {
                            continue;
                        }

                        if let Ok(contents) = fs::read_to_string(&path) {
                            if let Ok(chat) =
                                serde_json::from_str::<Conversation>(&contents)
                            {
                                self.chat_entries.push(ChatEntry {
                                    path,
                                    title: chat.title,
                                    folder: folder_name.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        self.chat_entries
            .sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
    }

    fn save_current_chat(&mut self) {
        let Some(path) = self.current_chat_path.clone() else {
            return;
        };

        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                self.status = format!("Could not create chat folder: {e}");
                return;
            }
        }

        match serde_json::to_string_pretty(&self.current_chat) {
            Ok(json) => {
                if let Err(e) = fs::write(&path, json) {
                    self.status = format!("Could not save conversation: {e}");
                }
            }
            Err(e) => {
                self.status = format!("Could not serialize conversation: {e}");
            }
        }

        self.reload_chats();
    }

    fn new_chat(&mut self) {
        if self.busy {
            return;
        }

        let folder = Self::safe_name(&self.selected_folder);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let folder_path = Self::chats_dir().join(&folder);

        if let Err(e) = fs::create_dir_all(&folder_path) {
            self.status = format!("Could not create folder: {e}");
            return;
        }

        let path = folder_path.join(format!("conversation-{timestamp}.json"));

        self.current_chat = Conversation {
            title: "New conversation".to_owned(),
            folder: folder.clone(),
            messages: Vec::new(),
        };

        self.current_chat_path = Some(path);
        self.status = "New conversation created".to_owned();

        self.save_current_chat();
    }

    fn open_chat(&mut self, path: PathBuf) {
        if self.busy {
            return;
        }

        match fs::read_to_string(&path) {
            Ok(contents) => {
                match serde_json::from_str::<Conversation>(&contents) {
                    Ok(chat) => {
                        self.selected_folder = chat.folder.clone();
                        self.current_chat = chat;
                        self.current_chat_path = Some(path);
                        self.status = "Conversation opened".to_owned();
                    }
                    Err(e) => {
                        self.status = format!("Invalid conversation file: {e}");
                    }
                }
            }
            Err(e) => {
                self.status = format!("Could not open conversation: {e}");
            }
        }
    }

    fn delete_chat(&mut self, path: PathBuf) {
        if self.busy {
            return;
        }

        if let Err(e) = fs::remove_file(&path) {
            self.status = format!("Could not delete conversation: {e}");
        } else {
            if self.current_chat_path.as_ref() == Some(&path) {
                self.current_chat = Conversation::default();
                self.current_chat_path = None;
            }

            self.reload_chats();
            self.status = "Conversation deleted".to_owned();
        }
    }

    fn create_folder(&mut self) {
        let input = self.folder_input.trim();

        if input.is_empty() {
            return;
        }

        let folder = Self::safe_name(input);
        let path = Self::chats_dir().join(&folder);

        match fs::create_dir_all(path) {
            Ok(_) => {
                self.selected_folder = folder.clone();
                self.folder_input.clear();
                self.status = format!("Folder created: {folder}");
            }
            Err(e) => {
                self.status = format!("Could not create folder: {e}");
            }
        }
    }

    fn folders(&self) -> Vec<String> {
        let mut folders = vec!["General".to_owned()];

        if let Ok(entries) = fs::read_dir(Self::chats_dir()) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();

                    if !folders.contains(&name) {
                        folders.push(name);
                    }
                }
            }
        }

        folders.sort();
        folders
    }

    fn refresh_models(&mut self) {
        let url = format!("{}/api/tags", self.base_url());
        let tx = self.sender.clone();

        self.set_busy(true);

        thread::spawn(move || {
            let result = Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .and_then(|client| client.get(url).send())
                .and_then(|response| response.error_for_status())
                .and_then(|response| response.json::<TagsResponse>());

            match result {
                Ok(data) => {
                    let _ = tx.send(Event::Models(data.models));
                    let _ = tx.send(Event::Message("Connected to Ollama".to_owned()));
                }
                Err(err) => {
                    let _ = tx.send(Event::Message(format!("Connection failed: {err}")));
                }
            }

            let _ = tx.send(Event::Busy(false));
        });
    }

    fn pull_model(&mut self) {
        let name = self.model_name.trim().to_owned();

        if name.is_empty() || self.busy {
            return;
        }

        let url = format!("{}/api/pull", self.base_url());
        let tx = self.sender.clone();

        self.set_busy(true);

        thread::spawn(move || {
            let client = match Client::builder().timeout(None).build() {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Event::Message(format!("HTTP client error: {e}")));
                    let _ = tx.send(Event::Busy(false));
                    return;
                }
            };

            let response = client
                .post(url)
                .json(&serde_json::json!({
                    "name": name,
                    "stream": true
                }))
                .send();

            match response.and_then(|r| r.error_for_status()) {
                Ok(response) => {
                    let reader = BufReader::new(response);

                    for line in reader.lines() {
                        match line {
                            Ok(line) if !line.trim().is_empty() => {
                                if let Ok(value) =
                                    serde_json::from_str::<serde_json::Value>(&line)
                                {
                                    if let Some(status) =
                                        value.get("status").and_then(|v| v.as_str())
                                    {
                                        let _ = tx.send(Event::Message(status.to_owned()));
                                    }

                                    if let Some(error) =
                                        value.get("error").and_then(|v| v.as_str())
                                    {
                                        let _ = tx.send(Event::Message(format!(
                                            "Download failed: {error}"
                                        )));
                                        break;
                                    }
                                }
                            }
                            Ok(_) => {}
                            Err(e) => {
                                let _ = tx.send(Event::Message(format!(
                                    "Download stream error: {e}"
                                )));
                                break;
                            }
                        }
                    }

                    let _ = tx.send(Event::Message(
                        "Download finished. Refreshing model list…".to_owned(),
                    ));
                    let _ = tx.send(Event::RefreshRequested);
                }
                Err(err) => {
                    let _ = tx.send(Event::Message(format!("Download failed: {err}")));
                }
            }

            let _ = tx.send(Event::Busy(false));
        });
    }

    fn delete_model(&mut self, name: String) {
        if self.busy {
            return;
        }

        let url = format!("{}/api/delete", self.base_url());
        let tx = self.sender.clone();

        self.set_busy(true);

        thread::spawn(move || {
            let result = Client::new()
                .delete(url)
                .json(&serde_json::json!({ "name": name }))
                .send()
                .and_then(|r| r.error_for_status());

            match result {
                Ok(_) => {
                    let _ = tx.send(Event::Message(format!("Deleted {name}")));
                    let _ = tx.send(Event::RefreshRequested);
                }
                Err(e) => {
                    let _ = tx.send(Event::Message(format!("Delete failed: {e}")));
                }
            }

            let _ = tx.send(Event::Busy(false));
        });
    }

    fn stop_chat(&mut self) {
        if let Some(flag) = &self.chat_cancel {
            flag.store(true, Ordering::Relaxed);
            self.status = "Stopping generation…".to_owned();
        }
    }

    /// Mark the unfinished question/answer pair as stopped.
    ///
    /// The messages stay visible, but `send_chat` filters them out so the
    /// model is never asked to answer that question again.
    fn mark_last_exchange_stopped(&mut self) {
        mark_last_exchange_stopped(&mut self.current_chat.messages);
    }

    fn start_recording(&mut self) {
        if self.recording.is_some() || self.speaking || self.transcribing {
            return;
        }

        match Recording::start(&self.settings.speech) {
            Ok(recording) => {
                debug_log("recording started");
                self.recording = Some(recording);
                self.status = if self.settings.speech.conversation_mode {
                    "Listening… speak now, then click Stop & send.".to_owned()
                } else {
                    "Recording… click Stop & transcribe when finished.".to_owned()
                };
            }
            Err(e) => {
                debug_log(format!("recording failed: {e}"));
                self.status = e;

                if self.settings.speech.conversation_mode {
                    self.settings.speech.conversation_mode = false;
                    self.status =
                        format!("Voice conversation stopped: {}", self.status);
                }
            }
        }
    }

    fn set_conversation_mode(&mut self, enabled: bool) {
        self.settings.speech.conversation_mode = enabled;

        if enabled {
            self.settings.speech.auto_send = true;
            self.settings.speech.auto_speak = true;
            self.status =
                "Voice conversation on. Speak, then click Stop & send.".to_owned();
            self.maybe_start_voice_turn();
        } else {
            self.status = "Voice conversation off.".to_owned();

            if let Some(recording) = self.recording.take() {
                recording.cancel();
            }
        }

        // Persist both ON and OFF so the user's choice is restored next launch.
        self.save_settings();
    }

    fn maybe_start_voice_turn(&mut self) {
        let can_listen = self.settings.speech.conversation_mode
            && !self.chat_active
            && !self.busy
            && !self.speaking
            && !self.transcribing
            && !self.pending_send
            && self.draft.trim().is_empty()
            && self.recording.is_none();

        debug_log(format!(
            "voice turn check: can_listen={can_listen} mode={} chat_active={} busy={} speaking={} transcribing={} pending_send={} draft_len={} recording={}",
            self.settings.speech.conversation_mode,
            self.chat_active,
            self.busy,
            self.speaking,
            self.transcribing,
            self.pending_send,
            self.draft.len(),
            self.recording.is_some(),
        ));

        if can_listen {
            self.start_recording();
        }
    }

    fn finish_recording(&mut self) {
        let Some(recording) = self.recording.take() else {
            return;
        };

        debug_log("recording stopped, transcribing");
        self.status = "Transcribing…".to_owned();
        self.transcribing = true;

        match recording.stop() {
            Ok(path) => self.spawn_transcription(path),
            Err(e) => {
                self.transcribing = false;
                self.status = e;
            }
        }
    }

    fn spawn_transcription(&self, path: PathBuf) {
        let settings = self.settings.speech.clone();
        let tx = self.sender.clone();

        thread::spawn(move || {
            let result = speech::transcribe(&path, &settings);

            // Keep the last failed recording so it can be tested manually
            // with whisper-cli. It is overwritten on each failure.
            if result.is_err() {
                let keep = std::env::temp_dir()
                    .join("ollama-manager-last-failed.wav");
                let _ = fs::copy(&path, keep);
            }

            let _ = fs::remove_file(&path);
            let _ = tx.send(Event::Transcription(result));
        });
    }

    fn poll_recording(&mut self) {
        let finished = match &mut self.recording {
            Some(recording) => !recording.is_running(),
            None => false,
        };

        if finished {
            self.finish_recording();
        }
    }

    fn speak_text(&mut self, text: String) {
        let text = text.trim().to_owned();

        if text.is_empty() {
            self.status = "There is no text to read aloud yet.".to_owned();
            return;
        }

        // A new request replaces whatever is currently playing.
        if self.speaking {
            speech::stop_speaking();
        }

        let settings = self.settings.speech.clone();
        let tx = self.sender.clone();

        self.speech_generation = self.speech_generation.wrapping_add(1);
        let generation = self.speech_generation;

        self.speaking = true;
        self.status = "Speaking…".to_owned();

        thread::spawn(move || {
            let result = speech::speak(&text, &settings);
            let _ = tx.send(Event::SpeechFinished(generation, result));
        });
    }

    fn stop_speaking(&mut self) {
        speech::stop_speaking();
        // Ignore the playback thread's eventual `SpeechFinished` event so a
        // stale result cannot restart the voice loop behind our back.
        self.speech_generation = self.speech_generation.wrapping_add(1);
        self.speaking = false;
        self.status = "Speech stopped.".to_owned();
    }

    fn stt_engine_label(&self) -> String {
        let command = self.settings.speech.stt_command.trim();

        if !command.is_empty() && command != speech::DEFAULT_STT_COMMAND {
            "custom speech-to-text command".to_owned()
        } else {
            speech::active_stt_engine(&self.settings.speech).to_owned()
        }
    }

    fn speak_last_reply(&mut self) {
        let text = self
            .current_chat
            .messages
            .iter()
            .rev()
            .find(|message| {
                message.role == "assistant"
                    && !message.stopped
                    && !message.content.trim().is_empty()
            })
            .map(|message| message.content.clone())
            .unwrap_or_default();

        self.speak_text(text);
    }

    /// Send a transcript that arrived while the previous reply was still
    /// streaming, once that reply has finished.
    fn flush_pending_send(&mut self) {
        if !self.pending_send {
            return;
        }

        self.pending_send = false;

        debug_log(format!(
            "flush pending send: draft_len={} chat_active={} busy={}",
            self.draft.trim().len(),
            self.chat_active,
            self.busy,
        ));

        if !self.draft.trim().is_empty() && !self.chat_active {
            self.send_chat();
        }
    }

    fn add_dropped_file(&mut self, name: String, bytes: Vec<u8>) {
        let tx = self.sender.clone();
        self.status = format!("Preparing attachment: {name}…");
        thread::spawn(move || {
            let result = prepare_dropped_file(name, bytes);
            let _ = tx.send(Event::AttachmentReady(result));
        });
    }

    fn add_dropped_path(&mut self, name: String, path: PathBuf) {
        let tx = self.sender.clone();
        self.status = format!("Reading attachment: {name}…");
        thread::spawn(move || {
            let result = fs::read(&path)
                .map_err(|error| format!("Could not read {}: {error}", path.display()))
                .and_then(|bytes| prepare_dropped_file(name, bytes));
            let _ = tx.send(Event::AttachmentReady(result));
        });
    }

    fn send_chat(&mut self) {
        let mut text = self.draft.trim().to_owned();
        let attachments = self.pending_attachments.clone();
        if !attachments.is_empty() {
            for attachment in &attachments {
                if let Some(contents) = &attachment.text {
                    if !text.is_empty() { text.push_str("\n\n"); }
                    text.push_str(&format!("[Attached file: {}]\n```\n{}\n```", attachment.name, contents));
                } else if let Some(note) = &attachment.note {
                    if !text.is_empty() { text.push_str("\n\n"); }
                    text.push_str(&format!("[Attached file: {} — {}]", attachment.name, note));
                }
            }
        }
        let attached_images: Vec<String> = attachments.iter()
            .filter_map(|a| a.image_base64.clone()).collect();

        debug_log(format!(
            "send_chat: draft_len={} chat_active={} busy={} speaking={} model={:?}",
            text.len(),
            self.chat_active,
            self.busy,
            self.speaking,
            self.chat_model,
        ));

        if (text.is_empty() && attached_images.is_empty()) || self.chat_active {
            debug_log("send_chat: skipped (empty draft or chat already active)");
            return;
        }

        if self.chat_model.trim().is_empty() {
            self.status = "Select a model before chatting".to_owned();
            return;
        }

        // If an earlier request is somehow still running, stop it and exclude
        // its unfinished exchange before starting a new one. This can happen
        // when another background event clears `busy` early.
        if self.chat_cancel.is_some() {
            if let Some(flag) = &self.chat_cancel {
                flag.store(true, Ordering::Relaxed);
            }

            mark_last_exchange_stopped(&mut self.current_chat.messages);
            self.chat_cancel = None;
        }

        if self.current_chat_path.is_none() {
            self.new_chat();
        }

        if self.current_chat_path.is_none() {
            return;
        }

        if self.current_chat.messages.is_empty() {
            let title: String = if text.trim().is_empty() { "Image attachment".to_owned() } else { text.chars().take(40).collect() };
            self.current_chat.title = title;
        }

        // A new question replaces any reply that is still being spoken.
        if self.speaking {
            self.stop_speaking();
        }

        // Safety net: if an earlier generation ended without being marked
        // stopped, an empty assistant message would otherwise be sent to the
        // model as if it were a real answer.
        let unfinished = self
            .current_chat
            .messages
            .last()
            .map(|last| {
                last.role == "assistant"
                    && last.content.trim().is_empty()
                    && !last.stopped
            })
            .unwrap_or(false);

        if unfinished {
            mark_last_exchange_stopped(&mut self.current_chat.messages);
        }

        // A dangling user message means the previous answer never started
        // (for example after the app was closed mid-generation). Exclude it
        // too, so only the new question is answered.
        if self
            .current_chat
            .messages
            .last()
            .map(|last| last.role == "user" && !last.stopped)
            .unwrap_or(false)
        {
            if let Some(last) = self.current_chat.messages.last_mut() {
                last.stopped = true;
            }
        }

        self.current_chat.messages.push(ChatMessage {
            role: "user".to_owned(),
            content: if text.trim().is_empty() { "[Image attachment]".to_owned() } else { text },
            stopped: false,
        });

        self.draft.clear();
        self.pending_attachments.clear();
        self.draft_has_focus = false;

        self.save_current_chat();

        self.current_chat.messages.push(ChatMessage {
            role: "assistant".to_owned(),
            content: String::new(),
            stopped: false,
        });

        let mut messages = messages_for_request(
            &self.current_chat.messages[..self.current_chat.messages.len() - 1],
        );

        // Add the selected system prompt to the API request only.
        // It is not added to the visible or saved conversation.
        if !self.selected_prompt.is_empty() {
            if let Some(prompt) = self
                .prompts
                .iter()
                .find(|p| p.name == self.selected_prompt)
            {
                messages.insert(
                    0,
                    ChatMessage {
                        role: "system".to_owned(),
                        content: prompt.content.clone(),
                        stopped: false,
                    },
                );
            }
        }

        let model = self.chat_model.clone();
        let url = format!("{}/api/chat", self.base_url());
        let tx = self.sender.clone();

        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.chat_cancel = Some(cancel_flag.clone());

        self.chat_generation = self.chat_generation.wrapping_add(1);
        let generation = self.chat_generation;

        self.chat_active = true;
        self.pending_send = false;
        self.busy = true;
        self.status = format!("Waiting for {model}…");

        debug_log(format!(
            "chat request generation={generation} messages={:?}",
            messages
                .iter()
                .map(|m| format!("{}:{}", m.role, m.content.len()))
                .collect::<Vec<_>>(),
        ));

        thread::spawn(move || {
            let api_messages: Vec<serde_json::Value> = messages.iter().enumerate().map(|(index, message)| {
                let mut value = serde_json::json!({ "role": message.role, "content": message.content });
                if index + 1 == messages.len() && message.role == "user" && !attached_images.is_empty() {
                    value["images"] = serde_json::json!(attached_images);
                }
                value
            }).collect();
            let client = match Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(None)
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Event::ChatFinished(
                        generation,
                        Err(e.to_string()),
                    ));
                    return;
                }
            };

            let response = client
                .post(url)
                .json(&serde_json::json!({
                    "model": model,
                    "messages": api_messages,
                    "stream": true
                }))
                .send()
                .and_then(|r| r.error_for_status());

            match response {
                Ok(response) => {
                    let reader = BufReader::new(response);
                    let mut failed = None;
                    let mut stopped = false;

                    for line in reader.lines() {
                        if cancel_flag.load(Ordering::Relaxed) {
                            stopped = true;
                            break;
                        }

                        match line {
                            Ok(line) if !line.trim().is_empty() => {
                                if let Ok(value) =
                                    serde_json::from_str::<serde_json::Value>(&line)
                                {
                                    if let Some(error) =
                                        value.get("error").and_then(|v| v.as_str())
                                    {
                                        failed = Some(error.to_owned());
                                        break;
                                    }

                                    if let Some(content) = value
                                        .get("message")
                                        .and_then(|m| m.get("content"))
                                        .and_then(|c| c.as_str())
                                    {
                                        if !content.is_empty() {
                                            let _ = tx.send(Event::ChatToken(
                                                generation,
                                                content.to_owned(),
                                            ));
                                        }
                                    }

                                    if value.get("done").and_then(|v| v.as_bool())
                                        == Some(true)
                                    {
                                        break;
                                    }
                                }
                            }
                            Ok(_) => {}
                            Err(e) => {
                                if cancel_flag.load(Ordering::Relaxed) {
                                    stopped = true;
                                    break;
                                }

                                if e.kind() == ErrorKind::TimedOut
                                    || e.kind() == ErrorKind::WouldBlock
                                {
                                    continue;
                                }

                                failed = Some(e.to_string());
                                break;
                            }
                        }
                    }

                    if cancel_flag.load(Ordering::Relaxed) {
                        stopped = true;
                    }

                    if stopped {
                        let _ = tx.send(Event::ChatStopped(generation));
                    } else if let Some(error) = failed {
                        let _ = tx.send(Event::ChatFinished(
                            generation,
                            Err(error),
                        ));
                    } else {
                        let _ = tx.send(Event::ChatFinished(generation, Ok(())));
                    }
                }
                Err(e) => {
                    if cancel_flag.load(Ordering::Relaxed) {
                        let _ = tx.send(Event::ChatStopped(generation));
                    } else {
                        let _ = tx.send(Event::ChatFinished(
                            generation,
                            Err(e.to_string()),
                        ));
                    }
                }
            }
        });
    }

    fn set_busy(&mut self, value: bool) {
        self.busy = value;
    }

    fn poll_events(&mut self) {
        while let Ok(event) = self.receiver.try_recv() {
            match event {
                Event::AttachmentReady(result) => match result {
                    Ok(attachment) => {
                        debug_log(format!("Attachment prepared: {} ({} bytes)", attachment.name, attachment.size));
                        self.status = format!("Attached {}", attachment.name);
                        self.pending_attachments.push(attachment);
                    }
                    Err(error) => {
                        debug_log(format!("Attachment preparation failed: {error}"));
                        self.status = error;
                    }
                },
                Event::Models(models) => {
                    self.models = models;
                    self.connected = true;

                    if self.chat_model.is_empty() {
                        self.chat_model = self.models.first()
                            .map(|m| m.name.clone())
                            .unwrap_or_default();
                    }
                }

                Event::Message(message) => {
                    self.status = message;
                }

                Event::Busy(value) => {
                    self.busy = value;
                }

                Event::RefreshRequested => {
                    self.refresh_models();
                }

                Event::ChatToken(generation, token) => {
                    // Ignore tokens from a request that was stopped or replaced.
                    if generation != self.chat_generation {
                        continue;
                    }

                    if let Some(message) = self.current_chat.messages.last_mut() {
                        if message.role == "assistant" {
                            message.content.push_str(&token);
                        }
                    }
                }

                Event::ChatStopped(generation) => {
                    if generation != self.chat_generation {
                        continue;
                    }

                    debug_log(format!("chat stopped generation={generation}"));
                    self.busy = false;
                    self.chat_active = false;
                    self.chat_cancel = None;
                    self.mark_last_exchange_stopped();
                    self.status =
                        "Generation stopped. That exchange is kept for reading but will not be sent to the model again."
                            .to_owned();
                    self.save_current_chat();

                    self.flush_pending_send();
                    self.maybe_start_voice_turn();
                }

                Event::ChatFinished(generation, result) => {
                    if generation != self.chat_generation {
                        continue;
                    }

                    debug_log(format!(
                        "chat finished generation={generation} ok={}",
                        result.is_ok(),
                    ));
                    self.busy = false;
                    self.chat_active = false;
                    self.chat_cancel = None;

                    match result {
                        Ok(_) => {
                            self.status = "Response complete".to_owned();

                            let should_speak = self.settings.speech.conversation_mode
                                || self.settings.speech.auto_speak;

                            // Only the reply to the current question is
                            // spoken. Never fall back to an older answer.
                            let reply = self
                                .current_chat
                                .messages
                                .last()
                                .filter(|message| {
                                    message.role == "assistant"
                                        && !message.stopped
                                        && !message.content.trim().is_empty()
                                })
                                .map(|message| message.content.clone());

                            if self.pending_send {
                                // The user already asked the next question;
                                // answer that instead of reading the old reply.
                                self.save_current_chat();
                                self.flush_pending_send();
                                self.maybe_start_voice_turn();
                                continue;
                            }

                            if should_speak {
                                if let Some(reply) = reply {
                                    self.speak_text(reply);
                                } else {
                                    self.status =
                                        "The model returned an empty reply."
                                            .to_owned();
                                    self.maybe_start_voice_turn();
                                }
                            } else {
                                self.maybe_start_voice_turn();
                            }
                        }
                        Err(e) => {
                            self.status = format!("Chat error: {e}");

                            if let Some(message) =
                                self.current_chat.messages.last_mut()
                            {
                                if message.role == "assistant"
                                    && message.content.is_empty()
                                {
                                    message.content = format!("Error: {e}");
                                }
                            }

                            self.save_current_chat();
                            self.flush_pending_send();
                            self.maybe_start_voice_turn();
                            continue;
                        }
                    }

                    self.save_current_chat();
                }

                Event::Transcription(result) => {
                    self.transcribing = false;

                    match result {
                        Ok(text) => {
                            debug_log(format!(
                                "transcription ok: {:?} (chat_active={} busy={})",
                                text, self.chat_active, self.busy,
                            ));
                            self.draft = text;
                            self.draft_has_focus = true;
                            self.status =
                                "Transcription ready. Edit it or send it."
                                    .to_owned();

                            let should_send = self.settings.speech.conversation_mode
                                || self.settings.speech.auto_send;

                            if should_send {
                                if self.chat_active {
                                    // Do not lose the question: send it as
                                    // soon as the current reply finishes.
                                    self.pending_send = true;
                                    self.status =
                                        "Waiting for the current reply to finish…"
                                            .to_owned();
                                    debug_log(
                                        "transcription queued: chat still active",
                                    );
                                } else {
                                    self.send_chat();
                                }
                            }
                        }
                        Err(e) => {
                            debug_log(format!("transcription failed: {e}"));

                            if self.settings.speech.conversation_mode {
                                // Silence is common between turns; just listen again
                                // instead of ending the conversation.
                                if e.contains("No speech detected") {
                                    self.status =
                                        "No speech detected — listening again."
                                            .to_owned();
                                    self.maybe_start_voice_turn();
                                } else {
                                    self.settings.speech.conversation_mode = false;
                                    self.status =
                                        format!("Voice conversation stopped: {e}");
                                }
                            } else {
                                self.status = e;
                            }
                        }
                    }
                }

                Event::SpeechFinished(generation, result) => {
                    // Ignore a stale result from playback that was replaced.
                    if generation != self.speech_generation {
                        continue;
                    }

                    self.speaking = false;

                    match result {
                        Ok(()) => {
                            if self.settings.speech.conversation_mode {
                                self.status =
                                    "Your turn — speak, then click Stop & send."
                                        .to_owned();
                                self.maybe_start_voice_turn();
                            } else {
                                self.status = "Finished speaking.".to_owned();
                            }
                        }
                        Err(e) if e == speech::STOPPED => {
                            // The user stopped playback on purpose. Keep the
                            // conversation going if it was hands-free.
                            if self.settings.speech.conversation_mode {
                                self.status =
                                    "Your turn — speak, then click Stop & send."
                                        .to_owned();
                                self.maybe_start_voice_turn();
                            } else {
                                self.status = "Speech stopped.".to_owned();
                            }
                        }
                        Err(e) => {
                            if self.settings.speech.conversation_mode {
                                self.settings.speech.conversation_mode = false;
                                self.status =
                                    format!("Voice conversation stopped: {e}");
                            } else {
                                self.status = format!("Text-to-speech error: {e}");
                            }
                        }
                    }
                }
            }
        }
    }

    fn save_settings(&self) {
        save_settings(&self.settings);
    }
}

impl eframe::App for OllamaManager {
    fn update(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) {
        apply_golden_honey_oak_theme(ctx);
        self.poll_recording();
        self.poll_events();
        // eframe supplies dropped file bytes on the next frame. Read and prepare
        // them here, outside the chat rendering widgets.
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped_files.is_empty() {
            debug_log(format!("OS file drop received: {} file(s)", dropped_files.len()));
        }
        for file in dropped_files {
            let path_name = file.path.as_ref()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str());
            let name = if !file.name.is_empty() { file.name.clone() }
                else { path_name.unwrap_or("Dropped file").to_owned() };
            // Native desktop drops usually provide a path; some backends provide bytes.
            if let Some(path) = file.path {
                self.add_dropped_path(name, path);
            } else if let Some(bytes) = file.bytes {
                self.add_dropped_file(name, bytes.to_vec());
            } else {
                debug_log(format!("Dropped file had neither path nor bytes: {name}"));
                self.status = format!("The system did not provide readable data for {name}.");
            }
        }
        ctx.request_repaint_after(Duration::from_millis(100));

        // Top application header.
        egui::TopBottomPanel::top("header")
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(237, 228, 212))
                    .inner_margin(egui::Margin::symmetric(16, 10)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(
                        egui::RichText::new("Ollama Manager")
                            .strong()
                            .size(19.0)
                            .color(egui::Color32::BLACK),
                    );

                    let (status_color, status_text) = if self.connected {
                        (egui::Color32::from_rgb(28, 91, 43), "✓  Connected")
                    } else {
                        (egui::Color32::from_rgb(145, 48, 38), "×  Disconnected")
                    };
                    ui.label(
                        egui::RichText::new(status_text)
                            .strong()
                            .color(status_color),
                    );

                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui.add_enabled(
                                !self.busy,
                                egui::Button::new("↻ Refresh"),
                            ).clicked() {
                                self.connected = false;
                                self.refresh_models();
                            }
                        },
                    );
                });
            });

        // Sidebar.
        egui::SidePanel::left("navigation")
            .resizable(false)
            .show_separator_line(false)
            .default_width(245.0)
            .width_range(245.0..=245.0)
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(237, 228, 212))
                    .stroke(egui::Stroke::NONE)
                    .inner_margin(12.0),
            )
            .show(ctx, |ui| {
                ui.add_space(5.0);

                ui.heading("Chats");
                ui.add_space(8.0);

                ui.label(
                    egui::RichText::new("FOLDER")
                        .small()
                        .strong()
                        .color(egui::Color32::from_rgb(116, 98, 78)),
                );
                egui::ComboBox::from_id_salt("chat_folder")
                    .selected_text(&self.selected_folder)
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for folder in self.folders() {
                            ui.selectable_value(
                                &mut self.selected_folder,
                                folder.clone(),
                                folder,
                            );
                        }
                    });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.folder_input)
                            .hint_text("Create a folder…")
                            .desired_width(ui.available_width() - 34.0),
                    );
                    if ui.small_button("+").on_hover_text("Create folder").clicked() {
                        self.create_folder();
                    }
                });

                ui.add_space(14.0);
                ui.label(
                    egui::RichText::new("CONVERSATIONS")
                        .small()
                        .strong()
                        .color(egui::Color32::from_rgb(116, 98, 78)),
                );

                let entries: Vec<_> = self.chat_entries.iter()
                    .filter(|entry| entry.folder == self.selected_folder)
                    .cloned()
                    .collect();
                let selected_title = entries.iter()
                    .find(|entry| self.current_chat_path.as_ref() == Some(&entry.path))
                    .map(|entry| if entry.title.trim().is_empty() {
                        "Untitled conversation".to_owned()
                    } else {
                        entry.title.clone()
                    })
                    .unwrap_or_else(|| "Choose a conversation…".to_owned());

                egui::ComboBox::from_id_salt("chat_conversations")
                    .selected_text(selected_title)
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        if entries.is_empty() {
                            ui.weak("No conversations in this folder");
                        }
                        for entry in &entries {
                            let selected = self.current_chat_path.as_ref() == Some(&entry.path);
                            let title = if entry.title.trim().is_empty() {
                                "Untitled conversation"
                            } else {
                                &entry.title
                            };
                            ui.horizontal(|ui| {
                                if ui.add_enabled(
                                    !self.busy,
                                    egui::Button::new(title).selected(selected).wrap(),
                                ).clicked() {
                                    self.open_chat(entry.path.clone());
                                    self.active_tab = Tab::Chat;
                                }
                                if ui.add_enabled(!self.busy, egui::Button::new("×"))
                                    .on_hover_text("Delete conversation")
                                    .clicked()
                                {
                                    self.delete_chat(entry.path.clone());
                                }
                            });
                        }
                    });

                ui.add_space(10.0);
                if ui.add_enabled(
                    !self.busy,
                    egui::Button::new("+  New conversation")
                        .min_size(egui::vec2(ui.available_width(), 38.0)),
                ).clicked() {
                    self.new_chat();
                    self.active_tab = Tab::Chat;
                }

                ui.add_space(10.0);
                ui.separator();

                ui.add_space(8.0);
                ui.separator();

                ui.label(
                    egui::RichText::new("WORKSPACE")
                        .small()
                        .color(egui::Color32::from_rgb(116, 98, 78)),
                );

                ui.add_space(5.0);

                workspace_nav_button(
                    ui, &mut self.active_tab, Tab::Chat, "Chat", "C",
                    egui::Color32::from_rgb(104, 177, 245),
                );
                ui.add_space(5.0);
                workspace_nav_button(
                    ui, &mut self.active_tab, Tab::Models, "Models", "M",
                    egui::Color32::from_rgb(115, 205, 170),
                );
                ui.add_space(5.0);
                workspace_nav_button(
                    ui, &mut self.active_tab, Tab::Settings, "Settings", "S",
                    egui::Color32::from_rgb(190, 165, 245),
                );

                ui.add_space(12.0);
                ui.separator();

                ui.label(
                    egui::RichText::new("OLLAMA SERVER")
                        .small()
                        .color(egui::Color32::GRAY),
                );

                ui.add_space(4.0);
                ui.small(self.base_url());

                ui.add_space(10.0);
                ui.separator();
                ui.label(
                    egui::RichText::new("STATUS")
                        .small()
                        .color(egui::Color32::GRAY),
                );
                ui.add_space(4.0);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(&self.status)
                            .small()
                            .color(egui::Color32::from_rgb(116, 98, 78)),
                    )
                    .wrap(),
                );

                ui.add_space(12.0);

            });

        // Main content.
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(244, 239, 229))
                    .stroke(egui::Stroke::NONE)
                    .inner_margin(egui::Margin::symmetric(20, 14)),
            )
            .show(ctx, |ui| {
                if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
                    egui::Frame::new()
                        .fill(egui::Color32::from_rgb(229, 215, 192))
                        .stroke(egui::Stroke::new(1.5, egui::Color32::from_rgb(145, 91, 39)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(12, 8))
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Drop files to attach them")
                                .strong().color(egui::Color32::from_rgb(60, 47, 35)));
                        });
                }
                match self.active_tab {
                    Tab::Chat => {
                        ui.horizontal(|ui| {
                            ui.heading(
                                egui::RichText::new(&self.current_chat.title)
                                    .size(21.0)
                                    .strong()
                                    .color(egui::Color32::from_rgb(60, 47, 35)),
                            );

                            if self.chat_active {
                                ui.colored_label(
                                    egui::Color32::LIGHT_GREEN,
                                    "● Generating",
                                );
                            }

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.add_enabled(
                                        !self.chat_active && !self.busy,
                                        egui::Button::new("+  New chat"),
                                    ).clicked() {
                                        self.new_chat();
                                    }
                                },
                            );
                        });

                        ui.add_space(4.0);

ui.horizontal_wrapped(|ui| {
    ui.label("Model");

    egui::ComboBox::from_id_salt("main_chat_model")
        .selected_text(if self.chat_model.is_empty() {
            "Select a model"
        } else {
            &self.chat_model
        })
        .show_ui(ui, |ui| {
            for model in &self.models {
                ui.selectable_value(
                    &mut self.chat_model,
                    model.name.clone(),
                    &model.name,
                );
            }
        });

    ui.add_space(12.0);

    ui.label("Prompt");

    egui::ComboBox::from_id_salt("chat_system_prompt")
        .selected_text(if self.selected_prompt.is_empty() {
            "No prompt"
        } else {
            &self.selected_prompt
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(
                &mut self.selected_prompt,
                String::new(),
                "No prompt",
            );

            for prompt in &self.prompts {
                ui.selectable_value(
                    &mut self.selected_prompt,
                    prompt.name.clone(),
                    &prompt.name,
                );
            }
        });
});
                        ui.add_space(8.0);

                        let available_height = ui.available_height();
                        let chat_height = (available_height - 135.0).max(100.0);

                        egui::ScrollArea::vertical()
                            .id_salt("chat_messages")
                            .stick_to_bottom(true)
                            .max_height(chat_height)
                            .show(ui, |ui| {
                                let messages = self.current_chat.messages.clone();

                                for message in &messages {
                                    let is_user = message.role == "user";

                                    ui.horizontal(|ui| {
                                        if is_user {
                                            ui.with_layout(
                                                egui::Layout::right_to_left(
                                                    egui::Align::Min,
                                                ),
                                                |ui| {
                                                    egui::Frame::new()
                                                        .fill(egui::Color32::from_rgb(237, 228, 212))
                                                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(224, 211, 190)))
                                                        .corner_radius(14.0)
                                                        .inner_margin(egui::Margin::symmetric(16, 12))
                                                        .show(ui, |ui| {
                                                            ui.set_max_width(
                                                                ui.available_width().min(760.0),
                                                            );

                                                            ui.colored_label(
                                                                egui::Color32::from_rgb(145, 91, 39),
                                                                "You",
                                                            );

                                                            ui.add_space(4.0);
                                                            render_message(
                                                                ui,
                                                                ctx,
                                                                &message.content,
                                                            );

                                                            if message.stopped {
                                                                ui.add_space(4.0);
                                                                ui.label(
                                                                    egui::RichText::new(
                                                                        "⏹ stopped — not used in later replies",
                                                                    )
                                                                    .small()
                                                                    .color(
                                                                        egui::Color32::GRAY,
                                                                    ),
                                                                );
                                                            }
                                                        });
                                                },
                                            );
                                        } else {
                                            egui::Frame::new()
                                                .fill(egui::Color32::from_rgb(250, 247, 240))
                                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(232, 222, 205)))
                                                .corner_radius(14.0)
                                                .inner_margin(egui::Margin::symmetric(16, 12))
                                                .show(ui, |ui| {
                                                    ui.set_max_width(
                                                        ui.available_width().min(900.0),
                                                    );

                                                    ui.colored_label(
                                                        egui::Color32::from_rgb(74, 105, 65),
                                                        "Assistant",
                                                    );

                                                    ui.add_space(4.0);

                                                    if self.settings.speech.hide_reply_text
                                                        && (self
                                                            .settings
                                                            .speech
                                                            .auto_speak
                                                            || self
                                                                .settings
                                                                .speech
                                                                .conversation_mode)
                                                        && !message
                                                            .content
                                                            .starts_with("Error:")
                                                    {
                                                        ui.horizontal(|ui| {
                                                            if self.chat_active {
                                                                ui.spinner();
                                                                ui.label(
                                                                    egui::RichText::new(
                                                                        "Generating reply…",
                                                                    )
                                                                    .italics()
                                                                    .color(
                                                                        egui::Color32::GRAY,
                                                                    ),
                                                                );
                                                            } else {
                                                                ui.label("Speaking");
                                                                ui.label(
                                                                    egui::RichText::new(
                                                                        "Reply spoken aloud (text hidden)",
                                                                    )
                                                                    .italics()
                                                                    .color(
                                                                        egui::Color32::GRAY,
                                                                    ),
                                                                );
                                                            }
                                                        });
                                                    } else if message.content.is_empty()
                                                        && self.chat_active
                                                    {
                                                        ui.horizontal(|ui| {
                                                            ui.spinner();
                                                            ui.label("Thinking…");
                                                        });
                                                    } else {
                                                        render_message(
                                                            ui,
                                                            ctx,
                                                            &message.content,
                                                        );
                                                    }

                                                    if message.stopped {
                                                        ui.add_space(4.0);
                                                        ui.label(
                                                            egui::RichText::new(
                                                                "⏹ stopped — not used in later replies",
                                                            )
                                                            .small()
                                                            .color(
                                                                egui::Color32::GRAY,
                                                            ),
                                                        );
                                                    }
                                                });
                                        }
                                    });

                                    ui.add_space(12.0);
                                }

                                if self.current_chat.messages.is_empty() {
                                    ui.add_space(65.0);

                                    ui.vertical_centered(|ui| {
                                        ui.heading(
                                            egui::RichText::new(
                                                "What can I help you with?",
                                            )
                                            .size(25.0),
                                        );

                                        ui.add_space(8.0);

                                        ui.label(
                                            egui::RichText::new(
                                                "Your local AI assistant, powered by Ollama.",
                                            )
                                            .color(egui::Color32::GRAY),
                                        );

                                        ui.add_space(20.0);

                                        ui.horizontal_wrapped(|ui| {
                                            for suggestion in [
                                                "Explain a piece of code",
                                                "Help me debug an issue",
                                                "Write a script",
                                            ] {
                                                if ui.button(suggestion).clicked() {
                                                    self.draft =
                                                        format!("{suggestion}.");
                                                }
                                            }
                                        });
                                    });
                                }
                            });

                        ui.add_space(8.0);
                        ui.separator();

                        // Enter sends; Shift+Enter inserts a newline.
                        let send_on_enter = self.draft_has_focus
                            && ui.input(|i| {
                                i.key_pressed(egui::Key::Enter)
                                    && !i.modifiers.shift
                            });

                        if send_on_enter {
                            ui.input_mut(|i| {
                                i.consume_key(
                                    egui::Modifiers::NONE,
                                    egui::Key::Enter,
                                );
                            });
                        }

                        if !self.pending_attachments.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                let mut remove = None;
                                for (index, attachment) in self.pending_attachments.iter().enumerate() {
                                    ui.group(|ui| {
                                        ui.horizontal(|ui| {
                                            let kind = if attachment.image_base64.is_some() { "🖼" } else if attachment.text.is_some() { "📄" } else { "📎" };
                                            ui.label(format!("{} {} ({})", kind, attachment.name, format_bytes(attachment.size as u64)));
                                            if ui.small_button("×").on_hover_text("Remove attachment").clicked() { remove = Some(index); }
                                        });
                                        if let Some(note) = &attachment.note { ui.small(note); }
                                    });
                                }
                                if let Some(index) = remove { self.pending_attachments.remove(index); }
                            });
                        }

                        if self.pending_attachments.is_empty() {
                            ui.label(
                                egui::RichText::new("Tip: drag and drop files here to attach them")
                                    .small()
                                    .color(egui::Color32::from_rgb(116, 98, 78)),
                            );
                        }

                        let draft_response = ui.add_enabled(
                            !self.chat_active,
                            egui::TextEdit::multiline(&mut self.draft)
                                .text_color(egui::Color32::from_rgb(60, 47, 35))
                                .background_color(egui::Color32::from_rgb(244, 239, 229))
                                .hint_text(
                                    "Message your local model… (Enter to send, Shift+Enter for newline)",
                                )
                                .desired_rows(3)
                                .desired_width(f32::INFINITY),
                        );

                        self.draft_has_focus = draft_response.has_focus();

                        ui.add_space(5.0);

                        ui.horizontal(|ui| {
                            let can_send = !self.chat_active
                                && (!self.draft.trim().is_empty() || !self.pending_attachments.is_empty())
                                && !self.chat_model.is_empty();

                            if ui.add_enabled(
                                can_send,
                                egui::Button::new(">  Send message")
                                    .fill(egui::Color32::from_rgb(174, 119, 53)),
                            ).clicked() || (send_on_enter && can_send) {
                                self.send_chat();
                            }

                            let voice = self.settings.speech.conversation_mode;

                            if ui.add(
                                egui::Button::new(if voice {
                                    "🎙  Voice conversation: ON"
                                } else {
                                    "🎙  Voice conversation"
                                })
                                .fill(if voice {
                                    egui::Color32::from_rgb(45, 105, 75)
                                } else {
                                    egui::Color32::from_rgb(229, 215, 192)
                                }),
                            ).on_hover_text(
                                "Automatically listen, transcribe, send, speak the reply, and listen again.",
                            ).clicked() {
                                self.set_conversation_mode(!voice);
                            }

                            let is_recording = self.recording.is_some();
                            let record_label = if is_recording {
                                if voice {
                                    "■  Stop & send"
                                } else {
                                    "■  Stop & transcribe"
                                }
                            } else if voice {
                                "🎙  Talk"
                            } else {
                                "Speak"
                            };

                            if ui.add_enabled(
                                !self.chat_active
                                    && !self.speaking
                                    && !self.transcribing,
                                egui::Button::new(
                                    egui::RichText::new(record_label)
                                        .strong()
                                        .color(egui::Color32::from_rgb(60, 47, 35)),
                                ).fill(if is_recording {
                                    egui::Color32::from_rgb(150, 55, 60)
                                } else {
                                    egui::Color32::from_rgb(218, 178, 112)
                                }),
                            ).clicked() {
                                if is_recording {
                                    self.finish_recording();
                                } else {
                                    self.start_recording();
                                }
                            }

                            if let Some(recording) = &self.recording {
                                ui.colored_label(
                                    egui::Color32::LIGHT_RED,
                                    format!(
                                        "● {} {}s",
                                        if voice { "Listening" } else { "Recording" },
                                        recording.elapsed().as_secs()
                                    ),
                                );
                            }

                            if ui.add_enabled(
                                !self.current_chat.messages.is_empty()
                                    && !self.speaking,
                                egui::Button::new(
                                    egui::RichText::new(">  Speak reply")
                                        .strong()
                                        .color(egui::Color32::from_rgb(60, 47, 35)),
                                ).fill(egui::Color32::from_rgb(229, 215, 192)),
                            ).clicked() {
                                self.speak_last_reply();
                            }

                            if self.speaking && ui.add(
                                egui::Button::new("||  Stop speaking")
                                    .fill(egui::Color32::from_rgb(150, 55, 60)),
                            ).on_hover_text(
                                "Stop the voice playback immediately.",
                            ).clicked() {
                                self.stop_speaking();

                                // Stopping playback on purpose should keep a
                                // hands-free conversation going.
                                if self.settings.speech.conversation_mode {
                                    self.maybe_start_voice_turn();
                                }
                            }

                            if ui.checkbox(
                                &mut self.settings.speech.auto_speak,
                                "Speak replies",
                            ).changed() {
                                self.save_settings();
                            }

                            if ui.checkbox(
                                &mut self.settings.speech.hide_reply_text,
                                "Hide text",
                            ).on_hover_text(
                                "Speak the assistant's replies without showing the text in the chat.",
                            ).changed() {
                                if self.settings.speech.hide_reply_text {
                                    self.settings.speech.auto_speak = true;
                                }

                                self.save_settings();
                            }

                            if self.chat_active {
                                if ui.add(
                                    egui::Button::new("[]  Stop")
                                        .fill(egui::Color32::from_rgb(150, 55, 60)),
                                ).clicked() {
                                    self.stop_chat();
                                }

                                ui.label(
                                    egui::RichText::new("Generating response…")
                                        .color(egui::Color32::LIGHT_GREEN),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new(
                                        "Enter to send · Shift+Enter for newline",
                                    )
                                    .small()
                                    .color(egui::Color32::GRAY),
                                );
                            }
                        });
                    }

                    Tab::Models => {
                        ui.heading("Model library");

                        ui.label(
                            egui::RichText::new(
                                "Manage the models installed in your Ollama instance.",
                            )
                            .color(egui::Color32::GRAY),
                        );

                        ui.add_space(16.0);

                        egui::Frame::group(ui.style())
                            .inner_margin(14.0)
                            .show(ui, |ui| {
                                ui.heading("Download a model");
                                ui.add_space(8.0);

                                ui.horizontal(|ui| {
                                    ui.add_enabled(
                                        !self.busy,
                                        egui::TextEdit::singleline(
                                            &mut self.model_name,
                                        )
                                        .hint_text("e.g. qwen3:4b")
                                        .desired_width(280.0),
                                    );

                                    if ui.add_enabled(
                                        !self.busy,
                                        egui::Button::new("v  Download"),
                                    ).clicked() {
                                        self.pull_model();
                                    }
                                });
                            });

                        ui.add_space(18.0);

                        ui.horizontal(|ui| {
                            ui.heading("Installed models");
                            ui.label(format!("({})", self.models.len()));
                        });

                        ui.add_space(6.0);

                        if self.models.is_empty() {
                            ui.label(if self.connected {
                                "No models found. Download one above."
                            } else {
                                "Connect to Ollama in Settings."
                            });
                        } else {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                for model in self.models.clone() {
                                    egui::Frame::group(ui.style())
                                        .inner_margin(12.0)
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.vertical(|ui| {
                                                    ui.label(
                                                        egui::RichText::new(&model.name)
                                                            .strong()
                                                            .size(15.0)
                                                            .color(egui::Color32::from_rgb(60, 47, 35)),
                                                    );

                                                    ui.label(
                                                        egui::RichText::new(format!(
                                                            "{} • {}",
                                                            format_bytes(model.size),
                                                            model.modified_at
                                                        ))
                                                        .small()
                                                        .color(egui::Color32::from_rgb(116, 98, 78)),
                                                    );
                                                });

                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        if ui.add_enabled(
                                                            !self.busy,
                                                            egui::Button::new(
                                                                "Delete",
                                                            ),
                                                        ).clicked() {
                                                            self.delete_model(
                                                                model.name.clone(),
                                                            );
                                                        }
                                                    },
                                                );
                                            });
                                        });

                                    ui.add_space(6.0);
                                }
                            });
                        }
                    }

                    Tab::Settings => {
                        egui::ScrollArea::vertical()
                            .id_salt("settings_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                        ui.heading("Settings");

                        ui.label(
                            egui::RichText::new(
                                "Configure your local Ollama server connection.",
                            )
                            .color(egui::Color32::GRAY),
                        );

                        ui.add_space(20.0);

                        egui::Frame::group(ui.style())
                            .inner_margin(14.0)
                            .show(ui, |ui| {
                                ui.label("Host");

                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut self.settings.host,
                                    )
                                    .desired_width(350.0),
                                );

                                ui.add_space(10.0);

                                ui.label("Port");

                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut self.settings.port,
                                    )
                                    .desired_width(150.0),
                                );

                                ui.add_space(14.0);

                                ui.horizontal(|ui| {
                                    if ui.button("+  Save settings").clicked() {
                                        self.save_settings();
                                        self.connected = false;
                                        self.status = "Settings saved".to_owned();
                                    }

                                    if ui.add_enabled(
                                        !self.busy,
                                        egui::Button::new(">  Test connection"),
                                    ).clicked() {
                                        self.save_settings();
                                        self.connected = false;
                                        self.refresh_models();
                                    }
                                });
                            });

                        ui.add_space(14.0);

                        ui.label("Default: http://localhost:11434");

                        ui.add_space(10.0);


                        ui.colored_label(
                            egui::Color32::from_rgb(105, 28, 48),
                            egui::RichText::new(
                                "Security: only expose Ollama to trusted networks.",
                            ).strong(),
                        );

                        ui.add_space(25.0);
                        ui.separator();
                        ui.add_space(12.0);

                        ui.heading("Voice models");
                        ui.label(
                            egui::RichText::new(format!(
                                "Pipeline: {} (speech → text) → Ollama (text → reply) → Piper (reply → speech).",
                                self.stt_engine_label(),
                            ))
                            .color(egui::Color32::GRAY),
                        );
                        ui.label(
                            egui::RichText::new(
                                "The app uses local speech models and Piper voice files. No screen reader is used.",
                            )
                            .small()
                            .color(egui::Color32::GRAY),
                        );

                        let tools_text = if self.speech_tools.is_empty() {
                            "Detected tools: none".to_owned()
                        } else {
                            format!(
                                "Detected tools: {}",
                                self.speech_tools.join(", ")
                            )
                        };

                        ui.horizontal(|ui| {
                            ui.small(tools_text);

                            if ui.small_button("↻ Refresh").clicked() {
                                self.speech_tools = speech::detected_tools()
                                    .into_iter()
                                    .map(str::to_owned)
                                    .collect();
                            }
                        });

                        ui.add_space(10.0);

                        egui::Frame::group(ui.style())
                            .inner_margin(14.0)
                            .show(ui, |ui| {
                                ui.checkbox(
                                    &mut self.settings.speech.auto_send,
                                    "Send transcription automatically",
                                );
                                ui.checkbox(
                                    &mut self.settings.speech.auto_speak,
                                    "Speak assistant replies with Piper",
                                );
                                if ui
                                    .checkbox(
                                        &mut self.settings.speech.hide_reply_text,
                                        "Voice only — speak replies without showing the text in the chat",
                                    )
                                    .changed()
                                {
                                    if self.settings.speech.hide_reply_text {
                                        self.settings.speech.auto_speak = true;
                                    }

                                    self.save_settings();
                                }

                                if ui
                                    .checkbox(
                                        &mut self.settings.speech.conversation_mode,
                                        "Voice conversation mode (listen → transcribe → send → speak → listen)",
                                    )
                                    .changed()
                                {
                                    let enabled =
                                        self.settings.speech.conversation_mode;
                                    self.set_conversation_mode(enabled);
                                }

                                ui.add_space(10.0);
                                ui.label("Speech-to-text engine");

                                let engine_before =
                                    self.settings.speech.stt_engine.clone();

                                egui::ComboBox::from_id_salt("stt_engine")
                                    .selected_text(match engine_before.as_str() {
                                        "whisper" => "Whisper",
                                        "parakeet" => "Parakeet Redux",
                                        _ => "Auto (Parakeet if installed)",
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut self.settings.speech.stt_engine,
                                            "auto".to_owned(),
                                            "Auto (Parakeet if installed)",
                                        );
                                        ui.selectable_value(
                                            &mut self.settings.speech.stt_engine,
                                            "parakeet".to_owned(),
                                            "Parakeet Redux (ONNX Runtime)",
                                        );
                                        ui.selectable_value(
                                            &mut self.settings.speech.stt_engine,
                                            "whisper".to_owned(),
                                            "Whisper (whisper.cpp)",
                                        );
                                    });

                                if self.settings.speech.stt_engine != engine_before {
                                    // The old default template is equivalent to
                                    // "auto"; store it as empty so the engine
                                    // selector is not overridden.
                                    let command =
                                        self.settings.speech.stt_command.trim();
                                    if command == speech::DEFAULT_STT_COMMAND {
                                        self.settings.speech.stt_command.clear();
                                    }

                                    self.save_settings();
                                }

                                ui.small(format!(
                                    "Active engine: {}",
                                    self.stt_engine_label()
                                ));

                                ui.add_space(10.0);
                                ui.label("Parakeet folder");
                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut self.settings.speech.parakeet_dir,
                                    )
                                    .hint_text("~/.local/share/parakeet")
                                    .desired_width(f32::INFINITY),
                                );

                                if speech::parakeet_ready(&self.settings.speech) {
                                    ui.small("Parakeet files found (transcribe.py + encoder-model.onnx).");
                                } else {
                                    ui.small(
                                        egui::RichText::new(format!(
                                            "Parakeet not ready: expected transcribe.py and the ONNX model files in {}.",
                                            speech::parakeet_dir(
                                                &self.settings.speech
                                            )
                                            .display()
                                        ))
                                        .color(egui::Color32::GRAY),
                                    );
                                }

                                ui.add_space(10.0);
                                ui.label("Whisper language (Whisper only)");
                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut self.settings.speech.language,
                                    )
                                    .hint_text("en")
                                    .desired_width(100.0),
                                );

                                ui.add_space(10.0);
                                ui.label("Whisper model (.bin)");
                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut self.settings.speech.stt_model,
                                    )
                                    .hint_text("~/.local/share/whisper/ggml-base.en.bin")
                                    .desired_width(f32::INFINITY),
                                );

                                ui.add_space(10.0);
                                ui.label("Speech-to-text command (optional)");
                                ui.add(
                                    egui::TextEdit::multiline(
                                        &mut self.settings.speech.stt_command,
                                    )
                                    .hint_text(
                                        "Leave empty to use the selected engine automatically",
                                    )
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(2),
                                );
                                ui.small(
                                    "A custom command overrides the engine above. Placeholders: {wav}, {model}, {lang}, {parakeet_dir}, {parakeet_script}, {python}. Output must go to stdout.",
                                );

                                ui.add_space(12.0);
                                ui.label("Piper voice (.onnx)");
                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut self.settings.speech.tts_model,
                                    )
                                    .hint_text("~/.local/share/piper/en_US-lessac-medium.onnx")
                                    .desired_width(f32::INFINITY),
                                );

                                ui.add_space(10.0);
                                ui.label("Text-to-speech command (optional)");
                                ui.add(
                                    egui::TextEdit::multiline(
                                        &mut self.settings.speech.tts_command,
                                    )
                                    .hint_text(
                                        "piper --model {model} --output_file /tmp/piper.wav && aplay -q /tmp/piper.wav",
                                    )
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(2),
                                );
                                ui.small(
                                    "Text is written to stdin. Leave empty to use the Piper voice model with paplay/pw-play/aplay.",
                                );

                                ui.add_space(10.0);
                                ui.label("Recording command");
                                ui.add(
                                    egui::TextEdit::singleline(
                                        &mut self.settings.speech.record_command,
                                    )
                                    .hint_text(
                                        "Leave empty to auto-detect pw-record, parec or arecord",
                                    )
                                    .desired_width(f32::INFINITY),
                                );

                                ui.add_space(14.0);

                                ui.horizontal(|ui| {
                                    if ui.button("+  Save voice settings").clicked() {
                                        self.save_settings();
                                        self.status = "Voice settings saved".to_owned();
                                    }

                                    if ui.button(">  Test voice").clicked() {
                                        self.speak_text(
                                            "Hello. Speech output is working.".to_owned(),
                                        );
                                    }
                                });

                                ui.add_space(8.0);
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(&self.status)
                                            .small()
                                            .color(egui::Color32::from_rgb(116, 98, 78)),
                                    )
                                    .wrap(),
                                );
                            });

                        ui.add_space(25.0);
                        ui.separator();
                        ui.add_space(12.0);

                        ui.heading("System prompts");

                        ui.label(
                            egui::RichText::new(
                                "Create reusable instructions for your local AI models.",
                            )
                            .color(egui::Color32::GRAY),
                        );

                        ui.add_space(12.0);

                        egui::Frame::group(ui.style())
                            .inner_margin(14.0)
                            .show(ui, |ui| {
                                ui.label("Prompt name");

                                ui.add(
                                    egui::TextEdit::singleline(&mut self.prompt_name)
                                        .hint_text("e.g. Rust Developer")
                                        .desired_width(350.0),
                                );

                                ui.add_space(12.0);

                                ui.label("Prompt instructions");

                                ui.add(
                                    egui::TextEdit::multiline(&mut self.prompt_content)
                                        .hint_text(
                                            "Write the system instructions for your model here..."
                                        )
                                        .desired_width(f32::INFINITY)
                                        .desired_rows(12),
                                );

                                ui.add_space(12.0);

                                ui.horizontal(|ui| {
                                    if ui.button("+  Save prompt").clicked() {
                                        self.save_prompt();
                                    }

                                    if ui.button("x  Clear").clicked() {
                                        self.prompt_name.clear();
                                        self.prompt_content.clear();
                                    }
                                });
                            });

                        ui.add_space(18.0);

                        ui.heading("Saved prompts");

                        if self.prompts.is_empty() {
                            ui.label("No saved prompts yet.");
                        } else {
                            for prompt in self.prompts.clone() {
                                egui::Frame::group(ui.style())
                                    .inner_margin(10.0)
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.strong(&prompt.name);

                                            ui.with_layout(
                                                egui::Layout::right_to_left(
                                                    egui::Align::Center,
                                                ),
                                                |ui| {
                                                    if ui.small_button("Edit").clicked() {
                                                        self.prompt_name = prompt.name.clone();
                                                        self.prompt_content = prompt.content.clone();
                                                    }
                                                },
                                            );
                                        });

                                        ui.add_space(4.0);

                                        ui.label(
                                            egui::RichText::new(
                                                prompt.content.lines().next().unwrap_or("")
                                            )
                                            .small()
                                            .color(egui::Color32::GRAY),
                                        );
                                    });

                                ui.add_space(5.0);
                            }
                        }
                            });
                    }
                }
            });
        }
    }

                        fn main() -> eframe::Result<()> {
    // winit 0.30 (used by eframe 0.31) does not deliver file-drop events on
    // native Wayland. Prefer XWayland when available so desktop file drops
    // reach egui. If there is no X11 display, leave the session untouched.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WAYLAND_DISPLAY").is_some()
        && std::env::var_os("DISPLAY").is_some()
    {
        std::env::remove_var("WAYLAND_DISPLAY");
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1250.0, 850.0])
            .with_min_inner_size([850.0, 600.0])
            .with_drag_and_drop(true)
            .with_title("Ollama Manager"),
        ..Default::default()
    };

    eframe::run_native(
        "Ollama Manager",
        options,
        Box::new(|_cc| Ok(Box::new(OllamaManager::default()))),
    )
}

fn apply_golden_honey_oak_theme(ctx: &egui::Context) {
    // Golden Honey Oak: soft ivory, warm oak, muted caramel and readable
    // walnut-brown text. Kept deliberately low-glare rather than saturated.
    let mut visuals = egui::Visuals::light();

    let background = egui::Color32::from_rgb(244, 239, 229); // warm ivory
    let base = egui::Color32::from_rgb(255, 250, 241);       // parchment input
    let panel = egui::Color32::from_rgb(237, 228, 212);      // pale oak
    let surface = egui::Color32::from_rgb(229, 215, 192);    // honey-oak controls
    let surface_hover = egui::Color32::from_rgb(218, 195, 160); // soft amber hover
    let border = egui::Color32::from_rgb(199, 178, 145);     // oak grain border
    let text = egui::Color32::from_rgb(60, 47, 35);
    let muted_text = egui::Color32::from_rgb(116, 98, 78);
    let accent = egui::Color32::from_rgb(174, 119, 53);      // honey amber
    let accent_hover = egui::Color32::from_rgb(145, 91, 39); // deeper caramel

    visuals.dark_mode = false;
    visuals.override_text_color = Some(text);
    visuals.hyperlink_color = accent_hover;
    visuals.faint_bg_color = panel;
    visuals.extreme_bg_color = base;
    visuals.code_bg_color = egui::Color32::from_rgb(247, 242, 233);
    visuals.window_fill = background;
    visuals.panel_fill = background;
    visuals.selection.bg_fill = egui::Color32::from_rgb(226, 202, 164);
    visuals.selection.stroke = egui::Stroke::new(1.0_f32, text);
    visuals.window_stroke = egui::Stroke::new(1.0_f32, border);

    visuals.widgets.noninteractive.bg_fill = panel;
    visuals.widgets.noninteractive.weak_bg_fill = panel;
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0_f32, text);
    // Avoid a dark seam between the fixed navigation panel and main content.
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;

    visuals.widgets.inactive.bg_fill = surface;
    visuals.widgets.inactive.weak_bg_fill = surface;
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0_f32, text);
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0_f32, border);

    visuals.widgets.hovered.bg_fill = surface_hover;
    visuals.widgets.hovered.weak_bg_fill = surface_hover;
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.2_f32, text);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0_f32, accent);

    visuals.widgets.active.bg_fill = accent;
    visuals.widgets.active.weak_bg_fill = accent;
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.2_f32, egui::Color32::WHITE);
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0_f32, accent_hover);

    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(9.0, 9.0);
    style.spacing.button_padding = egui::vec2(13.0, 8.0);
    style.spacing.interact_size = egui::vec2(42.0, 32.0);
    style.spacing.combo_width = 180.0;
    style.visuals.override_text_color = Some(text);
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0_f32, muted_text);
    ctx.set_style(style);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;
    use std::net::TcpListener;

    fn message(role: &str, content: &str, stopped: bool) -> ChatMessage {
        ChatMessage {
            role: role.to_owned(),
            content: content.to_owned(),
            stopped,
        }
    }

    /// Minimal fake Ollama server: answers every `/api/chat` request with
    /// `answer` and closes the connection.
    fn fake_ollama(
        answer: &'static str,
        requests: usize,
    ) -> (String, String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let handle = thread::spawn(move || {
            for _ in 0..requests {
                let (mut stream, _) = listener.accept().unwrap();
                let mut received = Vec::new();
                let mut chunk = [0u8; 4096];
                let mut header_end = None;
                let mut content_length = 0usize;

                loop {
                    // Stop once the headers and the full body are present.
                    if let Some(end) = header_end {
                        if received.len() >= end + content_length {
                            break;
                        }
                    }

                    let read = stream.read(&mut chunk).unwrap();

                    if read == 0 {
                        break;
                    }

                    received.extend_from_slice(&chunk[..read]);

                    if header_end.is_none() {
                        if let Some(pos) =
                            received.windows(4).position(|w| w == b"\r\n\r\n")
                        {
                            let end = pos + 4;
                            header_end = Some(end);

                            let headers =
                                String::from_utf8_lossy(&received[..pos]);

                            for line in headers.lines() {
                                if let Some((name, value)) = line.split_once(':')
                                {
                                    if name.eq_ignore_ascii_case("content-length")
                                    {
                                        content_length =
                                            value.trim().parse().unwrap_or(0);
                                    }
                                }
                            }
                        }
                    }
                }

                let body_start = header_end.unwrap_or(received.len());

                let payload = format!(
                    "{{\"message\":{{\"content\":\"{answer}\"}},\"done\":false}}\n{{\"done\":true}}\n"
                );

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload,
                );

                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();

                assert!(
                    received[body_start..]
                        .windows(10)
                        .any(|w| w == b"\"messages\""),
                    "request body did not contain messages: {:?}",
                    String::from_utf8_lossy(&received[body_start..]),
                );
            }
        });

        (
            "http://127.0.0.1".to_owned(),
            port.to_string(),
            handle,
        )
    }

    fn test_manager(host: &str, port: &str) -> OllamaManager {
        let (sender, receiver) = mpsc::channel();
        let mut settings = Settings::default();
        settings.host = host.to_owned();
        settings.port = port.to_owned();
        settings.speech.auto_speak = false;

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let path = std::env::temp_dir().join(format!(
            "ollama-manager-test-chat-{stamp}.json"
        ));

        OllamaManager {
            settings,
            model_name: String::new(),
            models: Vec::new(),
            status: String::new(),
            connected: true,
            busy: false,
            receiver,
            sender,
            active_tab: Tab::Chat,
            chat_model: "test-model".to_owned(),
            chat_entries: Vec::new(),
            current_chat: Conversation::default(),
            current_chat_path: Some(path),
            selected_folder: "General".to_owned(),
            folder_input: String::new(),
            draft: String::new(),
            draft_has_focus: false,
            prompts: Vec::new(),
            prompt_name: String::new(),
            prompt_content: String::new(),
            selected_prompt: String::new(),
            chat_cancel: None,
            chat_generation: 0,
            chat_active: false,
            transcribing: false,
            pending_send: false,
            recording: None,
            speech_tools: Vec::new(),
            speaking: false,
            speech_generation: 0,
        }
    }

    fn wait_until_idle(app: &mut OllamaManager) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);

        while std::time::Instant::now() < deadline {
            app.poll_events();

            if !app.chat_active {
                return;
            }

            thread::sleep(Duration::from_millis(10));
        }

        panic!("chat request did not finish in time");
    }

    #[test]
    fn two_sequential_questions_both_get_answers() {
        let (host, port, server) = fake_ollama("answer", 2);
        let mut app = test_manager(&host, &port);

        app.draft = "first question".to_owned();
        app.send_chat();
        assert!(app.chat_active);
        wait_until_idle(&mut app);

        assert_eq!(app.current_chat.messages.len(), 2);
        assert_eq!(app.current_chat.messages[1].content, "answer");

        app.draft = "second question".to_owned();
        app.send_chat();
        assert!(app.chat_active);
        wait_until_idle(&mut app);

        assert_eq!(app.current_chat.messages.len(), 4);
        assert_eq!(app.current_chat.messages[2].content, "second question");
        assert_eq!(app.current_chat.messages[3].content, "answer");

        server.join().unwrap();
        let _ = fs::remove_file(app.current_chat_path.clone().unwrap());
    }

    #[test]
    fn transcript_is_queued_while_a_reply_is_generating() {
        let mut app = test_manager("http://127.0.0.1", "1");

        // Pretend the previous reply is still streaming.
        app.chat_active = true;
        app.sender
            .send(Event::Transcription(Ok("next question".to_owned())))
            .unwrap();
        app.poll_events();

        assert_eq!(app.draft, "next question");
        assert!(app.pending_send);
        assert!(!app.transcribing);
    }

    #[test]
    fn queued_transcript_is_sent_when_the_reply_finishes() {
        let (host, port, server) = fake_ollama("answer", 1);
        let mut app = test_manager(&host, &port);

        app.current_chat.messages = vec![
            message("user", "first question", false),
            message("assistant", "first answer", false),
        ];
        app.draft = "second question".to_owned();
        app.pending_send = true;
        app.chat_active = true;
        app.chat_generation = 1;

        app.sender
            .send(Event::ChatFinished(1, Ok(())))
            .unwrap();
        app.poll_events();

        assert!(!app.pending_send);
        assert!(app.chat_active);
        assert!(app.draft.is_empty());

        wait_until_idle(&mut app);

        assert_eq!(app.current_chat.messages.len(), 4);
        assert_eq!(app.current_chat.messages[2].content, "second question");
        assert_eq!(app.current_chat.messages[3].content, "answer");

        server.join().unwrap();
        let _ = fs::remove_file(app.current_chat_path.clone().unwrap());
    }

    #[test]
    fn stopped_exchange_is_excluded_from_the_next_request() {
        let mut messages = vec![
            message("user", "first question", false),
            message("assistant", "partial answer", false),
        ];

        // The user clicks Stop while the first answer is streaming.
        mark_last_exchange_stopped(&mut messages);

        assert!(messages[0].stopped);
        assert!(messages[1].stopped);

        // The next question starts a fresh exchange.
        messages.push(message("user", "second question", false));
        messages.push(message("assistant", "", false));

        let request = messages_for_request(&messages[..3]);

        assert_eq!(request.len(), 1);
        assert_eq!(request[0].content, "second question");
    }

    #[test]
    fn completed_exchanges_are_still_sent() {
        let messages = vec![
            message("user", "first question", false),
            message("assistant", "first answer", false),
            message("user", "second question", false),
            message("assistant", "", false),
        ];

        let request = messages_for_request(&messages[..3]);

        assert_eq!(request.len(), 3);
        assert_eq!(request[1].content, "first answer");
    }

    #[test]
    fn legacy_empty_assistant_message_is_not_sent() {
        // A chat saved by an older build: the first answer was stopped before
        // any token arrived, so it has empty content and no stopped flag.
        let messages = vec![
            message("user", "first question", false),
            message("assistant", "", false),
            message("user", "second question", false),
            message("assistant", "", false),
        ];

        let request = messages_for_request(&messages[..3]);

        assert_eq!(request.len(), 1);
        assert_eq!(request[0].content, "second question");
    }

    #[test]
    fn old_chats_without_the_stopped_field_still_load() {
        let json = r#"{"role":"user","content":"hello"}"#;
        let message: ChatMessage = serde_json::from_str(json).unwrap();

        assert!(!message.stopped);
        assert_eq!(message.content, "hello");
    }
}
