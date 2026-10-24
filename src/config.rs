use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const VAD_MODEL_URL: &str =
    "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin";
pub const MODEL_FILE: &str = "ggml-large-v3.bin";
pub const VAD_FILE: &str = "ggml-silero-v5.1.2.bin";

pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub file: &'static str,
    pub size: u64,
    pub description: &'static str,
    pub speed: &'static str,
    pub languages: &'static str,
    pub recommended: bool,
}

pub const MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "large-v3",
        name: "Large v3",
        file: "ggml-large-v3.bin",
        size: 3_095_033_483,
        description: "Máxima precisão em português, inclusive termos técnicos. É o mais lento.",
        speed: "lento",
        languages: "multilíngue",
        recommended: true,
    },
    ModelInfo {
        id: "large-v3-turbo",
        name: "Large v3 Turbo",
        file: "ggml-large-v3-turbo.bin",
        size: 1_624_555_275,
        description: "Quase a precisão do large-v3, cerca de 2 a 3 vezes mais rápido. Melhor custo-benefício.",
        speed: "rápido",
        languages: "multilíngue",
        recommended: false,
    },
    ModelInfo {
        id: "large-v3-turbo-q5_0",
        name: "Large v3 Turbo (q5_0)",
        file: "ggml-large-v3-turbo-q5_0.bin",
        size: 574_041_195,
        description: "Turbo quantizado em 5 bits: download e VRAM menores, com pequena perda de precisão.",
        speed: "rápido",
        languages: "multilíngue",
        recommended: false,
    },
    ModelInfo {
        id: "large-v3-q5_0",
        name: "Large v3 (q5_0)",
        file: "ggml-large-v3-q5_0.bin",
        size: 1_081_140_203,
        description: "Large v3 quantizado: boa precisão com cerca de um terço do tamanho. Um pouco mais lento que o turbo.",
        speed: "médio",
        languages: "multilíngue",
        recommended: false,
    },
    ModelInfo {
        id: "medium",
        name: "Medium",
        file: "ggml-medium.bin",
        size: 1_533_763_059,
        description: "Boa precisão, bem mais leve que o large. Boa opção para computadores mais modestos.",
        speed: "médio",
        languages: "multilíngue",
        recommended: false,
    },
    ModelInfo {
        id: "small",
        name: "Small",
        file: "ggml-small.bin",
        size: 487_601_967,
        description: "Equilíbrio entre velocidade e precisão. Bom para rascunhos e vídeos curtos.",
        speed: "rápido",
        languages: "multilíngue",
        recommended: false,
    },
    ModelInfo {
        id: "base",
        name: "Base",
        file: "ggml-base.bin",
        size: 147_951_465,
        description: "Rápido e leve, mas erra termos técnicos e sotaques. Indicado só para testes.",
        speed: "muito rápido",
        languages: "multilíngue",
        recommended: false,
    },
    ModelInfo {
        id: "tiny",
        name: "Tiny",
        file: "ggml-tiny.bin",
        size: 77_691_713,
        description: "O mais rápido de todos. Precisão baixa; use apenas para testar o aplicativo.",
        speed: "muito rápido",
        languages: "multilíngue",
        recommended: false,
    },
];

pub fn model_url(file: &str) -> String {
    format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file}")
}

pub fn model_path(file: &str) -> PathBuf {
    models_dir().join(file)
}

pub fn model_for_path(path: &str) -> Option<&'static ModelInfo> {
    let path = Path::new(path);
    MODELS.iter().find(|info| model_path(info.file).as_path() == path)
}

pub struct ParagraphModelInfo {
    pub name: &'static str,
    pub file: &'static str,
    pub size: u64,
    pub description: &'static str,
    pub recommended: bool,
}

pub const PARAGRAPH_MODELS: &[ParagraphModelInfo] = &[
    ParagraphModelInfo {
        name: "Qwen2.5 1.5B (recomendado)",
        file: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        size: 1_117_320_736,
        description: "Melhores parágrafos. Download ~1 GB, usa cerca de 2 GB de RAM e roda na CPU.",
        recommended: true,
    },
    ParagraphModelInfo {
        name: "Qwen2.5 0.5B (leve)",
        file: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        size: 491_400_032,
        description: "Para PCs mais fracos. Download ~470 MB, usa cerca de 1 GB de RAM; parágrafos um pouco menos precisos.",
        recommended: false,
    },
];

pub fn paragraph_model_url(file: &str) -> String {
    let repo = if file.contains("1.5b") {
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    } else {
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    };
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

pub fn paragraph_model_path(file: &str) -> PathBuf {
    models_dir().join(file)
}

pub fn paragraph_model_for_path(path: &str) -> Option<&'static ParagraphModelInfo> {
    let path = Path::new(path);
    PARAGRAPH_MODELS
        .iter()
        .find(|info| paragraph_model_path(info.file).as_path() == path)
}

pub const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mkv", "avi", "mov", "m4v"];


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    #[default]
    Timecode,
    Srt,
    Plain,
}

impl OutputMode {
    pub const ALL: &'static [OutputMode] = &[
        OutputMode::Timecode,
        OutputMode::Srt,
        OutputMode::Plain,
    ];

    pub fn label(self) -> &'static str {
        match self {
            OutputMode::Timecode => "Com timecodes (.txt)",
            OutputMode::Srt => "Legenda (.srt)",
            OutputMode::Plain => "Texto corrido (.txt)",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            OutputMode::Timecode => "Blocos de texto precedidos de timecode.",
            OutputMode::Srt => "Legendas com numeração, tempo no formato SRT e linhas curtas.",
            OutputMode::Plain => "Somente o texto, em parágrafos, sem timecodes.",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            OutputMode::Timecode | OutputMode::Plain => "txt",
            OutputMode::Srt => "srt",
        }
    }
}

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .or_else(|| {
            let drive = std::env::var_os("HOMEDRIVE")?;
            let path = std::env::var_os("HOMEPATH")?;
            let mut full = PathBuf::from(drive);
            full.push(path);
            Some(full)
        })
        .unwrap_or_else(|| PathBuf::from("."))
}

#[allow(dead_code)]
fn xdg_dir(var: &str, fallback: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| home_dir().join(fallback))
}

pub fn config_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        xdg_dir("XDG_CONFIG_HOME", ".config").join("whisper-gui")
    }
    #[cfg(target_os = "macos")]
    {
        home_dir().join("Library/Application Support/whisper-gui")
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home_dir().join("AppData/Roaming"))
            .join("whisper-gui")
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        xdg_dir("XDG_CONFIG_HOME", ".config").join("whisper-gui")
    }
}

pub fn data_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        xdg_dir("XDG_DATA_HOME", ".local/share").join("whisper-gui")
    }
    #[cfg(target_os = "macos")]
    {
        home_dir().join("Library/Application Support/whisper-gui")
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home_dir().join("AppData/Local"))
            .join("whisper-gui")
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        xdg_dir("XDG_DATA_HOME", ".local/share").join("whisper-gui")
    }
}

pub fn default_videos_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        home_dir().join("Movies")
    }
    #[cfg(not(target_os = "macos"))]
    {
        home_dir().join("Videos")
    }
}

pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

pub fn default_model_path() -> PathBuf {
    models_dir().join(MODEL_FILE)
}

pub fn default_vad_model_path() -> PathBuf {
    models_dir().join(VAD_FILE)
}

pub fn default_paragraph_model_path() -> PathBuf {
    paragraph_model_path(PARAGRAPH_MODELS[0].file)
}


pub fn config_path() -> PathBuf {
    config_dir().join("config.json")
}

fn default_threads() -> i32 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    cores.clamp(1, 8) as i32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model_path: String,
    pub vad_model_path: String,
    pub language: String,
    pub translate: bool,
    pub beam_size: i32,
    pub vad: bool,
    pub vad_min_silence_ms: i32,
    pub chunk_seconds: f64,
    pub threads: i32,
    pub flash_attn: bool,
    pub use_gpu: bool,
    pub output_dir: String,
    pub videos_dir: String,
    pub prefix: String,
    pub skip_existing: bool,
    pub output_mode: OutputMode,
    pub subtitle_max_line_chars: usize,
    pub subtitle_max_lines: usize,
    pub subtitle_max_duration: f64,
    pub use_ai_paragraphs: bool,
    pub paragraph_model_path: String,
    pub paragraph_pause_ms: i32,
    pub paragraph_max_chars: usize,
    #[serde(default)]
    pub config_version: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: default_model_path().display().to_string(),
            vad_model_path: default_vad_model_path().display().to_string(),
            language: "pt".to_string(),
            translate: false,
            beam_size: 1,
            vad: true,
            vad_min_silence_ms: 500,
            chunk_seconds: 30.0,
            threads: default_threads(),
            flash_attn: false,
            use_gpu: true,
            output_dir: String::new(),
            videos_dir: default_videos_dir().display().to_string(),
            prefix: String::new(),
            skip_existing: true,
            output_mode: OutputMode::default(),
            subtitle_max_line_chars: 42,
            subtitle_max_lines: 2,
            subtitle_max_duration: 7.0,
            use_ai_paragraphs: true,
            paragraph_model_path: default_paragraph_model_path()
                .display()
                .to_string(),
            paragraph_pause_ms: 1500,
            paragraph_max_chars: 1200,
            config_version: 1,
        }
    }
}

impl Config {
    pub fn load() -> Self {
        let path = config_path();
        let mut cfg: Config = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        if cfg.migrate() {
            cfg.save();
        }
        cfg
    }

    fn migrate(&mut self) -> bool {
        if self.config_version >= 1 {
            return false;
        }
        self.config_version = 1;
        true
    }

    pub fn save(&self) {
        if let Some(parent) = config_path().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(config_path(), text);
        }
    }
}

pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_ascii_lowercase();
            VIDEO_EXTENSIONS.iter().any(|candidate| *candidate == ext)
        })
        .unwrap_or(false)
}

pub fn sanitize_prefix(prefix: &str) -> String {
    let mut out: String = prefix
        .chars()
        .filter(|c| !matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|'))
        .collect();
    while out.ends_with(' ') || out.ends_with('.') {
        out.pop();
    }
    out.trim().to_string()
}

pub fn format_timecode(seconds: f64) -> String {
    let seconds = seconds.max(0.0);
    let h = (seconds / 3600.0) as u64;
    let m = ((seconds % 3600.0) / 60.0) as u64;
    let s = (seconds % 60.0) as u64;
    format!("{h:02}:{m:02}:{s:02}")
}

pub fn format_srt_timestamp(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

pub fn format_duration(seconds: f64) -> String {
    let seconds = seconds.max(0.0) as u64;
    let h = seconds / 3600;
    let m = (seconds % 3600) / 60;
    let s = seconds % 60;
    if h > 0 {
        format!("{h}h{m:02}m")
    } else {
        format!("{m}m{s:02}s")
    }
}

pub fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
