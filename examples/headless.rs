use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use open_transcribe::config::{Config, OutputMode, TranslateTarget};
use open_transcribe::transcribe::{self, Event};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("uso: headless <arquivo>...   (WHISPER_OUT / WHISPER_PREFIX / WHISPER_MODE=timecode|srt|plain)");
        std::process::exit(1);
    }

    let mut cfg = if std::env::var("WHISPER_LOAD_CONFIG").as_deref() == Ok("1") {
        Config::load()
    } else {
        Config::default()
    };
    if let Ok(dir) = std::env::var("WHISPER_OUT") {
        cfg.output_dir = dir;
    }
    if let Ok(prefix) = std::env::var("WHISPER_PREFIX") {
        cfg.prefix = prefix;
    }
    if let Ok(mode) = std::env::var("WHISPER_MODE") {
        cfg.output_mode = match mode.as_str() {
            "srt" => OutputMode::Srt,
            "plain" => OutputMode::Plain,
            _ => OutputMode::Timecode,
        };
    }
    if let Ok(model) = std::env::var("WHISPER_MODEL") {
        cfg.model_path = model;
    }
    if let Ok(language) = std::env::var("WHISPER_LANGUAGE") {
        cfg.language = language;
    }
    if let Ok(prompt) = std::env::var("WHISPER_INITIAL_PROMPT") {
        cfg.initial_prompt = prompt;
    }
    if let Ok(value) = std::env::var("WHISPER_BEAM") {
        if let Ok(beam) = value.parse() {
            cfg.beam_size = beam;
        }
    }
    if let Ok(value) = std::env::var("WHISPER_VAD") {
        cfg.vad = value != "0";
    }
    if let Ok(value) = std::env::var("WHISPER_FLASH_ATTN") {
        cfg.flash_attn = value != "0";
    }
    if let Ok(value) = std::env::var("WHISPER_VAD_MIN_SILENCE") {
        if let Ok(ms) = value.parse() {
            cfg.vad_min_silence_ms = ms;
        }
    }
    if let Ok(path) = std::env::var("WHISPER_PARAGRAPH_MODEL") {
        cfg.paragraph_model_path = path;
    }
    if let Ok(value) = std::env::var("WHISPER_AI_PARAGRAPHS") {
        cfg.use_ai_paragraphs = value != "0";
    }
    if let Ok(value) = std::env::var("WHISPER_PAUSE_MS") {
        if let Ok(ms) = value.parse() {
            cfg.paragraph_pause_ms = ms;
        }
    }
    if let Ok(value) = std::env::var("WHISPER_MAX_CHARS") {
        if let Ok(chars) = value.parse() {
            cfg.paragraph_max_chars = chars;
        }
    }
    if let Ok(value) = std::env::var("WHISPER_TRANSLATE_TO") {
        cfg.translate_to = match value.as_str() {
            "en" => TranslateTarget::English,
            "pt" => TranslateTarget::Portuguese,
            "es" => TranslateTarget::Spanish,
            _ => TranslateTarget::None,
        };
    }

    let files: Vec<PathBuf> = args.into_iter().map(PathBuf::from).collect();
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    std::thread::spawn(move || transcribe::run_job(cfg, files, tx, cancel));

    for event in rx {
        match event {
            Event::Log(message) => println!("{message}"),
            Event::Partial { text, .. } => {
                print!("{text} ");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            Event::Progress { percent, .. } => {
                print!("\r  {percent}%   ");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            Event::Translating { done, total, .. } => {
                print!("\r  traduzindo {done}/{total}   ");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            Event::Done {
                output,
                chunks,
                duration,
                ..
            } => println!(
                "\nOK: {} ({} trechos, {:.1}s de áudio)",
                output.display(),
                chunks,
                duration
            ),
            Event::AllDone => break,
            other => println!("\n{other:?}"),
        }
    }
}
