use crate::config::{self, Config, OutputMode};
use crate::platform;
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::collections::BTreeSet;
#[cfg(feature = "cuda")]
use std::ffi::c_void;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadParams,
};

#[derive(Debug)]
pub enum Event {
    Log(String),
    ModelLoading,
    Extracting {
        index: usize,
    },
    Transcribing {
        index: usize,
        name: String,
    },
    Progress {
        index: usize,
        percent: i32,
    },
    Partial {
        index: usize,
        text: String,
    },
    Skipped {
        index: usize,
        output: PathBuf,
    },
    Done {
        index: usize,
        output: PathBuf,
        chunks: usize,
        duration: f64,
    },
    Failed {
        index: usize,
        error: String,
    },
    Cancelled {
        index: usize,
    },
    AllDone,
    DownloadProgress {
        label: String,
        received: u64,
        total: u64,
    },
    DownloadDone {
        label: String,
        path: PathBuf,
    },
    DownloadFailed {
        label: String,
        error: String,
    },
    DownloadCancelled {
        label: String,
    },
}

fn send(tx: &Sender<Event>, event: Event) {
    let _ = tx.send(event);
}

#[cfg(feature = "cuda")]
fn free_vram_bytes() -> Option<u64> {
    nvml_free_vram().or_else(nvidia_smi_free_vram)
}

#[cfg(feature = "cuda")]
fn nvml_free_vram() -> Option<u64> {
    #[repr(C)]
    struct NvmlMemory {
        total: u64,
        free: u64,
        used: u64,
    }

    unsafe {
        let library = ["libnvidia-ml.so.1", "nvml.dll"]
            .iter()
            .find_map(|name| libloading::Library::new(name).ok())?;
        let init: libloading::Symbol<unsafe extern "C" fn() -> i32> =
            library.get(b"nvmlInit_v2\0").ok()?;
        if init() != 0 {
            return None;
        }

        let get_handle: libloading::Symbol<unsafe extern "C" fn(u32, *mut *mut c_void) -> i32> =
            library.get(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
        let mut device = std::ptr::null_mut();
        if get_handle(0, &mut device) != 0 {
            return None;
        }

        let get_memory: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut NvmlMemory) -> i32,
        > = library.get(b"nvmlDeviceGetMemoryInfo\0").ok()?;
        let mut memory = NvmlMemory {
            total: 0,
            free: 0,
            used: 0,
        };
        if get_memory(device, &mut memory) != 0 {
            return None;
        }
        Some(memory.free)
    }
}

#[cfg(feature = "cuda")]
fn nvidia_smi_free_vram() -> Option<u64> {
    for command in ["nvidia-smi", "/usr/bin/nvidia-smi"] {
        let Ok(output) = Command::new(command)
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output()
        else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        if let Some(mib) = text.lines().next().and_then(|line| line.trim().parse::<u64>().ok()) {
            return Some(mib * 1024 * 1024);
        }
    }
    None
}

pub fn run_job(cfg: Config, files: Vec<PathBuf>, tx: Sender<Event>, cancel: Arc<AtomicBool>) {
    if let Err(err) = run_job_inner(&cfg, &files, &tx, &cancel) {
        send(&tx, Event::Log(format!("ERRO: {err:#}")));
    }
    send(&tx, Event::AllDone);
}

fn run_job_inner(
    cfg: &Config,
    files: &[PathBuf],
    tx: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> Result<()> {
    let model_path = PathBuf::from(&cfg.model_path);
    if !model_path.is_file() {
        bail!(
            "modelo não encontrado em {} — baixe o modelo em Configurações",
            model_path.display()
        );
    }

    std::fs::create_dir_all(&cfg.output_dir)
        .with_context(|| format!("não foi possível criar a pasta de saída {}", cfg.output_dir))?;

    send(tx, Event::Log("Carregando modelo Whisper...".to_string()));
    send(tx, Event::ModelLoading);

    #[cfg(feature = "cuda")]
    let mut use_gpu = cfg.use_gpu && platform::gpu_enabled();
    #[cfg(not(feature = "cuda"))]
    let use_gpu = cfg.use_gpu && platform::gpu_enabled();
    #[cfg(feature = "cuda")]
    if use_gpu {
        let model_bytes = std::fs::metadata(&model_path).map_or(0, |meta| meta.len());
        let required = model_bytes.saturating_add(1536 * 1024 * 1024);
        if let Some(free) = free_vram_bytes()
            && free < required
        {
            send(
                tx,
                Event::Log(format!(
                    "Aviso: VRAM insuficiente (livre {:.1} GiB, necessário ~{:.1} GiB); \
                     usando CPU. Feche outros programas que usam a GPU para acelerar.",
                    free as f64 / 1024f64.powi(3),
                    required as f64 / 1024f64.powi(3),
                )),
            );
            use_gpu = false;
        }
    }

    let mut ctx_params = WhisperContextParameters::default();
    ctx_params.use_gpu(use_gpu);
    if cfg.flash_attn {
        ctx_params.flash_attn(true);
    }

    let ctx = WhisperContext::new_with_params(&model_path, ctx_params)
        .map_err(|e| anyhow!("falha ao carregar o modelo: {e}"))?;

    let backend = if use_gpu {
        format!("GPU ({})", platform::gpu_backend())
    } else {
        "CPU".to_string()
    };
    send(
        tx,
        Event::Log(format!("Modelo carregado — processando em {backend}.")),
    );

    let vad_available = cfg.vad && Path::new(&cfg.vad_model_path).is_file();
    if cfg.vad && !vad_available {
        send(
            tx,
            Event::Log(
                "Aviso: modelo VAD ausente; seguindo sem remoção de silêncios.".to_string(),
            ),
        );
    }

    for (index, file) in files.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            send(tx, Event::Cancelled { index });
            return Ok(());
        }

        if let Err(err) = transcribe_file(&ctx, cfg, file, index, tx, cancel, vad_available) {
            if cancel.load(Ordering::Relaxed) {
                send(tx, Event::Cancelled { index });
                return Ok(());
            }
            send(
                tx,
                Event::Failed {
                    index,
                    error: format!("{err:#}"),
                },
            );
        }
    }

    Ok(())
}

fn transcribe_file(
    ctx: &WhisperContext,
    cfg: &Config,
    file: &Path,
    index: usize,
    tx: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
    vad_available: bool,
) -> Result<()> {
    let file_name = file
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| file.display().to_string());
    let stem = file
        .file_stem()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "transcricao".to_string());
    let prefix = config::sanitize_prefix(&cfg.prefix);
    let final_name = if prefix.is_empty() {
        stem
    } else {
        format!("{prefix} - {stem}")
    };

    let output = Path::new(&cfg.output_dir)
        .join(format!("{final_name}.{}", cfg.output_mode.extension()));
    if cfg.skip_existing && output.is_file() {
        send(tx, Event::Skipped { index, output });
        return Ok(());
    }

    send(
        tx,
        Event::Transcribing {
            index,
            name: file_name.clone(),
        },
    );
    send(tx, Event::Log(format!("Transcrevendo: {file_name}")));

    let pcm = extract_audio(file, index, tx)?;
    let duration = pcm.len() as f64 / 16_000.0;
    send(
        tx,
        Event::Log(format!(
            "Áudio extraído: {} de duração.",
            config::format_duration(duration)
        )),
    );

    if cancel.load(Ordering::Relaxed) {
        send(tx, Event::Cancelled { index });
        return Ok(());
    }

    let mut params = FullParams::new(match cfg.beam_size.max(1) {
        1 => SamplingStrategy::Greedy { best_of: 1 },
        n => SamplingStrategy::BeamSearch {
            beam_size: n,
            patience: -1.0,
        },
    });

    let language = cfg.language.trim();
    if language.is_empty() || language.eq_ignore_ascii_case("auto") {
        params.set_language(None);
        params.set_detect_language(true);
    } else {
        params.set_language(Some(language));
    }

    params.set_n_threads(cfg.threads.clamp(1, 64));
    params.set_translate(cfg.translate);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_no_timestamps(false);

    if vad_available {
        params.set_vad_model_path(Some(&cfg.vad_model_path));
        let mut vad_params = WhisperVadParams::default();
        vad_params.set_threshold(0.5);
        vad_params.set_min_silence_duration(cfg.vad_min_silence_ms.clamp(0, 60_000));
        params.set_vad_params(vad_params);
        params.enable_vad(true);
    }

    let progress_tx = tx.clone();
    params.set_progress_callback_safe(move |percent: i32| {
        send(&progress_tx, Event::Progress { index, percent });
    });

    let segment_tx = tx.clone();
    params.set_segment_callback_safe(move |data: whisper_rs::SegmentCallbackData| {
        let text = data.text.split_whitespace().collect::<Vec<_>>().join(" ");
        if !text.is_empty() {
            send(
                &segment_tx,
                Event::Partial {
                    index,
                    text,
                },
            );
        }
    });

    let abort_cancel = cancel.clone();
    params.set_abort_callback_safe(move || abort_cancel.load(Ordering::Relaxed));

    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow!("falha ao criar o estado do Whisper: {e}"))?;

    if let Err(err) = state.full(params, &pcm) {
        if cancel.load(Ordering::Relaxed) {
            send(tx, Event::Cancelled { index });
            return Ok(());
        }
        return Err(anyhow!("falha na transcrição: {err}"));
    }

    if cancel.load(Ordering::Relaxed) {
        send(tx, Event::Cancelled { index });
        return Ok(());
    }

    let segments = collect_segments(&state);
    let chunks = match cfg.output_mode {
        OutputMode::Srt => {
            let cues = build_subtitles(&segments, cfg);
            write_srt(&output, &cues, cfg)?;
            cues.len()
        }
        OutputMode::Plain => {
            let chunks = build_plain_paragraphs(&segments, cfg, tx);
            if cancel.load(Ordering::Relaxed) {
                send(tx, Event::Cancelled { index });
                return Ok(());
            }
            write_plain(&output, &chunks)?;
            chunks.len()
        }
        OutputMode::Timecode => {
            let chunks = group_segments(&segments, cfg.chunk_seconds.max(1.0));
            write_timecode(&output, &final_name, file, &chunks)?;
            chunks.len()
        }
    };

    send(
        tx,
        Event::Done {
            index,
            output,
            chunks,
            duration,
        },
    );
    Ok(())
}

struct Segment {
    start: f64,
    end: f64,
    text: String,
}

struct Chunk {
    start: f64,
    text: String,
}

struct Cue {
    start: f64,
    end: f64,
    text: String,
}

fn collect_segments(state: &whisper_rs::WhisperState) -> Vec<Segment> {
    let mut segments = Vec::new();
    for segment in state.as_iter() {
        let start = segment.start_timestamp() as f64 / 100.0;
        let end = segment.end_timestamp() as f64 / 100.0;
        let text = match segment.to_str_lossy() {
            Ok(value) => value.split_whitespace().collect::<Vec<_>>().join(" "),
            Err(_) => continue,
        };
        if text.is_empty() {
            continue;
        }
        segments.push(Segment { start, end, text });
    }
    segments
}

fn group_segments(segments: &[Segment], chunk_seconds: f64) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut text = String::new();
    let mut start: Option<f64> = None;

    for segment in segments {
        if start.is_none() {
            start = Some(segment.start);
        }
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&segment.text);

        if segment.end - start.unwrap_or(segment.start) >= chunk_seconds {
            chunks.push(Chunk {
                start: start.take().unwrap_or(segment.start),
                text: std::mem::take(&mut text),
            });
        }
    }

    if let Some(begin) = start {
        if !text.is_empty() {
            chunks.push(Chunk {
                start: begin,
                text,
            });
        }
    }

    chunks
}

const SOFT_CLAUSE_CHARS: usize = 240;
const UNIT_MAX_CHARS: usize = 700;

fn split_sentence_units(segments: &[Segment], max_unit_chars: usize) -> Vec<Segment> {
    let max_unit_chars = max_unit_chars.max(SOFT_CLAUSE_CHARS);
    let mut units = Vec::new();
    let mut pending = String::new();
    let mut pending_start = None;
    let mut pending_end = 0.0;

    for segment in segments {
        for piece in sentence_pieces(&segment.text) {
            let chunks = soft_split_piece(&piece, SOFT_CLAUSE_CHARS);
            let chunk_count = chunks.len();
            for (chunk_index, chunk) in chunks.into_iter().enumerate() {
                if chunk.is_empty() {
                    continue;
                }
                if pending.is_empty() {
                    pending_start = Some(segment.start);
                } else {
                    pending.push(' ');
                }
                pending.push_str(&chunk);
                pending_end = segment.end;

                let soft_break = chunk_index + 1 < chunk_count;
                if soft_break
                    || ends_sentence(&pending)
                    || pending.chars().count() >= max_unit_chars
                {
                    units.push(Segment {
                        start: pending_start.take().unwrap_or(segment.start),
                        end: pending_end,
                        text: std::mem::take(&mut pending),
                    });
                }
            }
        }
    }

    if !pending.is_empty() {
        units.push(Segment {
            start: pending_start.unwrap_or(0.0),
            end: pending_end,
            text: pending,
        });
    }

    units
}

fn soft_split_piece(text: &str, limit: usize) -> Vec<String> {
    let limit = limit.max(80);
    let mut pieces = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, ',' | ';' | ':') && current.chars().count() >= limit {
            let piece = current.trim();
            if !piece.is_empty() {
                pieces.push(piece.to_string());
            }
            current.clear();
        }
    }

    let piece = current.trim();
    if !piece.is_empty() {
        pieces.push(piece.to_string());
    }
    pieces
}

fn sentence_pieces(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut current = String::new();

    for (index, ch) in chars.iter().enumerate() {
        current.push(*ch);
        if !matches!(ch, '.' | '!' | '?') {
            continue;
        }

        let next = chars.get(index + 1);
        if next.is_some_and(|c| !c.is_whitespace()) {
            continue;
        }
        let previous = index.checked_sub(1).and_then(|i| chars.get(i));
        let decimal = *ch == '.'
            && previous.is_some_and(|c| c.is_ascii_digit())
            && next.is_some_and(|c| c.is_ascii_digit());
        let ellipsis = *ch == '.' && (previous == Some(&'.') || next == Some(&'.'));
        if decimal || ellipsis {
            continue;
        }

        let piece = current.trim();
        if !piece.is_empty() {
            pieces.push(piece.to_string());
        }
        current.clear();
    }

    let piece = current.trim();
    if !piece.is_empty() {
        pieces.push(piece.to_string());
    }
    pieces
}

fn ends_sentence(text: &str) -> bool {
    let text = text.trim_end();
    if text.ends_with("...") || text.ends_with('…') {
        return false;
    }
    text.chars()
        .last()
        .is_some_and(|c| matches!(c, '.' | '!' | '?'))
}

const COHESION_WINDOW: usize = 3;
const AI_BREAK_BONUS: f64 = 0.20;
const PARAGRAPH_PENALTY: f64 = 0.50;
const LENGTH_PENALTY: f64 = 0.40;
const MIN_FINAL_PARAGRAPH_CHARS: usize = 80;

fn build_plain_paragraphs(
    segments: &[Segment],
    cfg: &Config,
    tx: &Sender<Event>,
) -> Vec<Chunk> {
    let units = split_sentence_units(segments, UNIT_MAX_CHARS);
    if units.is_empty() {
        return Vec::new();
    }

    let mut scores = boundary_scores(&units, cfg);

    if cfg.use_ai_paragraphs {
        send(
            tx,
            Event::Log("Separando parágrafos com o modelo local...".to_string()),
        );
        match ai_paragraph_breaks(&units, cfg) {
            Ok(breaks) => {
                if breaks.len() * 2 >= units.len().max(2) {
                    send(
                        tx,
                        Event::Log(format!(
                            "Aviso: a IA sugeriu divisões demais ({} para {} frases); \
                             usando apenas pausas, pontuação e assunto.",
                            breaks.len(),
                            units.len()
                        )),
                    );
                } else {
                    for &index in &breaks {
                        if index < scores.len() {
                            scores[index] += AI_BREAK_BONUS;
                        }
                    }
                    send(
                        tx,
                        Event::Log("IA: separação de parágrafos concluída.".to_string()),
                    );
                }
            }
            Err(err) => {
                send(
                    tx,
                    Event::Log(format!(
                        "Aviso: separação por IA indisponível ({err:#}); \
                         usando pausas, pontuação e assunto."
                    )),
                );
            }
        }
    } else {
        send(
            tx,
            Event::Log(
                "Separando parágrafos por pausas, pontuação e assunto...".to_string(),
            ),
        );
    }

    let breaks = choose_paragraph_breaks(&units, &scores, cfg);

    let mut chunks = Vec::new();
    let mut start = 0usize;
    for index in 1..=units.len() {
        if index == units.len() || breaks.contains(&index) {
            let text = units[start..index]
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            if !text.trim().is_empty() {
                chunks.push(Chunk {
                    start: units[start].start,
                    text,
                });
            }
            start = index;
        }
    }
    send(
        tx,
        Event::Log(format!("{} parágrafo(s) no texto.", chunks.len())),
    );
    chunks
}

fn boundary_scores(units: &[Segment], cfg: &Config) -> Vec<f64> {
    let words: Vec<BTreeSet<String>> = units
        .iter()
        .map(|unit| content_words(&unit.text))
        .collect();
    let pause = cfg.paragraph_pause_ms.clamp(0, 60_000) as f64 / 1000.0;
    let mut scores = vec![0.0; units.len()];

    for index in 1..units.len() {
        let before_start = index.saturating_sub(COHESION_WINDOW);
        let after_end = (index + COHESION_WINDOW).min(units.len());
        let before = union_words(&words[before_start..index]);
        let after = union_words(&words[index..after_end]);
        let cohesion = (1.0 - jaccard_similarity(&before, &after)).clamp(0.0, 1.0);

        let gap = (units[index].start - units[index - 1].end).max(0.0);
        let pause_score = if pause > 0.0 {
            (gap / pause).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let punctuation = if ends_sentence(&units[index - 1].text) {
            1.0
        } else {
            0.0
        };
        let marker = if starts_new_topic(&units[index].text) {
            1.0
        } else {
            0.0
        };

        scores[index] = 0.55 * cohesion
            + 0.35 * pause_score
            + 0.10 * punctuation
            + 0.20 * marker;
    }

    scores
}

fn choose_paragraph_breaks(
    units: &[Segment],
    scores: &[f64],
    cfg: &Config,
) -> BTreeSet<usize> {
    let n = units.len();
    if n <= 1 {
        return BTreeSet::new();
    }

    let mut cumulative = vec![0usize; n + 1];
    for index in 0..n {
        cumulative[index + 1] =
            cumulative[index] + units[index].text.chars().count() + 1;
    }

    let longest = units
        .iter()
        .map(|unit| unit.text.chars().count() + 1)
        .max()
        .unwrap_or(0);
    let max_chars = cfg.paragraph_max_chars.max(400).max(longest);
    let min_chars = (max_chars / 3).clamp(250, 500).min(max_chars);
    let target = (max_chars as f64 * 0.66).max(min_chars as f64);
    let total = *cumulative.last().unwrap_or(&0);
    if total <= max_chars && total < min_chars {
        return BTreeSet::new();
    }

    let mut dp = vec![f64::NEG_INFINITY; n + 1];
    let mut previous = vec![0usize; n + 1];
    dp[0] = 0.0;

    for end in 1..=n {
        let mut start = end;
        while start > 0 {
            start -= 1;
            let length = cumulative[end] - cumulative[start];
            if length > max_chars {
                break;
            }
            let is_last = end == n;
            if length < min_chars
                && !(is_last && length >= MIN_FINAL_PARAGRAPH_CHARS)
            {
                continue;
            }
            if !dp[start].is_finite() {
                continue;
            }
            let boundary = if start == 0 {
                0.0
            } else {
                scores.get(start).copied().unwrap_or(0.0)
            };
            let fit = -LENGTH_PENALTY * (length as f64 - target).abs() / target;
            let value = dp[start] + boundary + fit - PARAGRAPH_PENALTY;
            if value > dp[end] {
                dp[end] = value;
                previous[end] = start;
            }
        }
    }

    if !dp[n].is_finite() {
        return BTreeSet::new();
    }

    let mut breaks = BTreeSet::new();
    let mut end = n;
    while end > 0 {
        let start = previous[end];
        if start > 0 {
            breaks.insert(start);
        }
        end = start;
    }
    breaks
}

fn content_words(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter_map(|word| {
            let word = word.to_lowercase();
            if word.chars().count() < 3 || is_stopword(&word) {
                None
            } else {
                Some(word)
            }
        })
        .collect()
}

fn is_stopword(word: &str) -> bool {
    const WORDS: &[&str] = &[
        "a", "à", "às", "ao", "aos", "aquela", "aquele", "aquilo", "as", "até",
        "bem", "cada", "como", "com", "cujo", "da", "das", "de", "dela", "dele",
        "depois", "do", "dos", "e", "é", "ela", "elas", "ele", "eles", "em",
        "entre", "era", "essa", "esse", "esta", "está", "estão", "este", "eu",
        "foi", "já", "lhe", "lhes", "mais", "mas", "me", "meu", "minha", "muito",
        "na", "nas", "não", "né", "no", "nos", "nós", "nossa", "nosso", "o",
        "os", "ou", "para", "pela", "pelo", "por", "porque", "pra", "pro",
        "quais", "qual", "quando", "que", "quem", "se", "sem", "ser", "seu",
        "só", "sua", "suas", "tá", "te", "tem", "ter", "tô", "todo", "todos",
        "toda", "todas", "um", "uma", "vai", "você", "vocês", "vou", "vos",
    ];
    WORDS.contains(&word)
}

fn union_words(sets: &[BTreeSet<String>]) -> BTreeSet<String> {
    let mut union = BTreeSet::new();
    for set in sets {
        union.extend(set.iter().cloned());
    }
    union
}

fn jaccard_similarity(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn starts_new_topic(text: &str) -> bool {
    let lower = text.trim_start().to_lowercase();
    const MARKERS: &[&str] = &[
        "a outra",
        "agora",
        "beleza",
        "concluindo",
        "e aí",
        "enfim",
        "entretanto",
        "eu quero",
        "eu vou",
        "imagina",
        "no entanto",
        "o outro",
        "outra coisa",
        "outro ponto",
        "pessoal",
        "por exemplo",
        "por outro lado",
        "porém",
        "primeiro",
        "resumindo",
        "segundo",
        "só que",
        "terceiro",
        "turma",
        "vamos",
    ];
    MARKERS.iter().any(|marker| lower.starts_with(marker))
}

#[derive(Deserialize)]
struct ParagraphResponse {
    #[serde(default)]
    breaks: Vec<usize>,
}

fn ai_paragraph_breaks(segments: &[Segment], cfg: &Config) -> Result<BTreeSet<usize>> {
    let model = Path::new(&cfg.paragraph_model_path);
    if !model.is_file() {
        bail!(
            "modelo de parágrafos não encontrado em {}",
            model.display()
        );
    }

    let sentences: Vec<&str> = segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect();
    let gaps: Vec<f64> = segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            if index == 0 {
                0.0
            } else {
                (segment.start - segments[index - 1].end).max(0.0)
            }
        })
        .collect();
    let request = serde_json::json!({
        "sentences": sentences,
        "gaps": gaps,
        "threads": cfg.threads.clamp(1, 32),
        "pause_ms": cfg.paragraph_pause_ms.clamp(0, 60_000),
    });

    let helper = paragraph_helper_path();
    let mut child = Command::new(&helper)
        .arg("--model")
        .arg(model)
        .arg("--input")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("não foi possível executar {}", helper.display()))?;

    {
        let mut stdin = child.stdin.take().context("llm-paragraph sem stdin")?;
        stdin
            .write_all(request.to_string().as_bytes())
            .context("falha ao enviar o texto para o llm-paragraph")?;
    }

    let output = child
        .wait_with_output()
        .context("erro aguardando o llm-paragraph")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let reason = stderr
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("erro desconhecido");
        bail!("llm-paragraph falhou: {}", reason.trim());
    }

    let response: ParagraphResponse =
        serde_json::from_slice(&output.stdout).context("resposta inválida do llm-paragraph")?;
    let mut breaks = BTreeSet::new();
    for index in response.breaks {
        if index > 0 && index < segments.len() {
            breaks.insert(index);
        }
    }
    Ok(breaks)
}

fn paragraph_helper_path() -> PathBuf {
    let names: &[&str] = if cfg!(target_os = "windows") {
        &["llm-paragraph.exe", "llm-paragraph"]
    } else {
        &["llm-paragraph"]
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for name in names {
                for candidate in [dir.join(name), dir.join("..").join(name)] {
                    if candidate.is_file() {
                        return candidate;
                    }
                }
            }
        }
    }
    PathBuf::from(names[0])
}

fn build_subtitles(segments: &[Segment], cfg: &Config) -> Vec<Cue> {
    let max_line = cfg.subtitle_max_line_chars.clamp(15, 80);
    let max_lines = cfg.subtitle_max_lines.clamp(1, 3);
    let max_chars = max_line * max_lines;
    let max_duration = cfg.subtitle_max_duration.clamp(1.5, 30.0);

    let mut cues = Vec::new();
    for segment in segments {
        let words: Vec<&str> = segment.text.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let weights: Vec<f64> = words
            .iter()
            .map(|word| word.chars().count() as f64 + 1.0)
            .collect();
        let total_weight: f64 = weights.iter().sum();
        let duration = (segment.end - segment.start).max(0.1);

        let mut cursor = segment.start;
        let mut cue_words: Vec<&str> = Vec::new();
        let mut cue_start = segment.start;
        let mut cue_end = segment.start;
        let mut cue_chars = 0usize;

        for (word, weight) in words.iter().zip(&weights) {
            let word_start = cursor;
            let word_end = (cursor + duration * weight / total_weight).min(segment.end);
            cursor = word_end;

            if !cue_words.is_empty() {
                let would_chars = cue_chars + 1 + word.chars().count();
                let would_duration = word_end - cue_start;
                if would_chars > max_chars || would_duration > max_duration {
                    cues.push(Cue {
                        start: cue_start,
                        end: cue_end.max(cue_start + 0.2),
                        text: cue_words.join(" "),
                    });
                    cue_words.clear();
                    cue_chars = 0;
                }
            }

            if cue_words.is_empty() {
                cue_start = word_start;
                cue_chars = word.chars().count();
            } else {
                cue_chars += 1 + word.chars().count();
            }
            cue_words.push(word);
            cue_end = word_end;
        }

        if !cue_words.is_empty() {
            cues.push(Cue {
                start: cue_start,
                end: cue_end.max(cue_start + 0.2),
                text: cue_words.join(" "),
            });
        }
    }
    cues
}

fn wrap_caption(text: &str, max_line: usize, max_lines: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        if let Some(line) = lines.last_mut() {
            if line.chars().count() + 1 + word.chars().count() <= max_line {
                line.push(' ');
                line.push_str(word);
                continue;
            }
        }
        lines.push(word.to_string());
    }
    if lines.len() > max_lines {
        let overflow = lines.split_off(max_lines);
        if let Some(last) = lines.last_mut() {
            for word in overflow {
                last.push(' ');
                last.push_str(&word);
            }
        }
    }
    lines
}

fn write_timecode(
    output: &Path,
    final_name: &str,
    source: &Path,
    chunks: &[Chunk],
) -> Result<()> {
    let extension = source
        .extension()
        .map(|ext| format!(".{}", ext.to_string_lossy()))
        .unwrap_or_default();

    let file = File::create(output)
        .with_context(|| format!("não foi possível criar {}", output.display()))?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "=== TRANSCRIÇÃO OFICIAL: {final_name}{extension} ==="
    )?;
    writeln!(writer, "=== FONTE: {} ===", source.display())?;
    writeln!(writer, "{}", "-".repeat(50))?;
    writeln!(writer)?;

    for chunk in chunks {
        writeln!(
            writer,
            "TIMECODE [{}] >> {}",
            config::format_timecode(chunk.start),
            chunk.text
        )?;
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

fn write_plain(output: &Path, chunks: &[Chunk]) -> Result<()> {
    let file = File::create(output)
        .with_context(|| format!("não foi possível criar {}", output.display()))?;
    let mut writer = BufWriter::new(file);

    for chunk in chunks {
        writeln!(writer, "{}", chunk.text)?;
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

fn write_srt(output: &Path, cues: &[Cue], cfg: &Config) -> Result<()> {
    let max_line = cfg.subtitle_max_line_chars.clamp(15, 80);
    let max_lines = cfg.subtitle_max_lines.clamp(1, 3);

    let file = File::create(output)
        .with_context(|| format!("não foi possível criar {}", output.display()))?;
    let mut writer = BufWriter::new(file);

    for (index, cue) in cues.iter().enumerate() {
        writeln!(writer, "{}", index + 1)?;
        writeln!(
            writer,
            "{} --> {}",
            config::format_srt_timestamp(cue.start),
            config::format_srt_timestamp(cue.end)
        )?;
        for line in wrap_caption(&cue.text, max_line, max_lines) {
            writeln!(writer, "{line}")?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;
    Ok(())
}

fn extract_audio(file: &Path, index: usize, tx: &Sender<Event>) -> Result<Vec<f32>> {
    send(tx, Event::Extracting { index });

    let mut child = Command::new(platform::ffmpeg_program())
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(file)
        .args([
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("não foi possível executar o ffmpeg (ele está instalado?)")?;

    let mut stdout = child.stdout.take().context("ffmpeg sem stdout")?;
    let stderr = child.stderr.take().context("ffmpeg sem stderr")?;

    let stderr_handle = std::thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut text = String::new();
        let _ = reader.read_to_string(&mut text);
        text
    });

    let mut samples: Vec<f32> = Vec::new();
    let mut buffer = vec![0u8; 64 * 1024];
    let mut carry: Vec<u8> = Vec::new();

    loop {
        let read = stdout.read(&mut buffer).context("erro lendo áudio do ffmpeg")?;
        if read == 0 {
            break;
        }
        carry.extend_from_slice(&buffer[..read]);

        let complete = carry.len() - (carry.len() % 4);
        samples.reserve(complete / 4);
        for bytes in carry[..complete].chunks_exact(4) {
            samples.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }
        carry.drain(..complete);
    }

    let status = child.wait().context("erro aguardando o ffmpeg")?;
    let stderr_text = stderr_handle.join().unwrap_or_default();

    if !status.success() {
        bail!("ffmpeg falhou: {}", stderr_text.trim());
    }
    if samples.is_empty() {
        bail!("o ffmpeg não produziu áudio (arquivo inválido ou sem faixa de áudio)");
    }

    Ok(samples)
}

pub fn download_file(
    label: String,
    url: String,
    dest: PathBuf,
    tx: Sender<Event>,
    cancel: Arc<AtomicBool>,
) {
    let result = download_inner(&label, &url, &dest, &tx, &cancel);
    match result {
        Ok(true) => send(
            &tx,
            Event::DownloadDone {
                label,
                path: dest,
            },
        ),
        Ok(false) => send(&tx, Event::DownloadCancelled { label }),
        Err(err) => send(
            &tx,
            Event::DownloadFailed {
                label,
                error: format!("{err:#}"),
            },
        ),
    }
}

fn download_inner(
    label: &str,
    url: &str,
    dest: &Path,
    tx: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> Result<bool> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("não foi possível criar {}", parent.display()))?;
    }

    let partial = dest.with_extension("part");
    let client = reqwest::blocking::Client::builder()
        .user_agent("open-transcribe/0.1")
        .build()
        .context("falha ao criar cliente HTTP")?;
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("falha ao baixar {url}"))?
        .error_for_status()
        .with_context(|| format!("servidor recusou o download de {url}"))?;

    let total = response.content_length().unwrap_or(0);
    let mut file = File::create(&partial)
        .with_context(|| format!("não foi possível criar {}", partial.display()))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut received = 0u64;

    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(&partial);
            return Ok(false);
        }
        let read = response.read(&mut buffer).context("erro durante o download")?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .context("erro gravando o arquivo baixado")?;
        received += read as u64;
        send(
            tx,
            Event::DownloadProgress {
                label: label.to_string(),
                received,
                total,
            },
        );
    }

    file.flush()?;
    drop(file);
    std::fs::rename(&partial, dest)
        .with_context(|| format!("não foi possível salvar {}", dest.display()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(start: f64, end: f64, text: &str) -> Segment {
        Segment {
            start,
            end,
            text: text.to_string(),
        }
    }

    #[test]
    fn split_sentences_at_strong_punctuation() {
        assert_eq!(
            sentence_pieces("Olá. Tudo bem? Sim... claro!"),
            vec!["Olá.", "Tudo bem?", "Sim... claro!"]
        );
        assert_eq!(
            sentence_pieces("São 5.5 milhões. Fim"),
            vec!["São 5.5 milhões.", "Fim"]
        );
    }

    #[test]
    fn keeps_units_whole_until_punctuation_or_limit() {
        let units = split_sentence_units(
            &[
                segment(0.0, 1.0, "Olá, tudo"),
                segment(2.0, 3.0, "bem? Sim."),
            ],
            1_200,
        );
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].text, "Olá, tudo bem?");
        assert_eq!(units[0].start, 0.0);
        assert_eq!(units[1].text, "Sim.");
        assert_eq!(units[1].start, 2.0);
    }

    #[test]
    fn ellipsis_is_not_a_sentence_boundary() {
        assert!(!ends_sentence("está f..."));
        assert!(!ends_sentence("espera ..."));
        assert!(ends_sentence("certo?"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn probes_free_vram() {
        eprintln!("free_vram_bytes = {:?}", free_vram_bytes());
    }

    #[test]
    fn soft_splits_long_comma_runs() {
        let text = "primeira parte da fala, ".repeat(12) + "final.";
        let pieces = soft_split_piece(&text, 240);
        assert!(pieces.len() > 1);
        assert!(pieces[..pieces.len() - 1]
            .iter()
            .all(|piece| piece.ends_with(',')));
        assert!(pieces.last().is_some_and(|piece| piece.ends_with('.')));
    }

    #[test]
    fn keeps_short_comma_runs_whole() {
        assert_eq!(
            soft_split_piece("Olá, tudo bem?", 240),
            vec!["Olá, tudo bem?"]
        );
    }

    #[test]
    fn breaks_paragraphs_within_configured_limits() {
        let units: Vec<Segment> = (0..20)
            .map(|index| {
                segment(
                    index as f64,
                    index as f64 + 0.9,
                    &"palavra ".repeat(14),
                )
            })
            .collect();
        let scores = vec![0.0; units.len()];
        let cfg = Config {
            paragraph_max_chars: 1_200,
            ..Config::default()
        };

        let breaks = choose_paragraph_breaks(&units, &scores, &cfg);
        assert!(!breaks.is_empty());

        let mut start = 0usize;
        for end in breaks
            .iter()
            .copied()
            .chain(std::iter::once(units.len()))
        {
            let length = units[start..end]
                .iter()
                .map(|unit| unit.text.chars().count() + 1)
                .sum::<usize>();
            if end == units.len() {
                assert!(length >= MIN_FINAL_PARAGRAPH_CHARS);
            } else {
                assert!(length <= 1_200);
            }
            start = end;
        }
    }

    #[test]
    fn prefers_a_strong_topic_boundary() {
        let units: Vec<Segment> = (0..10)
            .map(|index| {
                segment(
                    index as f64,
                    index as f64 + 0.9,
                    &"palavra ".repeat(14),
                )
            })
            .collect();
        let mut scores = vec![0.0; units.len()];
        scores[5] = 5.0;
        let cfg = Config {
            paragraph_max_chars: 1_200,
            ..Config::default()
        };

        let breaks = choose_paragraph_breaks(&units, &scores, &cfg);
        assert!(breaks.contains(&5));
    }
}
