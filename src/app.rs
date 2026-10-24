use crate::config::{self, Config, OutputMode};
use crate::hardware::HardwareInfo;
use crate::platform;
use crate::transcribe::{self, Event};
use crate::update::{self, UpdateInfo};
use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Duration;

const BUTTON_TEXT: egui::Color32 = egui::Color32::WHITE;
const START_BUTTON: egui::Color32 = egui::Color32::from_rgb(48, 145, 88);
const DELETE_BUTTON: egui::Color32 = egui::Color32::from_rgb(185, 60, 60);
const DOWNLOAD_BUTTON: egui::Color32 = egui::Color32::from_rgb(50, 105, 185);

fn colored_button(text: impl Into<String>, fill: egui::Color32) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text.into()).color(BUTTON_TEXT)).fill(fill)
}

const LANGS: &[(&str, &str)] = &[
    ("auto", "Detectar automaticamente"),
    ("pt", "Português"),
    ("en", "Inglês"),
    ("es", "Espanhol"),
    ("fr", "Francês"),
    ("it", "Italiano"),
    ("de", "Alemão"),
    ("ru", "Russo"),
    ("ja", "Japonês"),
    ("zh", "Chinês"),
];

#[derive(Clone, Debug, PartialEq)]
enum FileStatus {
    Pending,
    Extracting,
    Transcribing(i32),
    Done {
        output: PathBuf,
        chunks: usize,
        duration: f64,
    },
    Skipped(PathBuf),
    Failed(String),
    Cancelled,
}

struct FileItem {
    path: PathBuf,
    size: u64,
    status: FileStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DownloadKind {
    WhisperModel,
    ParagraphModel,
    Vad,
}

struct DownloadState {
    label: String,
    received: u64,
    total: u64,
    kind: DownloadKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeleteKind {
    WhisperModel,
    ParagraphModel,
}

#[derive(Clone, Debug)]
struct PendingDelete {
    kind: DeleteKind,
    label: String,
    path: PathBuf,
}

enum UpdateMsg {
    Checked(Result<Option<UpdateInfo>, String>),
    Progress { received: u64, total: u64 },
    Downloaded(Result<PathBuf, String>),
}

pub struct App {
    cfg: Config,
    cfg_json: String,
    files: Vec<FileItem>,
    running: bool,
    loading_model: bool,
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<Event>>,
    logs: Vec<String>,
    current: Option<String>,
    download: Option<DownloadState>,
    downloading: bool,
    download_cancel: Arc<AtomicBool>,
    ask_output_dir: bool,
    hardware: HardwareInfo,
    hardware_notes: Vec<String>,
    live_file: Option<String>,
    live_index: Option<usize>,
    live_transcript: String,
    pending_delete: Option<PendingDelete>,
    update_rx: Option<Receiver<UpdateMsg>>,
    checking_update: bool,
    update_quiet: bool,
    update_info: Option<UpdateInfo>,
    update_error: Option<String>,
    downloading_update: bool,
    update_progress: Option<(u64, u64)>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let cfg = Config::load();
        let cfg_json = serde_json::to_string(&cfg).unwrap_or_default();
        let mut app = Self {
            cfg,
            cfg_json,
            files: Vec::new(),
            running: false,
            loading_model: false,
            cancel: Arc::new(AtomicBool::new(false)),
            rx: None,
            logs: Vec::new(),
            current: None,
            download: None,
            downloading: false,
            download_cancel: Arc::new(AtomicBool::new(false)),
            ask_output_dir: false,
            hardware: HardwareInfo::detect(),
            hardware_notes: Vec::new(),
            live_file: None,
            live_index: None,
            live_transcript: String::new(),
            pending_delete: None,
            update_rx: None,
            checking_update: false,
            update_quiet: false,
            update_info: None,
            update_error: None,
            downloading_update: false,
            update_progress: None,
        };
        app.check_updates(true);
        app
    }

    fn log(&mut self, message: impl Into<String>) {
        self.logs.push(message.into());
        if self.logs.len() > 500 {
            let excess = self.logs.len() - 500;
            self.logs.drain(..excess);
        }
    }

    fn is_model_present(&self) -> bool {
        Path::new(&self.cfg.model_path).is_file()
    }

    fn is_vad_present(&self) -> bool {
        Path::new(&self.cfg.vad_model_path).is_file()
    }

    fn is_paragraph_model_present(&self) -> bool {
        Path::new(&self.cfg.paragraph_model_path).is_file()
    }

    fn pick_files(&mut self) {
        let mut dialog = rfd::FileDialog::new().add_filter(
            "Vídeos",
            &["mp4", "mkv", "avi", "mov", "m4v"],
        );
        if Path::new(&self.cfg.videos_dir).is_dir() {
            dialog = dialog.set_directory(&self.cfg.videos_dir);
        }
        if let Some(paths) = dialog.pick_files() {
            if let Some(parent) = paths.first().and_then(|p| p.parent()) {
                self.cfg.videos_dir = parent.display().to_string();
            }
            self.add_paths(paths, true);
        }
    }

    fn pick_folder(&mut self) {
        let mut dialog = rfd::FileDialog::new();
        if Path::new(&self.cfg.videos_dir).is_dir() {
            dialog = dialog.set_directory(&self.cfg.videos_dir);
        }
        if let Some(folder) = dialog.pick_folder() {
            self.cfg.videos_dir = folder.display().to_string();
            let mut paths = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&folder) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        paths.push(path);
                    }
                }
            }
            self.add_paths(paths, true);
        }
    }

    fn pick_model_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Modelo GGML", &["bin"])
            .set_directory(config::models_dir())
            .pick_file()
        {
            self.cfg.model_path = path.display().to_string();
        }
    }

    fn pick_vad_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Modelo VAD", &["bin"])
            .set_directory(config::models_dir())
            .pick_file()
        {
            self.cfg.vad_model_path = path.display().to_string();
        }
    }

    fn pick_paragraph_model_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Modelo GGUF", &["gguf"])
            .set_directory(config::models_dir())
            .pick_file()
        {
            self.cfg.paragraph_model_path = path.display().to_string();
        }
    }

    fn pick_output_dir(&mut self) {
        let mut dialog = rfd::FileDialog::new();
        if !self.cfg.output_dir.trim().is_empty()
            && Path::new(&self.cfg.output_dir).is_dir()
        {
            dialog = dialog.set_directory(&self.cfg.output_dir);
        } else if Path::new(&self.cfg.videos_dir).is_dir() {
            dialog = dialog.set_directory(&self.cfg.videos_dir);
        }
        if let Some(folder) = dialog.pick_folder() {
            self.cfg.output_dir = folder.display().to_string();
        }
    }

    fn add_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>, only_videos: bool) {
        let mut added = 0usize;
        for path in paths {
            if only_videos && !config::is_video(&path) {
                continue;
            }
            if !path.is_file() {
                continue;
            }
            if self.files.iter().any(|item| item.path == path) {
                continue;
            }
            let size = std::fs::metadata(&path)
                .map(|meta| meta.len())
                .unwrap_or(0);
            self.files.push(FileItem {
                path,
                size,
                status: FileStatus::Pending,
            });
            added += 1;
        }
        self.files.sort_by(|a, b| a.path.cmp(&b.path));
        if added > 0 {
            self.log(format!("{added} arquivo(s) adicionado(s)."));
        }
    }

    fn start(&mut self) {
        if self.running
            || self.downloading
            || self.downloading_update
            || self.files.is_empty()
        {
            return;
        }
        if !self.is_model_present() {
            self.log(
                "ERRO: modelo não encontrado. Escolha ou baixe o modelo em Configurações.",
            );
            return;
        }
        if self.cfg.output_dir.trim().is_empty() {
            self.ask_output_dir = true;
            return;
        }
        if self.cfg.output_mode == OutputMode::Plain
            && self.cfg.use_ai_paragraphs
            && !self.is_paragraph_model_present()
        {
            self.log(
                "Aviso: modelo de parágrafos ausente; esta transcrição usará \
                 pausas e pontuação.",
            );
        }

        self.cfg.save();
        for item in &mut self.files {
            item.status = FileStatus::Pending;
        }
        self.logs.clear();
        self.log("Iniciando transcrição...");
        self.current = None;
        self.live_file = None;
        self.live_index = None;
        self.live_transcript.clear();
        self.running = true;
        self.loading_model = true;

        let cfg = self.cfg.clone();
        let files: Vec<PathBuf> = self.files.iter().map(|item| item.path.clone()).collect();
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        self.cancel = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.clone();

        std::thread::spawn(move || transcribe::run_job(cfg, files, tx, cancel));
    }

    fn start_download(
        &mut self,
        label: &str,
        url: String,
        dest: PathBuf,
        kind: DownloadKind,
    ) {
        if self.downloading || self.running || self.downloading_update {
            return;
        }
        self.log(format!("Baixando {label}..."));
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        self.download_cancel = Arc::new(AtomicBool::new(false));
        self.downloading = true;
        self.download = Some(DownloadState {
            label: label.to_string(),
            received: 0,
            total: 0,
            kind,
        });
        let cancel = self.download_cancel.clone();
        let label = label.to_string();
        let url = url.to_string();
        std::thread::spawn(move || {
            transcribe::download_file(label, url, dest, tx, cancel);
        });
    }

    fn delete_model(&mut self, pending: PendingDelete) {
        match std::fs::remove_file(&pending.path) {
            Ok(()) => {
                self.log(format!(
                    "Modelo apagado: {} ({})",
                    pending.label,
                    pending.path.display()
                ));
                let path = pending.path.display().to_string();
                match pending.kind {
                    DeleteKind::WhisperModel => {
                        if self.cfg.model_path == path {
                            self.cfg.model_path = config::MODELS
                                .iter()
                                .filter_map(|info| {
                                    let path = config::model_path(info.file);
                                    path.is_file().then(|| path.display().to_string())
                                })
                                .next()
                                .unwrap_or_default();
                            if self.cfg.model_path.is_empty() {
                                self.log(
                                    "Nenhum modelo Whisper instalado; escolha ou baixe um modelo.",
                                );
                            }
                        }
                    }
                    DeleteKind::ParagraphModel => {
                        if self.cfg.paragraph_model_path == path {
                            self.cfg.paragraph_model_path = config::PARAGRAPH_MODELS
                                .iter()
                                .filter_map(|info| {
                                    let path =
                                        config::paragraph_model_path(info.file);
                                    path.is_file().then(|| path.display().to_string())
                                })
                                .next()
                                .unwrap_or_else(|| {
                                    config::default_paragraph_model_path()
                                        .display()
                                        .to_string()
                                });
                            if !self.is_paragraph_model_present() {
                                self.log(
                                    "Modelo de parágrafos apagado; o texto corrido usará pausas e pontuação.",
                                );
                            }
                        }
                    }
                }
            }
            Err(err) => self.log(format!(
                "ERRO ao apagar {}: {err}",
                pending.label
            )),
        }
        self.pending_delete = None;
    }

    fn delete_modal(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending_delete.clone() else {
            return;
        };

        let mut confirm = false;
        let mut cancel = false;
        egui::Modal::new(egui::Id::unique("delete_model_modal")).show(ctx, |ui| {
            ui.set_width(470.0);
            ui.heading("Apagar modelo?");
            ui.add_space(4.0);
            ui.label(format!("Apagar \"{}\" do disco?", pending.label));
            ui.label(
                egui::RichText::new(pending.path.display().to_string())
                    .monospace()
                    .small()
                    .weak(),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(
                    "O arquivo será removido. Você poderá baixá-lo novamente depois.",
                )
                .weak()
                .small(),
            );
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui
                    .add(colored_button("Apagar", DELETE_BUTTON))
                    .clicked()
                {
                    confirm = true;
                }
                if ui.button("Cancelar").clicked() {
                    cancel = true;
                }
            });
        });

        if confirm {
            self.delete_model(pending);
        } else if cancel {
            self.pending_delete = None;
        }
    }

    fn check_updates(&mut self, quiet: bool) {
        if self.checking_update || self.downloading_update {
            return;
        }
        self.checking_update = true;
        self.update_quiet = quiet;
        self.update_error = None;
        let (tx, rx) = mpsc::channel();
        self.update_rx = Some(rx);
        std::thread::spawn(move || {
            let result = update::check().map_err(|err| format!("{err:#}"));
            let _ = tx.send(UpdateMsg::Checked(result));
        });
    }

    fn download_update(&mut self) {
        if self.downloading_update {
            return;
        }
        let Some(info) = self.update_info.clone() else {
            return;
        };
        let Some(asset) = info.asset.clone() else {
            update::open_release_page(&info.html_url);
            return;
        };

        self.downloading_update = true;
        self.update_error = None;
        self.update_progress = Some((0, asset.size));
        let (tx, rx) = mpsc::channel();
        self.update_rx = Some(rx);
        std::thread::spawn(move || {
            let progress_tx = tx.clone();
            let result = update::download(&asset, &mut |received, total| {
                let _ = progress_tx.send(UpdateMsg::Progress { received, total });
            })
            .map_err(|err| format!("{err:#}"));
            let _ = tx.send(UpdateMsg::Downloaded(result));
        });
    }

    fn poll_updates(&mut self) {
        let mut messages = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = &self.update_rx {
            loop {
                match rx.try_recv() {
                    Ok(message) => messages.push(message),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        for message in messages {
            self.handle_update(message);
        }
        if disconnected && self.update_rx.is_some() {
            self.update_rx = None;
            self.checking_update = false;
            self.downloading_update = false;
        }
    }

    fn handle_update(&mut self, message: UpdateMsg) {
        match message {
            UpdateMsg::Checked(result) => {
                let quiet = self.update_quiet;
                self.update_quiet = false;
                self.checking_update = false;
                self.update_rx = None;
                match result {
                    Ok(Some(info)) => {
                        self.log(format!(
                            "Nova versão {} disponível.",
                            info.version
                        ));
                        self.update_info = Some(info);
                    }
                    Ok(None) => {
                        self.update_info = None;
                        if !quiet {
                            self.log("Você está na versão mais recente.");
                        }
                    }
                    Err(err) => {
                        if !quiet {
                            self.update_error =
                                Some(format!("Falha ao verificar atualizações: {err}"));
                        }
                    }
                }
            }
            UpdateMsg::Progress { received, total } => {
                self.update_progress = Some((received, total));
            }
            UpdateMsg::Downloaded(result) => {
                self.downloading_update = false;
                self.update_progress = None;
                self.update_rx = None;
                match result {
                    Ok(path) => {
                        #[cfg(target_os = "linux")]
                        match update::install_appimage(&path) {
                            Ok(installed) => {
                                self.cfg.save();
                                self.log("Atualização instalada. Reiniciando...");
                                if let Err(err) = update::restart_app(&installed) {
                                    self.log(format!(
                                        "ERRO ao reiniciar o aplicativo: {err:#}"
                                    ));
                                }
                            }
                            Err(err) => {
                                self.log(format!(
                                    "Atualização baixada em {} ({err:#}). \
                                     Abra o arquivo para instalar.",
                                    path.display()
                                ));
                                update::open_downloaded(&path);
                            }
                        }
                        #[cfg(not(target_os = "linux"))]
                        {
                            self.log(format!(
                                "Atualização baixada em {}. Abra o arquivo para instalar.",
                                path.display()
                            ));
                            update::open_downloaded(&path);
                        }
                    }
                    Err(err) => {
                        self.update_error =
                            Some(format!("Falha ao baixar a atualização: {err}"));
                    }
                }
            }
        }
    }

    fn open_output(&self) {
        if self.cfg.output_dir.trim().is_empty() {
            return;
        }
        let _ = std::fs::create_dir_all(&self.cfg.output_dir);
        platform::open_path(Path::new(&self.cfg.output_dir));
    }

    fn finished_count(&self) -> usize {
        self.files
            .iter()
            .filter(|item| {
                matches!(
                    item.status,
                    FileStatus::Done { .. }
                        | FileStatus::Skipped(_)
                        | FileStatus::Failed(_)
                        | FileStatus::Cancelled
                )
            })
            .count()
    }

    fn overall(&self) -> f32 {
        if self.files.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f32;
        for item in &self.files {
            total += match &item.status {
                FileStatus::Done { .. }
                | FileStatus::Skipped(_)
                | FileStatus::Failed(_)
                | FileStatus::Cancelled => 1.0,
                FileStatus::Transcribing(percent) => (*percent as f32 / 100.0).clamp(0.0, 1.0),
                _ => 0.0,
            };
        }
        total / self.files.len() as f32
    }

    fn poll_events(&mut self) {
        let mut events = Vec::new();
        let mut disconnected = false;
        if let Some(rx) = &self.rx {
            loop {
                match rx.try_recv() {
                    Ok(event) => events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        for event in events {
            self.handle_event(event);
        }
        if disconnected && self.rx.is_some() {
            self.rx = None;
            self.running = false;
            self.downloading = false;
            self.loading_model = false;
        }
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Log(message) => self.log(message),
            Event::ModelLoading => {
                self.loading_model = true;
            }
            Event::Extracting { index } => {
                if let Some(item) = self.files.get_mut(index) {
                    item.status = FileStatus::Extracting;
                }
            }
            Event::Transcribing { index, name } => {
                self.loading_model = false;
                self.current = Some(name.clone());
                self.live_file = Some(name);
                self.live_index = Some(index);
                self.live_transcript.clear();
                if let Some(item) = self.files.get_mut(index) {
                    item.status = FileStatus::Transcribing(0);
                }
            }
            Event::Partial { index, text } => {
                if self.live_index == Some(index) {
                    if !self.live_transcript.is_empty() {
                        self.live_transcript.push('\n');
                    }
                    self.live_transcript.push_str(&text);
                    const LIVE_MAX_BYTES: usize = 20_000;
                    if self.live_transcript.len() > LIVE_MAX_BYTES {
                        let mut cut = self.live_transcript.len() - LIVE_MAX_BYTES;
                        while !self.live_transcript.is_char_boundary(cut) {
                            cut += 1;
                        }
                        self.live_transcript.drain(..cut);
                    }
                }
            }
            Event::Progress { index, percent } => {
                if let Some(item) = self.files.get_mut(index) {
                    if matches!(item.status, FileStatus::Transcribing(_)) {
                        item.status = FileStatus::Transcribing(percent.clamp(0, 100));
                    }
                }
            }
            Event::Skipped { index, output } => {
                if let Some(item) = self.files.get_mut(index) {
                    let name = file_name(&item.path);
                    item.status = FileStatus::Skipped(output.clone());
                    self.log(format!("Já transcrito, pulando: {name}"));
                }
            }
            Event::Done {
                index,
                output,
                chunks,
                duration,
            } => {
                if let Some(item) = self.files.get_mut(index) {
                    let name = file_name(&item.path);
                    item.status = FileStatus::Done {
                        output: output.clone(),
                        chunks,
                        duration,
                    };
                    self.log(format!(
                        "Salvo: {} — {chunks} trechos",
                        output.display()
                    ));
                    let _ = name;
                }
            }
            Event::Failed { index, error } => {
                if let Some(item) = self.files.get_mut(index) {
                    let name = file_name(&item.path);
                    item.status = FileStatus::Failed(error.clone());
                    self.log(format!("ERRO em {name}: {error}"));
                }
            }
            Event::Cancelled { index } => {
                self.loading_model = false;
                let name = self.files.get(index).map(|item| file_name(&item.path));
                if let Some(item) = self.files.get_mut(index) {
                    item.status = FileStatus::Cancelled;
                }
                if let Some(name) = name {
                    self.log(format!("Cancelado: {name}"));
                }
            }
            Event::AllDone => {
                self.running = false;
                self.loading_model = false;
                self.current = None;
                self.rx = None;
                self.log("Concluído.");
            }
            Event::DownloadProgress {
                label,
                received,
                total,
            } => {
                if let Some(download) = &mut self.download {
                    if download.label == label {
                        download.received = received;
                        download.total = total;
                    }
                }
            }
            Event::DownloadDone { label, path } => {
                let kind = self
                    .download
                    .as_ref()
                    .map(|download| download.kind)
                    .unwrap_or(DownloadKind::Vad);
                match kind {
                    DownloadKind::WhisperModel => {
                        self.cfg.model_path = path.display().to_string();
                        self.log(format!("Modelo selecionado: {}", label));
                    }
                    DownloadKind::ParagraphModel => {
                        self.cfg.paragraph_model_path = path.display().to_string();
                        self.log(format!("Modelo de parágrafos selecionado: {}", label));
                    }
                    DownloadKind::Vad => {}
                }
                self.downloading = false;
                self.download = None;
                self.rx = None;
                self.log(format!(
                    "Download concluído: {} ({})",
                    label,
                    path.display()
                ));
            }
            Event::DownloadFailed { label, error } => {
                self.downloading = false;
                self.download = None;
                self.rx = None;
                self.log(format!("ERRO no download de {label}: {error}"));
            }
            Event::DownloadCancelled { label } => {
                self.downloading = false;
                self.download = None;
                self.rx = None;
                self.log(format!("Download cancelado: {label}"));
            }
        }
    }

    fn handle_drops(&mut self, ctx: &egui::Context) {
        if self.running || self.downloading {
            return;
        }
        let dropped: Vec<PathBuf> = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .map(|file| file.path().to_path_buf())
                .collect()
        });
        if !dropped.is_empty() {
            self.add_paths(dropped, true);
        }
    }

    fn top_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.heading("Open Transcribe");
            ui.label(
                egui::RichText::new(format!("v{}", update::VERSION))
                    .weak()
                    .small(),
            );
            ui.add_space(6.0);

            let backend = platform::gpu_backend();
            let backend_color = if platform::gpu_enabled() {
                egui::Color32::from_rgb(110, 200, 130)
            } else {
                egui::Color32::from_rgb(210, 170, 90)
            };
            let backend_text = backend;
            ui.label(
                egui::RichText::new(format!(" {backend_text} "))
                    .small()
                    .strong()
                    .color(egui::Color32::BLACK)
                    .background_color(backend_color),
            );
            let current_model = config::model_for_path(&self.cfg.model_path);
            let model_name = current_model.map(|info| info.name).unwrap_or("personalizado");
            ui.label(egui::RichText::new(model_name).weak());

            ui.with_layout(
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    if self.is_model_present() {
                        ui.label(
                            egui::RichText::new("modelo pronto")
                                .color(egui::Color32::from_rgb(110, 200, 130)),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new("modelo ausente")
                                .color(egui::Color32::from_rgb(220, 100, 100)),
                        );
                        if let Some(info) = current_model {
                            if ui
                                .add(
                                    colored_button("Baixar agora", DOWNLOAD_BUTTON)
                                        .small(),
                                )
                                .clicked()
                                && !self.downloading
                                && !self.running
                            {
                                self.start_download(
                                    info.name,
                                    config::model_url(info.file),
                                    config::model_path(info.file),
                                    DownloadKind::WhisperModel,
                                );
                            }
                        }
                    }
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "{} arquivo(s)",
                            self.files.len()
                        ))
                        .weak(),
                    );
                },
            );
        });
        ui.add_space(6.0);
    }

    fn files_ui(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Arquivos").strong());
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        if ui
                            .add_enabled(
                                !self.running && !self.files.is_empty(),
                                egui::Button::new("Limpar"),
                            )
                            .clicked()
                        {
                            self.files.clear();
                        }
                        if ui
                            .add_enabled(!self.running, egui::Button::new("Adicionar pasta"))
                            .clicked()
                        {
                            self.pick_folder();
                        }
                        if ui
                            .add_enabled(!self.running, egui::Button::new("Adicionar vídeos"))
                            .clicked()
                        {
                            self.pick_files();
                        }
                    },
                );
            });
            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);

            if self.files.is_empty() {
                ui.label(
                    egui::RichText::new(
                        "Arraste vídeos para cá ou use os botões acima.",
                    )
                    .weak(),
                );
                return;
            }

            let running = self.running;
            let mut remove = None;
            egui::ScrollArea::vertical()
                .max_height(190.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for (index, item) in self.files.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("{:>2}.", index + 1))
                                    .weak()
                                    .monospace(),
                            );
                            ui.add(
                                egui::Label::new(file_name(&item.path))
                                    .truncate()
                                    .sense(egui::Sense::hover()),
                            )
                            .on_hover_text(item.path.display().to_string());
                            ui.label(
                                egui::RichText::new(config::human_size(item.size))
                                    .weak()
                                    .small(),
                            );
                            let (text, color) = status_label(&item.status);
                            ui.label(egui::RichText::new(text).color(color).small());
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add_enabled(
                                            !running,
                                            egui::Button::new("✕").small(),
                                        )
                                        .clicked()
                                    {
                                        remove = Some(index);
                                    }
                                },
                            );
                        });
                    }
                });

            if let Some(index) = remove {
                self.files.remove(index);
            }
        });
    }

    fn settings_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        egui::CollapsingHeader::new(
            egui::RichText::new("Configurações").strong(),
        )
        .default_open(false)
        .show(ui, |ui| {
            let mut pick_model = false;
            let mut pick_vad = false;
            let mut pick_paragraph_model = false;
            let mut pick_output = false;
            let mut download_model: Option<&'static config::ModelInfo> = None;
            let mut select_model: Option<String> = None;
            let mut download_paragraph_model: Option<&'static config::ParagraphModelInfo> =
                None;
            let mut select_paragraph_model: Option<String> = None;
            let mut download_vad = false;
            let mut reset = false;
            let mut detect_hardware = false;
            let mut apply_recommendation = false;
            let mut delete_model: Option<PendingDelete> = None;
            let mut check_updates = false;
            let mut download_update = false;

            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Hardware detectado").strong());
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui
                                .add_enabled(
                                    !self.running && !self.downloading,
                                    egui::Button::new("Aplicar recomendação"),
                                )
                                .on_hover_text(
                                    "Ajusta modelo, GPU, threads e modelo de parágrafos para este computador.",
                                )
                                .clicked()
                            {
                                apply_recommendation = true;
                            }
                            if ui
                                .add_enabled(
                                    !self.running && !self.downloading,
                                    egui::Button::new("Detectar novamente"),
                                )
                                .clicked()
                            {
                                detect_hardware = true;
                            }
                        },
                    );
                });
                ui.label(
                    egui::RichText::new(self.hardware.summary())
                        .weak()
                        .small(),
                );
                for note in &self.hardware_notes {
                    ui.label(
                        egui::RichText::new(format!("• {note}"))
                            .color(egui::Color32::from_rgb(120, 175, 230))
                            .small(),
                    );
                }
                if self.hardware_notes.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "Clique em \"Aplicar recomendação\" para ajustar \
                             modelo, GPU e threads automaticamente.",
                        )
                        .weak()
                        .small(),
                    );
                }
            });
            ui.add_space(8.0);

            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Atualizações").strong());
                    ui.label(
                        egui::RichText::new(format!("versão {}", update::VERSION))
                            .weak()
                            .small(),
                    );
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if self.downloading_update {
                                ui.spinner();
                            } else if let Some(info) = &self.update_info {
                                if info.asset.is_some() {
                                    if ui
                                        .add_enabled(
                                            !self.running,
                                            colored_button(
                                                "Baixar e atualizar",
                                                DOWNLOAD_BUTTON,
                                            ),
                                        )
                                        .clicked()
                                    {
                                        download_update = true;
                                    }
                                } else if ui
                                    .button("Abrir página do release")
                                    .clicked()
                                {
                                    update::open_release_page(&info.html_url);
                                }
                            } else if !self.checking_update
                                && ui
                                    .add_enabled(
                                        !self.running,
                                        egui::Button::new("Verificar atualizações"),
                                    )
                                    .clicked()
                            {
                                check_updates = true;
                            }
                        },
                    );
                });
                if self.checking_update {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(
                            egui::RichText::new("Verificando atualizações...")
                                .weak()
                                .small(),
                        );
                    });
                }
                if let Some(info) = &self.update_info {
                    ui.label(
                        egui::RichText::new(format!(
                            "Nova versão {} disponível.",
                            info.version
                        ))
                        .color(egui::Color32::from_rgb(110, 200, 130))
                        .small(),
                    );
                    if !info.notes.is_empty() {
                        egui::CollapsingHeader::new("Notas da versão").show(
                            ui,
                            |ui| {
                                ui.label(egui::RichText::new(&info.notes).small());
                            },
                        );
                    }
                }
                if let Some((received, total)) = self.update_progress {
                    let fraction = if total > 0 {
                        received as f32 / total as f32
                    } else {
                        0.0
                    };
                    let text = if total > 0 {
                        format!(
                            "Baixando atualização — {} / {}",
                            config::human_size(received),
                            config::human_size(total)
                        )
                    } else {
                        "Baixando atualização...".to_string()
                    };
                    ui.add(egui::ProgressBar::new(fraction).text(text));
                }
                if let Some(error) = &self.update_error {
                    ui.label(
                        egui::RichText::new(error)
                            .color(egui::Color32::from_rgb(230, 110, 110))
                            .small(),
                    );
                }
            });
            ui.add_space(8.0);

            egui::Grid::new("settings_grid")
                .num_columns(2)
                .spacing([14.0, 8.0])
                .show(ui, |ui| {
                    ui.label("Modelo");
                    ui.vertical(|ui| {
                        egui::ScrollArea::vertical()
                            .id_salt("model_list")
                            .max_height(330.0)
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                for info in config::MODELS {
                                    let path = config::model_path(info.file);
                                    let path_str = path.display().to_string();
                                    let exists = path.is_file();
                                    let selected = self.cfg.model_path == path_str;
                                    ui.horizontal(|ui| {
                                        if ui
                                            .radio(selected, "")
                                            .on_hover_text("Usar este modelo")
                                            .clicked()
                                        {
                                            select_model = Some(path_str.clone());
                                        }
                                        ui.vertical(|ui| {
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    egui::RichText::new(info.name).strong(),
                                                );
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "· {}",
                                                        config::human_size(info.size)
                                                    ))
                                                    .weak()
                                                    .small(),
                                                );
                                                if info.recommended {
                                                    ui.label(
                                                        egui::RichText::new(" recomendado ")
                                                            .small()
                                                            .strong()
                                                            .color(egui::Color32::BLACK)
                                                            .background_color(
                                                                egui::Color32::from_rgb(
                                                                    110, 200, 130,
                                                                ),
                                                            ),
                                                    );
                                                }
                                                if selected {
                                                    ui.label(
                                                        egui::RichText::new(" em uso ")
                                                            .small()
                                                            .strong()
                                                            .color(egui::Color32::BLACK)
                                                            .background_color(
                                                                egui::Color32::from_rgb(
                                                                    100, 160, 230,
                                                                ),
                                                            ),
                                                    );
                                                }
                                            });
                                            ui.label(
                                                egui::RichText::new(info.description)
                                                    .weak()
                                                    .small(),
                                            );
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "{} • {}",
                                                    info.speed, info.languages
                                                ))
                                                .weak()
                                                .small(),
                                            );
                                        });
                                        ui.with_layout(
                                            egui::Layout::right_to_left(
                                                egui::Align::Center,
                                            ),
                                            |ui| {
                                                if exists {
                                                    if ui
                                                        .add_enabled(
                                                            !self.downloading && !self.running,
                                                            colored_button(
                                                                "Apagar",
                                                                DELETE_BUTTON,
                                                            )
                                                            .small(),
                                                        )
                                                        .on_hover_text(
                                                            "Remove este modelo do disco.",
                                                        )
                                                        .clicked()
                                                    {
                                                        delete_model = Some(PendingDelete {
                                                            kind: DeleteKind::WhisperModel,
                                                            label: info.name.to_string(),
                                                            path: path.clone(),
                                                        });
                                                    }
                                                    ui.label(
                                                        egui::RichText::new("✓ baixado")
                                                            .color(egui::Color32::from_rgb(
                                                                110, 200, 130,
                                                            ))
                                                            .small(),
                                                    );
                                                } else if ui
                                                    .add_enabled(
                                                        !self.downloading && !self.running,
                                                        colored_button(
                                                            format!(
                                                                "Baixar ({})",
                                                                config::human_size(info.size)
                                                            ),
                                                            DOWNLOAD_BUTTON,
                                                        ),
                                                    )
                                                    .clicked()
                                                {
                                                    download_model = Some(info);
                                                }
                                            },
                                        );
                                    });
                                    ui.separator();
                                }
                            });
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label("Personalizado:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.cfg.model_path)
                                    .desired_width(380.0)
                                    .hint_text("caminho para um .bin do whisper.cpp"),
                            );
                            pick_model = ui.button("Escolher…").clicked();
                            if !self.is_model_present() {
                                ui.label(
                                    egui::RichText::new("✗ não encontrado")
                                        .color(egui::Color32::from_rgb(220, 100, 100))
                                        .small(),
                                );
                            }
                        });
                    });
                    ui.end_row();

                    ui.label("Idioma");
                    ui.horizontal(|ui| {
                        let selected = LANGS
                            .iter()
                            .find(|(code, _)| *code == self.cfg.language)
                            .map(|(_, name)| *name)
                            .unwrap_or(self.cfg.language.as_str());
                        egui::ComboBox::from_id_salt("language")
                            .selected_text(selected)
                            .width(190.0)
                            .show_ui(ui, |ui| {
                                for (code, name) in LANGS {
                                    ui.selectable_value(
                                        &mut self.cfg.language,
                                        code.to_string(),
                                        *name,
                                    );
                                }
                            });
                        ui.add(
                            egui::TextEdit::singleline(&mut self.cfg.language)
                                .desired_width(52.0),
                        )
                        .on_hover_text("Código do idioma (ex.: pt, en, es). Use auto para detectar.");
                    });
                    ui.end_row();

                    ui.label("Traduzir");
                    ui.checkbox(
                        &mut self.cfg.translate,
                        "Traduzir a fala para inglês",
                    );
                    ui.end_row();

                    ui.label("Precisão");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.cfg.beam_size)
                                .range(1..=10)
                                .speed(0.1),
                        );
                        ui.label(
                            egui::RichText::new(
                                "beam size — 1 = rápido e padrão",
                            )
                            .weak()
                            .small(),
                        );
                    });
                    ui.end_row();

                    ui.label("CPU threads");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.cfg.threads)
                                .range(1..=64)
                                .speed(0.2),
                        );
                        ui.checkbox(&mut self.cfg.flash_attn, "Flash attention")
                            .on_hover_text(
                                "Acelera bastante, mas pode alterar levemente o resultado.",
                            );
                    });
                    ui.end_row();

                    ui.label("GPU");
                    ui.add_enabled_ui(platform::gpu_enabled(), |ui| {
                        ui.checkbox(
                            &mut self.cfg.use_gpu,
                            format!("Usar GPU ({})", platform::gpu_backend()),
                        );
                    });
                    ui.end_row();

                    ui.label("VAD");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.cfg.vad, "Remover silêncios");
                        ui.add_enabled(
                            self.cfg.vad,
                            egui::DragValue::new(&mut self.cfg.vad_min_silence_ms)
                                .range(0..=10_000)
                                .speed(10.0)
                                .suffix(" ms"),
                        );
                    });
                    ui.end_row();

                    ui.label("Modelo VAD");
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.cfg.vad_model_path)
                                    .desired_width(380.0),
                            );
                            pick_vad = ui.button("Escolher…").clicked();
                        });
                        ui.horizontal(|ui| {
                            if self.is_vad_present() {
                                ui.label(
                                    egui::RichText::new("✓ encontrado")
                                        .color(egui::Color32::from_rgb(110, 200, 130))
                                        .small(),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("✗ não encontrado")
                                        .color(egui::Color32::from_rgb(220, 100, 100))
                                        .small(),
                                );
                                download_vad = ui
                                    .add_enabled(
                                        !self.downloading && !self.running,
                                        colored_button(
                                            "Baixar VAD (~1 MB)",
                                            DOWNLOAD_BUTTON,
                                        ),
                                    )
                                    .clicked();
                            }
                        });
                    });
                    ui.end_row();

                    ui.label("Formato");
                    ui.vertical(|ui| {
                        egui::ComboBox::from_id_salt("output_mode")
                            .selected_text(self.cfg.output_mode.label())
                            .width(240.0)
                            .show_ui(ui, |ui| {
                                for mode in OutputMode::ALL {
                                    ui.selectable_value(
                                        &mut self.cfg.output_mode,
                                        *mode,
                                        mode.label(),
                                    );
                                }
                            });
                        ui.label(
                            egui::RichText::new(self.cfg.output_mode.description())
                                .weak()
                                .small(),
                        );
                    });
                    ui.end_row();

                    if self.cfg.output_mode == OutputMode::Plain {
                        ui.label("Parágrafos");
                        ui.vertical(|ui| {
                            ui.label(
                                egui::RichText::new(
                                    "O Whisper não sabe onde termina um assunto. Para o \
                                     texto corrido, é necessário baixar uma vez um \
                                     pequeno modelo de linguagem local (Qwen2.5), que lê \
                                     a transcrição e marca os parágrafos. Ele roda só na \
                                     sua CPU, sem enviar nada para a internet.",
                                )
                                .small(),
                            );
                            ui.checkbox(
                                &mut self.cfg.use_ai_paragraphs,
                                "Separar parágrafos com IA local (recomendado)",
                            );
                            if self.cfg.use_ai_paragraphs {
                                ui.add_space(4.0);
                                egui::ScrollArea::vertical()
                                    .id_salt("paragraph_model_list")
                                    .max_height(160.0)
                                    .auto_shrink([false, true])
                                    .show(ui, |ui| {
                                        for info in config::PARAGRAPH_MODELS {
                                            let path =
                                                config::paragraph_model_path(info.file);
                                            let path_str = path.display().to_string();
                                            let exists = path.is_file();
                                            let selected =
                                                self.cfg.paragraph_model_path == path_str;
                                            ui.horizontal(|ui| {
                                                if ui
                                                    .radio(selected, "")
                                                    .on_hover_text("Usar este modelo")
                                                    .clicked()
                                                {
                                                    select_paragraph_model =
                                                        Some(path_str.clone());
                                                }
                                                ui.vertical(|ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.label(
                                                            egui::RichText::new(info.name)
                                                                .strong(),
                                                        );
                                                        ui.label(
                                                            egui::RichText::new(format!(
                                                                "· {}",
                                                                config::human_size(
                                                                    info.size
                                                                )
                                                            ))
                                                            .weak()
                                                            .small(),
                                                        );
                                                        if selected {
                                                            ui.label(
                                                                egui::RichText::new(
                                                                    " em uso ",
                                                                )
                                                                .small()
                                                                .strong()
                                                                .color(
                                                                    egui::Color32::BLACK,
                                                                )
                                                                .background_color(
                                                                    egui::Color32::from_rgb(
                                                                        100, 160, 230,
                                                                    ),
                                                                ),
                                                            );
                                                        }
                                                    });
                                                    ui.label(
                                                        egui::RichText::new(
                                                            info.description,
                                                        )
                                                        .weak()
                                                        .small(),
                                                    );
                                                });
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        if exists {
                                                            if ui
                                                                .add_enabled(
                                                                    !self.downloading
                                                                        && !self.running,
                                                                    colored_button(
                                                                        "Apagar",
                                                                        DELETE_BUTTON,
                                                                    )
                                                                    .small(),
                                                                )
                                                                .on_hover_text(
                                                                    "Remove este modelo do disco.",
                                                                )
                                                                .clicked()
                                                            {
                                                                delete_model =
                                                                    Some(PendingDelete {
                                                                        kind: DeleteKind::ParagraphModel,
                                                                        label: info
                                                                            .name
                                                                            .to_string(),
                                                                        path: path.clone(),
                                                                    });
                                                            }
                                                            ui.label(
                                                                egui::RichText::new(
                                                                    "✓ baixado",
                                                                )
                                                                .color(
                                                                    egui::Color32::from_rgb(
                                                                        110, 200, 130,
                                                                    ),
                                                                )
                                                                .small(),
                                                            );
                                                        } else if ui
                                                            .add_enabled(
                                                                !self.downloading
                                                                    && !self.running,
                                                                colored_button(
                                                                    format!(
                                                                        "Baixar ({})",
                                                                        config::human_size(
                                                                            info.size
                                                                        )
                                                                    ),
                                                                    DOWNLOAD_BUTTON,
                                                                ),
                                                            )
                                                            .clicked()
                                                        {
                                                            download_paragraph_model =
                                                                Some(info);
                                                        }
                                                    },
                                                );
                                            });
                                            ui.separator();
                                        }
                                    });
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::TextEdit::singleline(
                                            &mut self.cfg.paragraph_model_path,
                                        )
                                        .desired_width(380.0)
                                        .hint_text("caminho para um .gguf do Qwen"),
                                    );
                                    pick_paragraph_model =
                                        ui.button("Escolher…").clicked();
                                    if !self.is_paragraph_model_present() {
                                        ui.label(
                                            egui::RichText::new("✗ ausente")
                                                .color(egui::Color32::from_rgb(
                                                    220, 100, 100,
                                                ))
                                                .small(),
                                        );
                                    }
                                });
                                if !self.is_paragraph_model_present() {
                                    ui.label(
                                        egui::RichText::new(
                                            "Sem o modelo, esta transcrição usará pausas e \
                                             pontuação.",
                                        )
                                        .color(egui::Color32::from_rgb(225, 190, 100))
                                        .small(),
                                    );
                                }
                            } else {
                                ui.label(
                                    egui::RichText::new(
                                        "Sem IA: os parágrafos serão definidos por pausas \
                                         e pontuação.",
                                    )
                                    .weak()
                                    .small(),
                                );
                            }
                            ui.horizontal(|ui| {
                                ui.label("Pausa");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut self.cfg.paragraph_pause_ms,
                                    )
                                    .range(0..=10_000)
                                    .speed(50.0)
                                    .suffix(" ms"),
                                )
                                .on_hover_text(
                                    "Silêncio mínimo para sugerir um novo parágrafo. \
                                     Também é usada pela IA.",
                                );
                                ui.label("Limite");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut self.cfg.paragraph_max_chars,
                                    )
                                    .range(200..=4000)
                                    .speed(10.0)
                                    .suffix(" car."),
                                )
                                .on_hover_text(
                                    "Tamanho máximo aproximado de cada parágrafo.",
                                );
                            });
                        });
                        ui.end_row();
                    }

                    if self.cfg.output_mode == OutputMode::Srt {
                        ui.label("Legenda");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(
                                    &mut self.cfg.subtitle_max_line_chars,
                                )
                                .range(15..=80)
                                .suffix(" car./linha"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.cfg.subtitle_max_lines)
                                    .range(1..=3)
                                    .suffix(" linhas"),
                            );
                            ui.add(
                                egui::DragValue::new(
                                    &mut self.cfg.subtitle_max_duration,
                                )
                                .range(2.0..=30.0)
                                .speed(0.5)
                                .suffix(" s máx."),
                            );
                        });
                        ui.end_row();
                    }

                    if self.cfg.output_mode == OutputMode::Timecode {
                        ui.label("Trecho");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.cfg.chunk_seconds)
                                    .range(5.0..=300.0)
                                    .speed(1.0)
                                    .suffix(" s"),
                            );
                            ui.label(
                                egui::RichText::new(
                                    "duração de cada bloco com timecode",
                                )
                                .weak()
                                .small(),
                            );
                        });
                        ui.end_row();
                    }

                    ui.label("Saída");
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.cfg.output_dir)
                                    .desired_width(430.0)
                                    .hint_text("obrigatório — escolha uma pasta"),
                            );
                            pick_output = ui.button("Escolher…").clicked();
                            if ui
                                .add_enabled(
                                    !self.cfg.output_dir.trim().is_empty(),
                                    egui::Button::new("Abrir"),
                                )
                                .clicked()
                            {
                                self.open_output();
                            }
                        });
                        if self.cfg.output_dir.trim().is_empty() {
                            ui.label(
                                egui::RichText::new(
                                    "Escolha a pasta onde as transcrições serão salvas.",
                                )
                                .color(egui::Color32::from_rgb(220, 100, 100))
                                .small(),
                            );
                        }
                    });
                    ui.end_row();

                    ui.label("Prefixo");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.cfg.prefix)
                            .desired_width(430.0)
                            .hint_text("ex.: mentoria → 'mentoria - nome do vídeo.txt'"),
                    );
                    ui.end_row();

                    ui.label("Repetidos");
                    ui.checkbox(
                        &mut self.cfg.skip_existing,
                        "Pular arquivos que já foram transcritos",
                    );
                    ui.end_row();
                });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("Restaurar padrões").clicked() {
                    reset = true;
                }
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(
                        "Padrão: large-v3, português, beam 1, VAD 500 ms. A pasta de saída é escolhida por você.",
                    )
                    .weak()
                    .small(),
                );
            });

            if check_updates {
                self.check_updates(false);
            }
            if download_update {
                self.download_update();
            }
            if detect_hardware {
                self.hardware = HardwareInfo::detect();
                self.hardware_notes.clear();
                self.log("Hardware detectado novamente.");
            }
            if apply_recommendation {
                let recommendation = self.hardware.recommendation(&self.cfg);
                self.cfg = recommendation.config;
                self.hardware_notes = recommendation.notes;
                let notes = self.hardware_notes.clone();
                for note in notes {
                    self.log(note);
                }
                self.log("Configuração recomendada aplicada.");
            }
            if let Some(pending) = delete_model {
                self.pending_delete = Some(pending);
            }
            if reset {
                self.cfg = Config::default();
                self.log("Configurações restauradas para o padrão.");
            }
            if pick_model {
                self.pick_model_file();
            }
            if pick_vad {
                self.pick_vad_file();
            }
            if pick_paragraph_model {
                self.pick_paragraph_model_file();
            }
            if pick_output {
                self.pick_output_dir();
            }
            if let Some(path) = select_model {
                self.cfg.model_path = path;
            }
            if let Some(path) = select_paragraph_model {
                self.cfg.paragraph_model_path = path;
            }
            if let Some(info) = download_model {
                self.start_download(
                    info.name,
                    config::model_url(info.file),
                    config::model_path(info.file),
                    DownloadKind::WhisperModel,
                );
            }
            if let Some(info) = download_paragraph_model {
                self.start_download(
                    info.name,
                    config::paragraph_model_url(info.file),
                    config::paragraph_model_path(info.file),
                    DownloadKind::ParagraphModel,
                );
            }
            if download_vad {
                let dest = PathBuf::from(&self.cfg.vad_model_path);
                self.start_download(
                    "Modelo VAD",
                    config::VAD_MODEL_URL.to_string(),
                    dest,
                    DownloadKind::Vad,
                );
            }
        });
    }

    fn output_dir_modal(&mut self, ctx: &egui::Context) {
        if !self.ask_output_dir {
            return;
        }

        let mut pick_folder = false;
        let mut confirm = false;
        let mut cancel = false;

        egui::Modal::new(egui::Id::unique("output_dir_modal")).show(ctx, |ui| {
            ui.set_width(540.0);
            ui.heading("Onde salvar as transcrições?");
            ui.add_space(4.0);
            ui.label(
                "Escolha a pasta de saída antes de iniciar a transcrição.",
            );
            ui.add_space(8.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.cfg.output_dir)
                    .desired_width(f32::INFINITY)
                    .hint_text("Escolha uma pasta para salvar as transcrições"),
            );
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                pick_folder = ui.button("Escolher pasta…").clicked();
                if ui
                    .add_enabled(
                        !self.cfg.output_dir.trim().is_empty(),
                        egui::Button::new("Salvar e transcrever"),
                    )
                    .clicked()
                {
                    confirm = true;
                }
                if ui.button("Cancelar").clicked() {
                    cancel = true;
                }
            });
        });

        if pick_folder {
            self.pick_output_dir();
            if !self.cfg.output_dir.trim().is_empty() {
                self.ask_output_dir = false;
                self.start();
            }
        }
        if confirm {
            self.ask_output_dir = false;
            self.start();
        }
        if cancel {
            self.ask_output_dir = false;
        }
    }

    fn bottom_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        let finished = self.finished_count();
        let total = self.files.len();
        let overall = self.overall();

        ui.horizontal(|ui| {
            if self.running || self.downloading {
                ui.spinner();
            }
            let bar = egui::ProgressBar::new(overall)
                .desired_width((ui.available_width() * 0.42).max(180.0))
                .text(format!("{finished}/{total}"));
            if self.loading_model {
                ui.add(bar.animate(true));
                ui.label(egui::RichText::new("carregando modelo...").weak());
            } else {
                ui.add(bar);
            }
            if let Some(current) = &self.current {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(current).weak().small(),
                    )
                    .truncate(),
                );
            }
            ui.with_layout(
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    if UiActions::should_start(ui, self) {
                        self.start();
                    }
                },
            );
        });

        if let Some(download) = &self.download {
            let fraction = if download.total > 0 {
                download.received as f32 / download.total as f32
            } else {
                0.0
            };
            let total_text = if download.total > 0 {
                config::human_size(download.total)
            } else {
                "?".to_string()
            };
            ui.add(
                egui::ProgressBar::new(fraction).text(format!(
                    "Baixando {} — {} / {}",
                    download.label,
                    config::human_size(download.received),
                    total_text
                )),
            );
        }

        if let Some(file) = &self.live_file {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let title = if self.running {
                    "Transcrição ao vivo"
                } else {
                    "Última transcrição"
                };
                ui.label(egui::RichText::new(title).strong().small());
                ui.label(egui::RichText::new(file).weak().small());
            });
            egui::ScrollArea::vertical()
                .id_salt("live_transcript")
                .max_height(110.0)
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if self.live_transcript.trim().is_empty() {
                        ui.label(
                            egui::RichText::new("Aguardando os primeiros trechos...")
                                .weak()
                                .small(),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(&self.live_transcript)
                                .monospace()
                                .size(12.0),
                        );
                    }
                });
        }

        ui.add_space(4.0);
        ui.separator();
        egui::ScrollArea::vertical()
            .max_height(165.0)
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if self.logs.is_empty() {
                    ui.label(egui::RichText::new("Sem mensagens ainda.").weak());
                }
                for line in &self.logs {
                    ui.label(
                        egui::RichText::new(line)
                            .monospace()
                            .size(12.0),
                    );
                }
            });
        ui.add_space(4.0);
    }
}

struct UiActions;

impl UiActions {
    fn should_start(ui: &mut egui::Ui, app: &mut App) -> bool {
        if app.running {
            if ui.button("Parar").clicked() {
                app.cancel.store(true, Ordering::Relaxed);
                app.log("Parando após o trecho atual...");
            }
            false
        } else {
            if ui
                .add_enabled(
                    !app.files.is_empty()
                        && !app.downloading
                        && !app.downloading_update,
                    colored_button("Iniciar transcrição", START_BUTTON),
                )
                .clicked()
            {
                return true;
            }
            if app.downloading {
                if ui.button("Cancelar download").clicked() {
                    app.download_cancel.store(true, Ordering::Relaxed);
                }
            } else if ui
                .add_enabled(
                    !app.cfg.output_dir.trim().is_empty(),
                    egui::Button::new("Abrir saída"),
                )
                .clicked()
            {
                app.open_output();
            }
            false
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_updates();
        self.poll_events();
        self.handle_drops(ui.ctx());

        egui::Panel::top("header").show(ui, |ui| self.top_ui(ui));
        egui::Panel::bottom("footer")
            .resizable(true)
            .default_size(335.0)
            .min_size(180.0)
            .show(ui, |ui| self.bottom_ui(ui));
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    self.files_ui(ui);
                    self.settings_ui(ui);
                });
        });

        self.output_dir_modal(ui.ctx());
        self.delete_modal(ui.ctx());

        if let Ok(json) = serde_json::to_string(&self.cfg) {
            if json != self.cfg_json {
                self.cfg.save();
                self.cfg_json = json;
            }
        }

        if self.running
            || self.downloading
            || self.checking_update
            || self.downloading_update
        {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string())
}

fn status_label(status: &FileStatus) -> (String, egui::Color32) {
    let green = egui::Color32::from_rgb(110, 200, 130);
    let red = egui::Color32::from_rgb(230, 110, 110);
    let yellow = egui::Color32::from_rgb(225, 190, 100);
    let blue = egui::Color32::from_rgb(120, 175, 230);
    let gray = egui::Color32::from_rgb(160, 160, 160);

    match status {
        FileStatus::Pending => ("na fila".to_string(), gray),
        FileStatus::Extracting => ("extraindo áudio...".to_string(), yellow),
        FileStatus::Transcribing(percent) => (format!("transcrevendo {percent}%"), blue),
        FileStatus::Done {
            chunks, duration, ..
        } => (
            format!(
                "ok • {} trechos • {}",
                chunks,
                config::format_duration(*duration)
            ),
            green,
        ),
        FileStatus::Skipped(_) => ("já transcrito".to_string(), blue),
        FileStatus::Failed(error) => (format!("erro: {error}"), red),
        FileStatus::Cancelled => ("cancelado".to_string(), gray),
    }
}
