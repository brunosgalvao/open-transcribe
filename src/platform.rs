use std::path::{Path, PathBuf};
use std::process::Command;

pub fn gpu_backend() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA"
    } else if cfg!(feature = "metal") {
        "Metal"
    } else if cfg!(feature = "vulkan") {
        "Vulkan"
    } else {
        "CPU"
    }
}

pub fn gpu_enabled() -> bool {
    cfg!(feature = "cuda") || cfg!(feature = "metal") || cfg!(feature = "vulkan")
}

pub fn open_path(path: &Path) {
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(path).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(path).spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("explorer").arg(path).spawn();
    }
}

pub fn reveal_path(path: &Path) {
    #[cfg(target_os = "linux")]
    {
        if let Some(parent) = path.parent() {
            open_path(parent);
        }
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg("-R").arg(path).spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let argument = format!("/select,{}", path.display());
        let _ = Command::new("explorer").arg(argument).spawn();
    }
}

pub fn open_url(url: &str) {
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(url).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(url).spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn();
    }
}

pub fn ffmpeg_program() -> PathBuf {
    if let Some(path) = std::env::var_os("OPEN_TRANSCRIBE_FFMPEG") {
        return PathBuf::from(path);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        for name in ["ffmpeg", "ffmpeg.exe"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from("ffmpeg")
}
