use crate::config::{self, Config, MODELS, PARAGRAPH_MODELS};
use crate::platform;
use std::process::Command;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub total_vram: Option<u64>,
    pub free_vram: Option<u64>,
    pub backend: &'static str,
    pub usable: bool,
}

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub os: &'static str,
    pub arch: &'static str,
    pub cpu_threads: usize,
    pub ram_bytes: Option<u64>,
    pub gpu: Option<GpuInfo>,
    pub gpu_backend: &'static str,
    pub gpu_enabled: bool,
}

pub struct Recommendation {
    pub config: Config,
    pub notes: Vec<String>,
}

impl HardwareInfo {
    pub fn detect() -> Self {
        Self {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            cpu_threads: std::thread::available_parallelism()
                .map(|threads| threads.get())
                .unwrap_or(1),
            ram_bytes: total_ram_bytes(),
            gpu: detect_gpu(),
            gpu_backend: platform::gpu_backend(),
            gpu_enabled: platform::gpu_enabled(),
        }
    }

    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "{} {} • {} thread(s) de CPU",
            self.os, self.arch, self.cpu_threads
        ));
        lines.push(match self.ram_bytes {
            Some(bytes) => format!("RAM: {}", config::human_size(bytes)),
            None => "RAM: não detectada".to_string(),
        });
        if let Some(gpu) = &self.gpu {
            let memory = match (gpu.total_vram, gpu.free_vram) {
                (Some(total), Some(free)) => format!(
                    " — {} total, {} livres",
                    config::human_size(total),
                    config::human_size(free)
                ),
                (Some(total), None) => format!(" — {}", config::human_size(total)),
                _ => String::new(),
            };
            lines.push(format!("GPU: {} ({}){memory}", gpu.name, gpu.backend));
        } else {
            lines.push("GPU: não detectada".to_string());
        }
        lines.push(format!(
            "Aceleração deste build: {}",
            if self.gpu_enabled {
                self.gpu_backend
            } else {
                "CPU (sem CUDA/Metal/Vulkan)"
            }
        ));
        lines.join("\n")
    }

    pub fn recommendation(&self, base: &Config) -> Recommendation {
        let gpu_available = self.gpu_enabled
            && self
                .gpu
                .as_ref()
                .is_some_and(|gpu| gpu.usable);
        let gpu_memory = if self.gpu_backend == "Metal" {
            self.ram_bytes
        } else {
            self.gpu
                .as_ref()
                .and_then(|gpu| gpu.free_vram.or(gpu.total_vram))
        };

        let (model_id, use_gpu, reason) = if !gpu_available {
            (
                cpu_model(self.ram_bytes, self.cpu_threads),
                false,
                "Sem GPU utilizável neste build; configuração otimizada para CPU.",
            )
        } else {
            match self.gpu_backend {
                "Metal" => {
                    let ram = self.ram_bytes.unwrap_or(0);
                    if ram >= 32 * GIB {
                        (
                            "large-v3",
                            true,
                            "Apple Silicon com RAM abundante para o modelo mais preciso.",
                        )
                    } else if ram >= 16 * GIB {
                        (
                            "large-v3-turbo",
                            true,
                            "Apple Silicon: turbo é quase tão preciso e bem mais rápido.",
                        )
                    } else if ram >= 8 * GIB {
                        (
                            "small",
                            true,
                            "Apple Silicon com RAM moderada: modelo menor para manter a fluidez.",
                        )
                    } else {
                        (
                            "base",
                            true,
                            "Apple Silicon com pouca RAM: modelo leve.",
                        )
                    }
                }
                "CUDA" | "Vulkan" => match gpu_memory {
                    Some(memory) if memory >= 7 * GIB => (
                        "large-v3",
                        true,
                        "GPU com VRAM suficiente para o modelo mais preciso.",
                    ),
                    Some(memory) if memory >= 4 * GIB => (
                        "large-v3-turbo",
                        true,
                        "GPU com boa VRAM: turbo é quase tão preciso e bem mais rápido.",
                    ),
                    Some(memory) if memory >= 3 * GIB => (
                        "large-v3-turbo-q5_0",
                        true,
                        "VRAM moderada: turbo quantizado reduz o uso de memória.",
                    ),
                    Some(memory) if memory >= 2 * GIB => (
                        "small",
                        true,
                        "VRAM apertada: modelo menor para caber na GPU.",
                    ),
                    _ => (
                        cpu_model(self.ram_bytes, self.cpu_threads),
                        false,
                        "VRAM não detectada; configuração otimizada para CPU.",
                    ),
                },
                _ => (
                    cpu_model(self.ram_bytes, self.cpu_threads),
                    false,
                    "Sem GPU utilizável neste build; configuração otimizada para CPU.",
                ),
            }
        };

        let info = MODELS
            .iter()
            .find(|info| info.id == model_id)
            .unwrap_or(&MODELS[0]);

        let paragraph_model = if self.ram_bytes.unwrap_or(0) >= 12 * GIB {
            &PARAGRAPH_MODELS[0]
        } else {
            PARAGRAPH_MODELS.get(1).unwrap_or(&PARAGRAPH_MODELS[0])
        };

        let threads = self.cpu_threads.clamp(1, 16);
        let mut config = base.clone();
        config.model_path = config::model_path(info.file).display().to_string();
        config.use_gpu = use_gpu;
        config.threads = threads as i32;
        config.beam_size = 1;
        config.flash_attn = use_gpu;
        config.paragraph_model_path = config::paragraph_model_path(paragraph_model.file)
            .display()
            .to_string();

        let mut notes = vec![
            format!("Modelo: {} — {reason}", info.name),
            format!(
                "Processamento: {}.",
                if use_gpu {
                    format!("GPU ({})", self.gpu_backend)
                } else {
                    "CPU".to_string()
                }
            ),
            format!("Threads de CPU: {threads}."),
            format!("Modelo de parágrafos: {}.", paragraph_model.name),
        ];
        if use_gpu {
            notes.push("Flash attention ativado.".to_string());
        }
        if !config::model_path(info.file).is_file() {
            notes.push(format!("O modelo {} ainda não foi baixado.", info.name));
        }

        Recommendation { config, notes }
    }
}

fn cpu_model(ram_bytes: Option<u64>, cpu_threads: usize) -> &'static str {
    let ram = ram_bytes.unwrap_or(0);
    if ram == 0 {
        return if cpu_threads >= 8 { "small" } else { "base" };
    }
    if ram >= 16 * GIB && cpu_threads >= 8 {
        "medium"
    } else if ram >= 8 * GIB {
        "small"
    } else if ram >= 4 * GIB {
        "base"
    } else {
        "tiny"
    }
}

fn detect_gpu() -> Option<GpuInfo> {
    if let Some(gpu) = nvidia_gpu() {
        return Some(gpu);
    }
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Some(GpuInfo {
            name: "Apple Silicon".to_string(),
            total_vram: None,
            free_vram: None,
            backend: "Metal",
            usable: cfg!(feature = "metal"),
        });
    }
    #[cfg(target_os = "linux")]
    if let Some(gpu) = linux_drm_gpu() {
        return Some(gpu);
    }
    None
}

fn nvidia_gpu() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?.trim();
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }
    Some(GpuInfo {
        name: parts[0].to_string(),
        total_vram: parse_mib(parts[1]),
        free_vram: parse_mib(parts[2]),
        backend: "CUDA",
        usable: cfg!(any(feature = "cuda", feature = "vulkan")),
    })
}

fn parse_mib(value: &str) -> Option<u64> {
    value
        .parse::<f64>()
        .ok()
        .map(|mib| (mib * 1024.0 * 1024.0) as u64)
}

#[cfg(target_os = "linux")]
fn linux_drm_gpu() -> Option<GpuInfo> {
    for index in 0..4 {
        let base = format!("/sys/class/drm/card{index}/device");
        let Ok(vendor) = std::fs::read_to_string(format!("{base}/vendor")) else {
            continue;
        };
        let vendor = vendor.trim();
        let (name, backend) = match vendor {
            "0x10de" => continue,
            "0x1002" => ("AMD GPU", "Vulkan/ROCm"),
            "0x8086" => ("Intel GPU", "Vulkan"),
            _ => continue,
        };
        let total_vram = std::fs::read_to_string(format!("{base}/mem_info_vram_total"))
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok());
        return Some(GpuInfo {
            name: name.to_string(),
            total_vram,
            free_vram: None,
            backend,
            usable: cfg!(feature = "vulkan"),
        });
    }
    None
}

#[cfg(target_os = "linux")]
fn total_ram_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<u64>().ok())?;
            return Some(kib * 1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn total_ram_bytes() -> Option<u64> {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .ok()
}

#[cfg(target_os = "windows")]
fn total_ram_bytes() -> Option<u64> {
    let output = Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        .output()
        .ok()?;
    if let Ok(value) = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
    {
        return Some(value);
    }

    let output = Command::new("wmic")
        .args(["computersystem", "get", "TotalPhysicalMemory", "/value"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines().find_map(|line| {
        line.strip_prefix("TotalPhysicalMemory=")
            .and_then(|value| value.trim().parse::<u64>().ok())
    })
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn total_ram_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hardware(
        gpu: Option<GpuInfo>,
        ram: u64,
        gpu_backend: &'static str,
        gpu_enabled: bool,
    ) -> HardwareInfo {
        HardwareInfo {
            os: "linux",
            arch: "x86_64",
            cpu_threads: 16,
            ram_bytes: Some(ram),
            gpu,
            gpu_backend,
            gpu_enabled,
        }
    }

    fn nvidia(free: u64) -> GpuInfo {
        GpuInfo {
            name: "NVIDIA Teste".to_string(),
            total_vram: Some(free),
            free_vram: Some(free),
            backend: "CUDA",
            usable: true,
        }
    }

    fn apple() -> GpuInfo {
        GpuInfo {
            name: "Apple Silicon".to_string(),
            total_vram: None,
            free_vram: None,
            backend: "Metal",
            usable: true,
        }
    }

    #[test]
    fn recommends_large_model_for_large_vram() {
        let recommendation = hardware(Some(nvidia(12 * GIB)), 32 * GIB, "CUDA", true)
            .recommendation(&Config::default());
        assert!(recommendation.config.use_gpu);
        assert_eq!(
            recommendation.config.model_path,
            config::model_path("ggml-large-v3.bin")
                .display()
                .to_string()
        );
        assert_eq!(
            recommendation.config.paragraph_model_path,
            config::paragraph_model_path("qwen2.5-1.5b-instruct-q4_k_m.gguf")
                .display()
                .to_string()
        );
    }

    #[test]
    fn falls_back_to_cpu_without_gpu_build() {
        let recommendation = hardware(Some(nvidia(12 * GIB)), 16 * GIB, "CPU", false)
            .recommendation(&Config::default());
        assert!(!recommendation.config.use_gpu);
        assert_eq!(
            recommendation.config.model_path,
            config::model_path("ggml-medium.bin")
                .display()
                .to_string()
        );
    }

    #[test]
    fn recommends_turbo_for_apple_silicon() {
        let recommendation = hardware(Some(apple()), 16 * GIB, "Metal", true)
            .recommendation(&Config::default());
        assert!(recommendation.config.use_gpu);
        assert_eq!(
            recommendation.config.model_path,
            config::model_path("ggml-large-v3-turbo.bin")
                .display()
                .to_string()
        );
    }

    #[test]
    fn uses_small_paragraph_model_with_little_ram() {
        let recommendation = hardware(None, 8 * GIB, "CPU", false)
            .recommendation(&Config::default());
        assert_eq!(
            recommendation.config.paragraph_model_path,
            config::paragraph_model_path("qwen2.5-0.5b-instruct-q4_k_m.gguf")
                .display()
                .to_string()
        );
    }
}
