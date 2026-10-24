use crate::{config, platform};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const REPO: &str = "brunosgalvao/open-transcribe";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    assets: Vec<Asset>,
}

#[derive(Debug, Clone, Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: u64,
}

#[derive(Debug, Clone)]
pub struct AssetInfo {
    pub name: String,
    pub url: String,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub version: String,
    pub html_url: String,
    pub notes: String,
    pub asset: Option<AssetInfo>,
}

pub fn check() -> Result<Option<UpdateInfo>> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(format!("open-transcribe/{VERSION}"))
        .timeout(Duration::from_secs(15))
        .build()
        .context("não foi possível criar o cliente HTTP")?;
    let release: Release = client
        .get(format!(
            "https://api.github.com/repos/{REPO}/releases/latest"
        ))
        .send()
        .context("falha ao consultar o GitHub")?
        .error_for_status()
        .context("o GitHub não retornou uma release válida")?
        .json()
        .context("resposta inválida do GitHub")?;

    let latest = release.tag_name.trim_start_matches(['v', 'V']);
    if !is_newer(latest, VERSION) {
        return Ok(None);
    }

    Ok(Some(UpdateInfo {
        version: release.tag_name,
        html_url: release.html_url,
        notes: release.body,
        asset: platform_asset(&release.assets).map(|asset| AssetInfo {
            name: asset.name.clone(),
            url: asset.browser_download_url.clone(),
            size: asset.size,
        }),
    }))
}

fn platform_asset(assets: &[Asset]) -> Option<&Asset> {
    #[cfg(target_os = "linux")]
    {
        let preferred = if cfg!(feature = "cuda") {
            assets
                .iter()
                .find(|asset| asset.name.ends_with(".AppImage") && asset.name.contains("CUDA"))
        } else if cfg!(feature = "vulkan") {
            assets
                .iter()
                .find(|asset| asset.name.ends_with(".AppImage") && asset.name.contains("Vulkan"))
                .or_else(|| {
                    assets
                        .iter()
                        .find(|asset| asset.name.ends_with(".AppImage") && asset.name.contains("CPU"))
                })
        } else {
            assets
                .iter()
                .find(|asset| asset.name.ends_with(".AppImage") && asset.name.contains("CPU"))
                .or_else(|| {
                    assets
                        .iter()
                        .find(|asset| asset.name.ends_with(".AppImage") && !asset.name.contains("CUDA"))
                })
        };
        preferred
    }
    #[cfg(target_os = "windows")]
    {
        if cfg!(feature = "cuda") {
            assets.iter().find(|asset| {
                asset.name.ends_with(".zip")
                    && asset.name.contains("windows")
                    && asset.name.contains("CUDA")
            })
        } else if cfg!(feature = "vulkan") {
            assets
                .iter()
                .find(|asset| {
                    asset.name.ends_with(".zip")
                        && asset.name.contains("windows")
                        && asset.name.contains("VULKAN")
                })
                .or_else(|| {
                    assets.iter().find(|asset| {
                        asset.name.ends_with(".zip")
                            && asset.name.contains("windows")
                            && asset.name.contains("CPU")
                    })
                })
        } else {
            assets
                .iter()
                .find(|asset| {
                    asset.name.ends_with(".zip")
                        && asset.name.contains("windows")
                        && asset.name.contains("CPU")
                })
                .or_else(|| {
                    assets
                        .iter()
                        .find(|asset| asset.name.ends_with(".zip") && asset.name.contains("windows"))
                })
        }
    }
    #[cfg(target_os = "macos")]
    {
        let arch = if std::env::consts::ARCH == "aarch64" {
            "arm64"
        } else {
            "x86_64"
        };
        assets
            .iter()
            .find(|asset| asset.name.ends_with(".dmg") && asset.name.contains(arch))
            .or_else(|| assets.iter().find(|asset| asset.name.ends_with(".dmg")))
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        assets.first()
    }
}

fn is_newer(latest: &str, current: &str) -> bool {
    let parse = |value: &str| {
        value
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|part| part.parse::<u64>().ok())
            .collect::<Vec<_>>()
    };
    let latest = parse(latest);
    let current = parse(current);
    for index in 0..latest.len().max(current.len()) {
        let new = latest.get(index).copied().unwrap_or(0);
        let old = current.get(index).copied().unwrap_or(0);
        if new != old {
            return new > old;
        }
    }
    false
}

pub fn download(asset: &AssetInfo, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
    let dir = config::home_dir().join("Downloads");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("não foi possível criar {}", dir.display()))?;
    let dest = dir.join(&asset.name);
    let mut response = reqwest::blocking::Client::builder()
        .user_agent(format!("open-transcribe/{VERSION}"))
        .timeout(Duration::from_secs(1800))
        .build()
        .context("não foi possível criar o cliente HTTP")?
        .get(&asset.url)
        .send()
        .context("falha ao baixar a atualização")?
        .error_for_status()
        .context("o download da atualização falhou")?;

    let total = response.content_length().unwrap_or(asset.size);
    let mut file = File::create(&dest)
        .with_context(|| format!("não foi possível criar {}", dest.display()))?;
    let mut buffer = [0u8; 64 * 1024];
    let mut received = 0u64;
    let mut last_report = 0u64;
    progress(0, total);
    loop {
        let read = response.read(&mut buffer).context("erro lendo o download")?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .context("erro gravando o download")?;
        received += read as u64;
        if received - last_report >= 512 * 1024 {
            progress(received, total);
            last_report = received;
        }
    }
    file.flush().context("erro finalizando o download")?;
    progress(received, total);
    Ok(dest)
}

pub fn install_appimage(downloaded: &Path) -> Result<PathBuf> {
    let raw = std::env::var_os("APPIMAGE")
        .map(PathBuf::from)
        .context("o aplicativo não está rodando como AppImage")?;
    let current = std::fs::canonicalize(&raw).unwrap_or(raw);
    let parent = current
        .parent()
        .context("caminho do AppImage inválido")?;
    let file_name = current
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("OpenTranscribe.AppImage");
    let temporary = parent.join(format!(".{file_name}.new"));

    if temporary.exists() {
        let _ = std::fs::remove_file(&temporary);
    }
    std::fs::copy(downloaded, &temporary).with_context(|| {
        format!(
            "não foi possível copiar a atualização para {}",
            temporary.display()
        )
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&temporary)?.permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&temporary, permissions)?;
    }
    std::fs::rename(&temporary, &current).with_context(|| {
        format!(
            "não foi possível substituir o AppImage em {}",
            current.display()
        )
    })?;
    Ok(current)
}

pub fn restart_app(path: &Path) -> Result<()> {
    std::process::Command::new(path)
        .spawn()
        .with_context(|| format!("não foi possível reiniciar {}", path.display()))?;
    std::process::exit(0);
}

pub fn open_downloaded(path: &Path) {
    #[cfg(target_os = "macos")]
    {
        platform::open_path(path);
    }
    #[cfg(target_os = "windows")]
    {
        platform::reveal_path(path);
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        platform::reveal_path(path);
    }
}

pub fn open_release_page(url: &str) {
    platform::open_url(url);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    #[test]
    fn picks_cuda_appimage_for_cuda_build() {
        let assets = vec![
            Asset {
                name: "OpenTranscribe-CPU-x86_64.AppImage".to_string(),
                browser_download_url: "https://example.com/cpu".to_string(),
                size: 1,
            },
            Asset {
                name: "OpenTranscribe-CUDA-x86_64.AppImage".to_string(),
                browser_download_url: "https://example.com/cuda".to_string(),
                size: 1,
            },
        ];
        assert_eq!(
            platform_asset(&assets).unwrap().browser_download_url,
            "https://example.com/cuda"
        );
    }

    #[cfg(all(target_os = "linux", not(feature = "cuda"), not(feature = "vulkan")))]
    #[test]
    fn picks_cpu_appimage_for_cpu_build() {
        let assets = vec![
            Asset {
                name: "OpenTranscribe-CUDA-x86_64.AppImage".to_string(),
                browser_download_url: "https://example.com/cuda".to_string(),
                size: 1,
            },
            Asset {
                name: "OpenTranscribe-CPU-x86_64.AppImage".to_string(),
                browser_download_url: "https://example.com/cpu".to_string(),
                size: 1,
            },
        ];
        assert_eq!(
            platform_asset(&assets).unwrap().browser_download_url,
            "https://example.com/cpu"
        );
    }

    #[test]
    fn compares_versions() {
        assert!(is_newer("0.2.0", "0.1.0"));
        assert!(is_newer("v1.0.0", "0.9.9"));
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("0.1.0", "0.2.0"));
        assert!(is_newer("1.2.10", "1.2.9"));
    }
}
