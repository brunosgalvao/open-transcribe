param(
    [ValidateSet("cpu", "cuda", "vulkan")]
    [string]$Features = "cpu"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($Features -eq "cuda") {
    if ($env:CUDA_PATH) {
        if (-not $env:CUDAToolkit_ROOT) { $env:CUDAToolkit_ROOT = $env:CUDA_PATH }
        if (-not $env:CMAKE_CUDA_COMPILER) {
            $env:CMAKE_CUDA_COMPILER = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        }
    }
    $mdFlag = "-Xcompiler /MD"
    if ($env:NVCC_PREPEND_FLAGS) {
        $env:NVCC_PREPEND_FLAGS = "$mdFlag $env:NVCC_PREPEND_FLAGS"
    } else {
        $env:NVCC_PREPEND_FLAGS = $mdFlag
    }
    if (-not $env:CMAKE_MSVC_RUNTIME_LIBRARY) {
        $env:CMAKE_MSVC_RUNTIME_LIBRARY = "MultiThreadedDLL"
    }
}

Write-Host "[1/4] Compilando (release, $Features)..."
if ($Features -eq "cpu") {
    cargo build --release --no-default-features
} else {
    cargo build --release --no-default-features --features $Features
}
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$Name = "OpenTranscribe"
$Variant = switch ($Features) {
    "cuda" { "CUDA" }
    "vulkan" { "VULKAN" }
    default { "CPU" }
}
$Stage = Join-Path $Root "dist\windows-$Variant"
$Zip = Join-Path $Root "dist\$Name-$Variant-windows-x86_64.zip"

Write-Host "[2/4] Montando pasta..."
Remove-Item -Recurse -Force $Stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Stage | Out-Null
Copy-Item "target\release\open-transcribe.exe" (Join-Path $Stage "$Name.exe")
Copy-Item "target\release\llm-paragraph.exe" (Join-Path $Stage "llm-paragraph.exe")
Copy-Item "target\release\llm-translate.exe" (Join-Path $Stage "llm-translate.exe")
Copy-Item "assets\icon.png" (Join-Path $Stage "icon.png")

$ffmpeg = $env:FFMPEG_BIN
if (-not $ffmpeg) {
    $command = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($command) { $ffmpeg = $command.Source }
}

function Resolve-RealFfmpeg([string]$Path) {
    if (-not $Path -or -not (Test-Path $Path)) { return $null }
    if ((Get-Item $Path).Length -ge 1MB) { return (Get-Item $Path).FullName }

    $roots = @()
    if ($env:ChocolateyInstall) {
        $roots += (Join-Path $env:ChocolateyInstall "lib\ffmpeg\tools")
    }
    $roots += "C:\ProgramData\chocolatey\lib\ffmpeg\tools"
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $real = Get-ChildItem $root -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue |
            Where-Object { $_.Length -ge 1MB } |
            Select-Object -First 1
        if ($real) { return $real.FullName }
    }
    return $null
}

$ffmpeg = Resolve-RealFfmpeg $ffmpeg
if (-not $ffmpeg) {
    throw "ffmpeg real não encontrado. Instale o ffmpeg completo ou defina FFMPEG_BIN."
}
Copy-Item $ffmpeg (Join-Path $Stage "ffmpeg.exe")

if ($Features -eq "cuda" -and $env:CUDA_PATH) {
    Write-Host "[3/4] Copiando DLLs do CUDA..."
    foreach ($pattern in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "nvJitLink*.dll")) {
        Get-ChildItem (Join-Path $env:CUDA_PATH "bin") -Filter $pattern -ErrorAction SilentlyContinue |
            Copy-Item -Destination $Stage
    }
    if (-not (Get-ChildItem $Stage -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue)) {
        throw "DLL cudart64_* não encontrada em $env:CUDA_PATH\bin"
    }
} else {
    Write-Host "[3/4] Sem DLLs CUDA."
}

Write-Host "[4/4] Gerando zip..."
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Zip -Force
Write-Host "Pronto: $Zip"
