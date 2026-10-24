#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export CUDAToolkit_ROOT="$CUDA_PATH"
export CMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc"
export LIBCLANG_PATH="${LIBCLANG_PATH:-$HOME/.local/lib/clang}"

NAME="OpenTranscribe"
APPDIR="$ROOT/dist/$NAME.AppDir"
TOOL="$ROOT/.tools/appimagetool-x86_64.AppImage"
FEATURES="cuda"
VARIANT="CUDA"

if [[ "${1:-}" == "--cpu" ]]; then
    FEATURES=""
    VARIANT="CPU"
    shift
fi

OUT="$ROOT/dist/$NAME-$VARIANT-x86_64.AppImage"

echo "[1/5] Compilando (release)..."
if [[ -n "$FEATURES" ]]; then
    cargo build --release --features "$FEATURES"
else
    cargo build --release --no-default-features
fi

echo "[2/5] Montando AppDir..."
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/lib" \
    "$APPDIR/usr/share/icons/hicolor/512x512/apps" \
    "$APPDIR/usr/share/applications"

install -m755 target/release/open-transcribe "$APPDIR/usr/bin/open-transcribe"
install -m755 target/release/llm-paragraph "$APPDIR/usr/bin/llm-paragraph"
if [[ -n "${FFMPEG_BIN:-}" && -x "$FFMPEG_BIN" ]]; then
    install -m755 "$FFMPEG_BIN" "$APPDIR/usr/bin/ffmpeg"
fi
install -m644 assets/icon.png "$APPDIR/usr/share/icons/hicolor/512x512/apps/open-transcribe.png"
install -m644 assets/icon.png "$APPDIR/open-transcribe.png"
install -m644 packaging/open-transcribe.desktop "$APPDIR/open-transcribe.desktop"
install -m644 packaging/open-transcribe.desktop "$APPDIR/usr/share/applications/open-transcribe.desktop"
install -m755 packaging/AppRun "$APPDIR/AppRun"

if [[ -n "$FEATURES" ]]; then
    echo "[3/5] Incluindo bibliotecas CUDA (runtime + cuBLAS)..."
    for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13 libnvJitLink.so.13; do
        src="$CUDA_PATH/lib64/$lib"
        if [[ ! -e "$src" ]]; then
            echo "ERRO: biblioteca não encontrada: $src" >&2
            exit 1
        fi
        install -m644 "$(readlink -f "$src")" "$APPDIR/usr/lib/$lib"
    done
else
    echo "[3/5] Build CPU: sem bibliotecas CUDA."
fi

if [[ ! -x "$TOOL" ]]; then
    echo "Baixando appimagetool..."
    mkdir -p "$(dirname "$TOOL")"
    curl -fL --retry 3 -o "$TOOL" \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x "$TOOL"
fi

echo "[4/5] Gerando AppImage..."
mkdir -p "$ROOT/dist"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" "$APPDIR" "$OUT"
chmod +x "$OUT"

echo "[5/5] Pronto: $OUT ($(du -h "$OUT" | cut -f1))"
