#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.cargo/bin:$PATH"
NAME="OpenTranscribe"
APP="$ROOT/dist/$NAME.app"
DMG="$ROOT/dist/$NAME-macos-$(uname -m).dmg"

echo "[1/5] Compilando (release, Metal)..."
cargo build --release --no-default-features --features metal

echo "[2/5] Montando .app..."
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

install -m755 target/release/open-transcribe "$APP/Contents/MacOS/$NAME"
install -m755 target/release/llm-paragraph "$APP/Contents/MacOS/llm-paragraph"
if [[ -n "${FFMPEG_BIN:-}" && -x "$FFMPEG_BIN" ]]; then
    install -m755 "$FFMPEG_BIN" "$APP/Contents/MacOS/ffmpeg"
    if command -v dylibbundler >/dev/null 2>&1; then
        dylibbundler -od -b \
            -x "$APP/Contents/MacOS/ffmpeg" \
            -d "$APP/Contents/Frameworks" \
            -p "@executable_path/../Frameworks"
    else
        echo "Aviso: dylibbundler ausente; ffmpeg pode depender do Homebrew."
    fi
fi

ICONSET="$ROOT/dist/$NAME.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
    sips -z "$size" "$size" assets/icon.png --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z "$((size * 2))" "$((size * 2))" assets/icon.png --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/$NAME.icns"
rm -rf "$ICONSET"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Open Transcribe</string>
    <key>CFBundleDisplayName</key>
    <string>Open Transcribe</string>
    <key>CFBundleIdentifier</key>
    <string>com.opentranscribe.app</string>
    <key>CFBundleExecutable</key>
    <string>$NAME</string>
    <key>CFBundleIconFile</key>
    <string>$NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

echo "[3/5] Assinando..."
if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
    codesign --force --deep --options runtime --timestamp --sign "$CODESIGN_IDENTITY" "$APP"
else
    codesign --force --deep --sign - "$APP" || true
fi

echo "[4/5] Gerando .dmg..."
STAGE="$ROOT/dist/dmg"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "Open Transcribe" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
rm -rf "$STAGE"

if [[ -n "${APPLE_ID:-}" && -n "${APPLE_TEAM_ID:-}" && -n "${APPLE_APP_PASSWORD:-}" ]]; then
    echo "[5/5] Notarizando..."
    xcrun notarytool submit "$DMG" \
        --apple-id "$APPLE_ID" \
        --team-id "$APPLE_TEAM_ID" \
        --password "$APPLE_APP_PASSWORD" \
        --wait
    xcrun stapler staple "$DMG"
else
    echo "[5/5] Sem credenciais Apple; .dmg não notarizado."
fi

echo "Pronto: $APP"
echo "Pronto: $DMG"
