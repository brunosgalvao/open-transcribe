<p align="center">
  <img src="logo/logo.webp" width="180" alt="Open Transcribe">
</p>

<h1 align="center">Open Transcribe</h1>

<p align="center">
  Transcrição local de vídeos e aulas com Whisper, interface desktop e parágrafos organizados por IA local.
</p>

## Recursos

- Transcrição local com Whisper (GPU CUDA, Metal ou CPU).
- Visualização da transcrição em tempo real durante o processamento.
- Texto corrido com parágrafos separados por pausas, pontuação, conteúdo e um LLM local opcional.
- Tradução para inglês pelo Whisper ou para português (Brasil)/espanhol com Google Tradutor online ou IA local.
- Detecção de hardware com configuração recomendada para o computador.
- Download e remoção de modelos pela própria interface.
- Verificação de atualizações pelo GitHub, com atualização automática do AppImage no Linux.
- Saídas em texto com timecodes, texto corrido ou legendas SRT.

## Tradução

- **Inglês:** usa a tradução nativa do Whisper.
- **Português (Brasil) e espanhol:** por padrão usa o **Google Tradutor online**, que dá a melhor qualidade. Também é possível escolher **IA local (Qwen)** em Configurações.
- O português é sempre pt-BR.

A tradução online envia apenas os trechos da transcrição para o Google. Se quiser tudo offline, selecione **IA local (Qwen)**; nesse caso o Qwen 2.5 3B é o mínimo recomendado.

## Download

Os binários ficam na página de releases:

https://github.com/brunosgalvao/open-transcribe/releases

- **Linux (CPU):** `OpenTranscribe-CPU-x86_64.AppImage`
- **Linux (NVIDIA/CUDA):** `OpenTranscribe-CUDA-x86_64.AppImage`
- **Windows (CPU):** `OpenTranscribe-CPU-windows-x86_64.zip`
- **Windows (NVIDIA/CUDA):** `OpenTranscribe-CUDA-windows-x86_64.zip`
- **macOS:** `OpenTranscribe-macos-arm64.dmg` ou `OpenTranscribe-macos-x86_64.dmg`

No Linux e no Windows, use o pacote **CUDA** se tiver placa NVIDIA com driver instalado. Não é preciso instalar o CUDA Toolkit; as bibliotecas necessárias vão dentro do AppImage/zip.

O `ffmpeg` é necessário para extrair o áudio dos vídeos. Ele é incluído nos pacotes quando o build consegue empacotá-lo; caso contrário, instale o `ffmpeg` no sistema.

## Modelos

Os modelos não ficam no repositório. O aplicativo baixa, sob demanda:

- Modelos Whisper do projeto whisper.cpp.
- Modelo Qwen2.5 para separar parágrafos (opcional).
- Modelo VAD para remover silêncios (opcional).

Os arquivos ficam na pasta de dados do usuário e podem ser apagados pela interface.

## Compilar

Requisitos: Rust estável e `ffmpeg`.

### Linux

```bash
sudo apt install -y ffmpeg libgtk-3-dev libxkbcommon-dev libwayland-dev \
  libx11-dev libxcursor-dev libxrandr-dev libxi-dev libgl1-mesa-dev

# Com CUDA (requer CUDA Toolkit):
bash packaging/build-appimage.sh

# Sem CUDA:
bash packaging/build-appimage.sh --cpu
```

### Windows

```powershell
# CPU
pwsh packaging/build-windows.ps1 -Features cpu

# CUDA (requer CUDA Toolkit instalado)
pwsh packaging/build-windows-cuda.ps1
```

### macOS

```bash
brew install ffmpeg dylibbundler
bash packaging/build-macos.sh
```

O script gera o `.app` e o `.dmg` com Metal habilitado.

## Release automático

O workflow `.github/workflows/release.yml` gera os pacotes de Linux, Windows e macOS quando uma tag `v*` é enviada:

```bash
git tag v0.1.1
git push origin v0.1.0
```

Ele cria uma release **draft** no GitHub para revisão antes de publicar.

## Atualizações

O aplicativo consulta a release mais recente em `brunosgalvao/open-transcribe`. Se você fizer um fork, altere a constante `REPO` em `src/update.rs` e o campo `repository` em `Cargo.toml`.

No Linux, a atualização substitui o AppImage atual e reinicia o aplicativo. No Windows e macOS, o arquivo correto é baixado e aberto para instalação.

## Licença

Código sob a licença MIT. Veja [LICENSE](LICENSE).

Este projeto usa e/ou se inspira em:

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) e [whisper-rs](https://github.com/tazz4843/whisper-rs) — MIT
- [llama.cpp](https://github.com/ggml-org/llama.cpp) e [llama-cpp-2](https://github.com/utilityai/llama-cpp-2) — MIT
- [egui](https://github.com/emilk/egui) e [eframe](https://github.com/emilk/egui/tree/master/crates/eframe) — MIT/Apache-2.0
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) — Apache-2.0

Os modelos baixados mantêm as licenças dos respectivos projetos.
