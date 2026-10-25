use anyhow::{Context, Result, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use serde::Deserialize;
use std::fs;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Request {
    texts: Vec<String>,
    target: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    pivot: Option<String>,
    #[serde(default)]
    threads: i32,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut model_path: Option<PathBuf> = None;
    let mut input_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = args.next().map(PathBuf::from),
            "--input" => input_path = args.next().map(PathBuf::from),
            other => bail!("argumento desconhecido: {other}"),
        }
    }

    let model_path = model_path.context("--model ausente")?;
    let input_path = input_path.context("--input ausente")?;
    let raw = if input_path == PathBuf::from("-") {
        use std::io::Read;
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .context("não foi possível ler a entrada padrão")?;
        text
    } else {
        fs::read_to_string(&input_path)
            .with_context(|| format!("não foi possível ler {}", input_path.display()))?
    };
    let request: Request = serde_json::from_str(&raw).context("JSON de entrada inválido")?;
    if request.texts.is_empty() {
        println!("{{\"translations\":[]}}");
        return Ok(());
    }

    let (target_name, target_instruction) = match request.target.as_str() {
        "pt" => (
            "português brasileiro",
            "Use português brasileiro (pt-BR), com vocabulário e gramática do Brasil. \
             Não use português de Portugal.",
        ),
        "es" => ("espanhol", "Use espanhol neutro."),
        "en" => ("inglês", "Use natural English."),
        other => bail!("idioma de destino não suportado: {other}"),
    };
    let source_name = request.source.as_deref().and_then(language_name);
    let pivot_name = request.pivot.as_deref().and_then(language_name);

    let threads = if request.threads > 0 {
        request.threads.clamp(1, 32)
    } else {
        4
    };

    let mut backend = LlamaBackend::init().context("falha ao iniciar o llama.cpp")?;
    backend.void_logs();
    let model = LlamaModel::load_from_file(
        &backend,
        &model_path,
        &LlamaModelParams::default().with_n_gpu_layers(99),
    )
    .with_context(|| format!("falha ao carregar {}", model_path.display()))?;
    let template = model
        .chat_template(None)
        .context("o modelo não tem chat template")?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(4096))
        .with_n_batch(1024)
        .with_n_threads(threads)
        .with_n_threads_batch(threads);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("falha ao criar o contexto")?;

    let mut translations = Vec::with_capacity(request.texts.len());
    let total = request.texts.len();
    for (index, text) in request.texts.iter().enumerate() {
        if text.trim().is_empty() {
            translations.push(String::new());
        } else {
        let mut parts = Vec::new();
        for piece in split_for_translation(text) {
            let translated = if let Some(pivot_name) = pivot_name {
                let english = translate_piece(
                    &model,
                    &mut ctx,
                    &template,
                    &piece,
                    pivot_name,
                    "Use natural English.",
                    source_name,
                )?;
                translate_piece(
                    &model,
                    &mut ctx,
                    &template,
                    &english,
                    target_name,
                    target_instruction,
                    Some(pivot_name),
                )?
            } else {
                translate_piece(
                    &model,
                    &mut ctx,
                    &template,
                    &piece,
                    target_name,
                    target_instruction,
                    source_name,
                )?
            };
            parts.push(translated);
        }
        translations.push(parts.join(" "));
        }
        eprintln!("progress {} {}", index + 1, total);
        let _ = std::io::stderr().flush();
    }

    println!("{}", serde_json::json!({ "translations": translations }));
    Ok(())
}

fn language_name(code: &str) -> Option<&'static str> {
    match code {
        "pt" => Some("português"),
        "es" => Some("espanhol"),
        "en" => Some("inglês"),
        "fr" => Some("francês"),
        "it" => Some("italiano"),
        "de" => Some("alemão"),
        "ja" => Some("japonês"),
        "ko" => Some("coreano"),
        "zh" => Some("chinês"),
        "ru" => Some("russo"),
        _ => None,
    }
}

fn translate_piece(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    template: &LlamaChatTemplate,
    text: &str,
    target_name: &str,
    target_instruction: &str,
    source_name: Option<&'static str>,
) -> Result<String> {
    let source_hint = source_name
        .map(|name| format!("O texto está em {name}. "))
        .unwrap_or_default();
    let user = format!(
        "{source_hint}Traduza o texto abaixo para {target_name}.\n\
         Regras: traduza fielmente; preserve nomes próprios, números e siglas; não resuma; \
         não explique; não acrescente comentários; responda somente com a tradução em \
         {target_name}. {target_instruction}\n\nTexto:\n{text}\n\nTradução em {target_name}:"
    );
    let messages = vec![
        LlamaChatMessage::new(
            "system".to_string(),
            format!(
                "Você é um tradutor profissional. {}Traduza sempre para {target_name} e \
                 responda apenas com a tradução em {target_name}. {target_instruction}",
                source_name
                    .map(|name| format!("O texto de entrada está em {name}. "))
                    .unwrap_or_default()
            ),
        )?,
        LlamaChatMessage::new("user".to_string(), user)?,
    ];
    let prompt = model
        .apply_chat_template(template, &messages, true)
        .context("falha ao aplicar o chat template")?;
    let answer = generate(model, ctx, &prompt)?;
    Ok(clean_translation(&answer, text))
}

fn generate(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    prompt: &str,
) -> Result<String> {
    ctx.clear_kv_cache();

    let tokens = model.vocab().tokenize(prompt.as_bytes(), true, true);
    if tokens.is_empty() {
        bail!("prompt vazio");
    }
    let n_ctx = ctx.n_ctx() as usize;
    if tokens.len() + 64 >= n_ctx {
        bail!("texto grande demais para o contexto ({} tokens)", tokens.len());
    }
    let max_new_tokens = (n_ctx - tokens.len() - 32).min(2048);

    const BATCH: usize = 1024;
    let mut batch = LlamaBatch::new(BATCH, 1);
    let mut position = 0usize;

    for chunk in tokens.chunks(BATCH) {
        batch.clear();
        for (offset, token) in chunk.iter().enumerate() {
            let absolute = position + offset;
            let is_last = absolute + 1 == tokens.len();
            batch
                .add(*token, absolute as i32, &[0], is_last)
                .context("falha ao adicionar token ao lote")?;
        }
        ctx.decode(&mut batch).context("falha no decode do prompt")?;
        position += chunk.len();
    }

    let mut sampler = LlamaSampler::greedy();
    let mut output: Vec<u8> = Vec::new();
    let mut current = tokens.len() as i32;

    for _ in 0..max_new_tokens {
        let logits_index = batch.n_tokens() as i32 - 1;
        let token = sampler.sample(ctx, logits_index);
        sampler.accept(token);

        if model.vocab().is_eog(token) {
            break;
        }

        output.extend(model.vocab().token_to_piece(token, false, None));

        batch.clear();
        batch
            .add(token, current, &[0], true)
            .context("falha ao adicionar token gerado")?;
        ctx.decode(&mut batch).context("falha no decode da resposta")?;
        current += 1;
    }

    Ok(String::from_utf8_lossy(&output).to_string())
}

const MAX_PIECE_CHARS: usize = 260;

fn split_for_translation(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut raw = Vec::new();
    let mut current = String::new();
    for (index, ch) in chars.iter().enumerate() {
        current.push(*ch);
        if !matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
            continue;
        }
        let next = chars.get(index + 1);
        let strong = matches!(ch, '!' | '?' | '。' | '！' | '？');
        if !strong && next.is_some_and(|c| !c.is_whitespace()) {
            continue;
        }
        let previous = index.checked_sub(1).and_then(|i| chars.get(i));
        let decimal = *ch == '.'
            && previous.is_some_and(|c| c.is_ascii_digit())
            && next.is_some_and(|c| c.is_ascii_digit());
        let ellipsis = *ch == '.'
            && (previous == Some(&'.') || next == Some(&'.'));
        if decimal || ellipsis {
            continue;
        }
        let piece = current.trim();
        if !piece.is_empty() {
            raw.push(piece.to_string());
        }
        current.clear();
    }
    let piece = current.trim();
    if !piece.is_empty() {
        raw.push(piece.to_string());
    }

    let mut pieces = Vec::new();
    for piece in raw {
        pieces.extend(split_long_piece(&piece, MAX_PIECE_CHARS));
    }
    if pieces.is_empty() {
        pieces.push(text.trim().to_string());
    }
    pieces
}

fn split_long_piece(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }

    let mut raw = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, ',' | ';' | ':') {
            raw.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        raw.push(current);
    }

    let mut pieces = Vec::new();
    let mut buffer = String::new();
    for part in raw {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if !buffer.is_empty()
            && buffer.chars().count() + 1 + part.chars().count() > max_chars
        {
            pieces.push(buffer.trim().to_string());
            buffer.clear();
        }
        if part.chars().count() > max_chars {
            if !buffer.is_empty() {
                pieces.push(std::mem::take(&mut buffer));
            }
            pieces.extend(split_by_words(part, max_chars));
        } else {
            if !buffer.is_empty() {
                buffer.push(' ');
            }
            buffer.push_str(part);
        }
    }
    if !buffer.trim().is_empty() {
        pieces.push(buffer.trim().to_string());
    }
    pieces
}

fn split_by_words(text: &str, max_chars: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty()
            && current.chars().count() + 1 + word.chars().count() > max_chars
        {
            pieces.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        pieces.push(current.trim().to_string());
    }
    pieces
}

fn clean_translation(output: &str, original: &str) -> String {
    let mut text = output.trim().to_string();
    for prefix in [
        "Tradução:",
        "Tradução para o português:",
        "Tradução para o espanhol:",
        "Tradução em português:",
        "Tradução em espanhol:",
        "Aqui está a tradução:",
        "Aqui está a tradução em português:",
        "Aqui está a tradução em espanhol:",
        "Segue a tradução:",
        "Claro, aqui está a tradução:",
        "Traducción:",
        "Translation:",
        "Here is the translation:",
        "Resposta:",
        "Resposta traduzida:",
    ] {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest.trim().to_string();
        }
    }
    let meta = [
        "aqui está a tradução",
        "segue a tradução",
        "tradução em português",
        "tradução em espanhol",
        "here is the translation",
    ];
    let normalized = text
        .trim()
        .trim_end_matches([':', '.', '!', '?'])
        .trim()
        .to_lowercase();
    if meta.iter().any(|phrase| normalized == *phrase) {
        return original.to_string();
    }
    if text.starts_with("```") {
        text = text
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string();
    }
    if text.len() >= 2 && text.starts_with('"') && text.ends_with('"') {
        text = text[1..text.len() - 1].trim().to_string();
    }
    if text.is_empty() {
        original.to_string()
    } else {
        text
    }
}
