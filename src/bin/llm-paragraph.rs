use anyhow::{Context, Result, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::num::NonZeroU32;
use std::path::PathBuf;

const WINDOW: usize = 30;
const OVERLAP: usize = 4;
const MAX_NEW_TOKENS: usize = 96;

#[derive(Deserialize)]
struct Request {
    sentences: Vec<String>,
    #[serde(default)]
    gaps: Vec<f64>,
    #[serde(default)]
    threads: i32,
    #[serde(default = "default_pause_ms")]
    pause_ms: i32,
}

fn default_pause_ms() -> i32 {
    800
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
    if request.sentences.is_empty() {
        println!("{{\"breaks\":[]}}");
        return Ok(());
    }

    let threads = if request.threads > 0 {
        request.threads.clamp(1, 32)
    } else {
        4
    };
    let pause = request.pause_ms.max(0) as f64 / 1000.0;

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

    let mut breaks = BTreeSet::new();
    let mut start = 0usize;
    let mut accepted_until = 0usize;

    while start < request.sentences.len() {
        let end = (start + WINDOW).min(request.sentences.len());
        let mut lines = String::new();
        for (index, sentence) in request.sentences[start..end].iter().enumerate() {
            let gap = request.gaps.get(start + index).copied().unwrap_or(0.0);
            let pause_marker = if gap >= pause {
                format!("[pausa {gap:.1}s] ")
            } else {
                String::new()
            };
            lines.push_str(&format!(
                "{}. {pause_marker}{sentence}\n",
                index + 1
            ));
        }

        let user = format!(
            r#"Transcrição em português. Cada linha é uma frase numerada; [pausa Xs] indica silêncio antes da frase.
Agrupe as frases em parágrafos sobre o mesmo assunto. Comece um novo parágrafo quando o assunto mudar de verdade. Não use um tamanho fixo e não crie um parágrafo por frase.
Responda apenas com intervalos de frases, cobrindo todas as frases, sem sobrepor. Exemplo de formato: 1-5, 6-12, 13-20.

Exemplo 1:
1. Oi, tudo bem?
2. Hoje eu quero falar sobre a minha relação com o dinheiro.
3. Ela é uma espada e um escudo.
4. A espada é a carteira de longo prazo.
5. O escudo é a reserva de emergência.
6. Agora eu quero mudar de assunto e falar do meu tumor.
7. Em 2023 eu descobri um neurinoma do acústico.
8. Foi um período difícil.
Resposta: 1-5, 6-8

Exemplo 2:
1. Eu fui ao médico.
2. Ele pediu exames.
3. O resultado saiu na semana seguinte.
4. Eu fiquei assustado.
5. No trabalho, ninguém sabia.
6. Eu contei para o meu chefe.
7. Ele foi compreensivo.
8. Depois disso, eu mudei a minha rotina.
9. Hoje eu cuido melhor da saúde.
Resposta: 1-4, 5-9

Agora a transcrição real. Use as pausas apenas como apoio. Não omita nem repita frases.
Frases:
{lines}
Resposta:"#
        );
        let messages = vec![
            LlamaChatMessage::new(
                "system".to_string(),
                "Você é um editor de transcrições. Sua tarefa é apenas indicar \
                 onde começam novos parágrafos. Não reescreva, não resuma e não \
                 explique."
                    .to_string(),
            )?,
            LlamaChatMessage::new("user".to_string(), user)?,
        ];
        let prompt = model
            .apply_chat_template(&template, &messages, true)
            .context("falha ao aplicar o chat template")?;

        let answer = generate(&model, &mut ctx, &prompt)?;
        let starts = parse_starts(&answer);
        if starts.len() * 2 < (end - start).max(2) {
            for number in starts {
                if number == 0 {
                    continue;
                }
                let global = start + number - 1;
                if global > 0 && global >= accepted_until && global < end {
                    breaks.insert(global);
                }
            }
        }

        accepted_until = end;
        if end == request.sentences.len() {
            break;
        }
        start = end.saturating_sub(OVERLAP);
    }

    let breaks: Vec<usize> = breaks.into_iter().collect();
    println!("{}", serde_json::json!({ "breaks": breaks }));
    Ok(())
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
    if tokens.len() + MAX_NEW_TOKENS >= n_ctx {
        bail!(
            "prompt grande demais para o contexto ({} tokens)",
            tokens.len()
        );
    }

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

    for _ in 0..MAX_NEW_TOKENS {
        let logits_index = batch.n_tokens() as i32 - 1;
        let token = sampler.sample(ctx, logits_index);
        sampler.accept(token);

        if model.vocab().is_eog(token) {
            break;
        }

        output.extend(model.vocab().token_to_piece(token, false, None));
        if output.ends_with(b"\n\n") {
            break;
        }

        batch.clear();
        batch
            .add(token, current, &[0], true)
            .context("falha ao adicionar token gerado")?;
        ctx.decode(&mut batch).context("falha no decode da resposta")?;
        current += 1;
    }

    Ok(String::from_utf8_lossy(&output).to_string())
}

fn parse_starts(text: &str) -> Vec<usize> {
    text.split(|c: char| c == ',' || c == ';' || c == '\n')
        .filter_map(|part| {
            part.split(|c: char| !c.is_ascii_digit())
                .filter_map(|piece| piece.parse::<usize>().ok())
                .next()
        })
        .collect()
}
