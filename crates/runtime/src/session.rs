//! Sesión de inferencia: prefill por bloques, decode con sampling y streaming de texto, y
//! reutilización del prefijo común con la secuencia que ya está en la KV cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use brasa_memory::planner::{self, Budget, Fit, MemoryPlan};
use brasa_metal::Context;
use brasa_models::qwen3::{Allocated, Limits, Qwen3, model_shape, weight_types};
use brasa_tokenizer::{StreamDecoder, Tokenizer};
use brasa_tuner::{Resolved, TuningStatus};

use crate::sampler::Sampler;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// El modelo emitió un token de parada (`<|im_end|>`, `<|endoftext|>`).
    Stop,
    MaxTokens,
    ContextFull,
    Cancelled,
}

/// Métricas de una generación, medidas en esta corrida.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenStats {
    pub prompt_tokens: usize,
    /// Tokens del prompt que ya estaban en la KV cache (no se recalcularon).
    pub reused_tokens: usize,
    pub prefill_ms: f64,
    /// Tiempo hasta el primer token generado (prefill + primer muestreo).
    pub ttft_ms: f64,
    pub generated: usize,
    pub decode_ms: f64,
    pub stop: StopReason,
}

impl GenStats {
    pub fn prefill_tok_s(&self) -> f64 {
        (self.prompt_tokens - self.reused_tokens) as f64 / (self.prefill_ms / 1e3)
    }

    /// Tokens por segundo después del primero.
    pub fn decode_tok_s(&self) -> f64 {
        if self.generated < 2 {
            return 0.0;
        }
        (self.generated - 1) as f64 / (self.decode_ms / 1e3)
    }
}

/// Tamaño y fecha de modificación de los archivos del tokenizer: si cambian, se vuelve a leer.
type Stamp = [Option<(u64, SystemTime)>; 2];

fn stamp(dir: &Path) -> Stamp {
    ["tokenizer.json", "tokenizer_config.json"].map(|name| {
        let m = std::fs::metadata(dir.join(name)).ok()?;
        Some((m.len(), m.modified().ok()?))
    })
}

/// Tokenizer de `dir`, leído una sola vez por proceso (ADR 0027). Parsear `tokenizer.json`
/// (~11 MB de JSON) en cada carga hace crecer la huella del proceso ~27 MB por ciclo
/// `load` → `idle`: el allocator de macOS no devuelve esas regiones grandes liberadas. Se guarda
/// una entrada por carpeta; si los archivos cambian, se reemplaza.
fn shared_tokenizer(dir: &Path) -> Result<Arc<Tokenizer>> {
    type Cache = HashMap<PathBuf, (Stamp, Arc<Tokenizer>)>;
    static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
    let key = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let st = stamp(&key);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let cache = cache.get_or_insert_with(HashMap::new);
    if let Some((_, tok)) = cache.get(&key).filter(|(s, _)| *s == st) {
        return Ok(tok.clone());
    }
    let tok = Arc::new(Tokenizer::from_dir(&key)?);
    cache.insert(key, (st, tok.clone()));
    Ok(tok)
}

/// Dimensiones de decode del modelo de `model_dir` para el autotuner (ADR 0029). Solo lee el
/// encabezado de `model.brasa`.
pub fn tune_dims(model_dir: &Path) -> Result<brasa_tuner::tune::ModelDims> {
    let path = model_dir.join("model.brasa");
    let (shape, _) = model_shape(&path)?;
    let (layer_type, head_type) = weight_types(&path)?;
    Ok(brasa_tuner::tune::ModelDims {
        layers: shape.layers,
        hidden: shape.hidden,
        heads: shape.heads,
        kv_heads: shape.kv_heads,
        head_dim: shape.head_dim,
        ffn: shape.ffn,
        vocab: shape.vocab,
        layer_type,
        head_type,
    })
}

/// Aplica los parámetros resueltos al modelo. Si el pipeline no admite alguno (una base escrita
/// por otro binario con los mismos kernels no debería traerlo), quedan los valores por defecto.
fn apply_tuning(ctx: &Context, model: &mut Qwen3, r: Resolved) -> TuningStatus {
    match model.set_launch(ctx, r.launch) {
        Ok(()) => r.status,
        Err(e) => TuningStatus::Invalid {
            fingerprint_id: match &r.status {
                TuningStatus::Tuned { fingerprint_id, .. } => fingerprint_id.clone(),
                _ => String::new(),
            },
            error: e.0,
        },
    }
}

#[derive(Debug)]
pub struct Session {
    ctx: Context,
    model: Qwen3,
    tok: Arc<Tokenizer>,
    logits: Vec<f32>,
    /// Tokens cuyo KV está en la caché, en orden de posición.
    cached: Vec<u32>,
    /// Si es `true`, los tokens de parada se bloquean (benchmarks: generar siempre `max_new`).
    pub ignore_stop: bool,
    /// De dónde salieron los parámetros de lanzamiento de decode (ADR 0029).
    tuning: TuningStatus,
}

impl Session {
    /// Plan de memoria de cargar `model_dir` con `limits`, sin cargar nada (ADR 0007).
    /// Devuelve el resultado de compararlo con `budget` y el contexto máximo del modelo.
    pub fn plan(model_dir: &Path, limits: Limits, budget: &Budget) -> Result<(Fit, usize)> {
        let (shape, cfg) = model_shape(&model_dir.join("model.brasa"))?;
        let fit = planner::check(&shape, &limits.session_shape(), budget, cfg.max_position);
        Ok((fit, cfg.max_position))
    }

    /// Carga una carpeta de modelo (`model.brasa`, `tokenizer.json`, `tokenizer_config.json`)
    /// después de verificar que entra en el presupuesto de esta máquina; si no, la rechaza.
    pub fn load(model_dir: &Path, limits: Limits) -> Result<Self> {
        let budget =
            Budget::this_machine().ok_or_else(|| Error("no hay dispositivo Metal".into()))?;
        Self::load_with_budget(model_dir, limits, &budget).map(|(s, _)| s)
    }

    /// Como `load`, con un presupuesto explícito. Devuelve también el plan aceptado.
    pub fn load_with_budget(
        model_dir: &Path,
        limits: Limits,
        budget: &Budget,
    ) -> Result<(Self, MemoryPlan)> {
        let plan = match Self::plan(model_dir, limits, budget)? {
            (Fit::Fits(p), _) => p,
            (Fit::TooBig { plan, max_ctx }, _) => {
                return Err(Error(planner::rejection_message(
                    limits.ctx, &plan, budget, max_ctx,
                )));
            }
        };
        let ctx = Context::new()?;
        let tok = shared_tokenizer(model_dir)?;
        let mut model = Qwen3::load(&ctx, &model_dir.join("model.brasa"), limits)?;
        // Base de tuning del fingerprint actual; sin base o inválida, los valores por defecto.
        let tuning = apply_tuning(&ctx, &mut model, brasa_tuner::resolve_current());
        let vocab = model.cfg.vocab;
        let session = Self {
            ctx,
            tok,
            logits: vec![0.0; vocab],
            cached: Vec::with_capacity(limits.ctx),
            ignore_stop: false,
            tuning,
            model,
        };
        Ok((session, plan))
    }

    /// Olvida la secuencia en caché (el próximo `generate` hace prefill completo).
    pub fn reset(&mut self) {
        self.cached.clear();
    }

    /// Origen de los parámetros de lanzamiento de decode: base de tuning o valores por defecto.
    pub fn tuning(&self) -> &TuningStatus {
        &self.tuning
    }

    /// Memoria reservada en buffers Metal por el modelo.
    pub fn allocated(&self) -> Allocated {
        self.model.allocated()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    pub fn limits(&self) -> Limits {
        self.model.limits
    }

    pub fn vocab(&self) -> usize {
        self.model.cfg.vocab
    }

    /// Genera a partir de `prompt` (tokens completos de la conversación). Llama a `on_text` con
    /// cada fragmento de texto nuevo; si devuelve `false`, se cancela.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampler: &mut Sampler,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<GenStats> {
        let limits = self.model.limits;
        if prompt.is_empty() {
            return Err(Error("prompt vacío".into()));
        }
        if prompt.len() >= limits.ctx {
            return Err(Error(format!(
                "el prompt tiene {} tokens y el contexto es de {}",
                prompt.len(),
                limits.ctx
            )));
        }
        // Prefijo común con lo que ya está en caché; siempre se recalcula al menos el último
        // token del prompt para obtener sus logits.
        let common = self
            .cached
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count()
            .min(prompt.len() - 1);
        self.cached.truncate(common);

        let t0 = Instant::now();
        for chunk in prompt[common..].chunks(limits.max_tokens) {
            let pos = self.cached.len();
            self.model
                .forward(&self.ctx, chunk, pos, 1, &mut self.logits)?;
            self.cached.extend_from_slice(chunk);
        }
        let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;

        let mut dec = StreamDecoder::new();
        let mut generated = 0;
        let mut ttft_ms = 0.0;
        let mut stop = StopReason::MaxTokens;
        let t1 = Instant::now();
        while generated < max_new {
            if self.ignore_stop {
                for id in &self.tok.stop_ids {
                    self.logits[*id as usize] = f32::NEG_INFINITY;
                }
            }
            let next = sampler.sample(&self.logits);
            if generated == 0 {
                ttft_ms = t0.elapsed().as_secs_f64() * 1e3;
            }
            generated += 1;
            if self.tok.stop_ids.contains(&next) {
                stop = StopReason::Stop;
                break;
            }
            let text = dec.push(&self.tok.bpe, next);
            if !text.is_empty() && !on_text(&text) {
                stop = StopReason::Cancelled;
                break;
            }
            if self.cached.len() >= limits.ctx {
                stop = StopReason::ContextFull;
                break;
            }
            let pos = self.cached.len();
            self.model
                .forward(&self.ctx, &[next], pos, 1, &mut self.logits)?;
            self.cached.push(next);
        }
        let rest = dec.finish();
        if !rest.is_empty() {
            on_text(&rest);
        }
        Ok(GenStats {
            prompt_tokens: prompt.len(),
            reused_tokens: common,
            prefill_ms,
            ttft_ms,
            generated,
            decode_ms: t1.elapsed().as_secs_f64() * 1e3 - (ttft_ms - prefill_ms),
            stop,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_se_lee_una_vez() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/qwen3-4b/tokenizer");
        let a = shared_tokenizer(&dir).unwrap();
        let b = shared_tokenizer(&dir.join("../tokenizer")).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "el tokenizer se volvió a leer");
    }
}
