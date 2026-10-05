//! Sesión de inferencia: prefill por bloques, decode con sampling y streaming de texto, y
//! reutilización del prefijo común con la secuencia que ya está en la KV cache.

use std::path::Path;
use std::time::Instant;

use brasa_memory::planner::{self, Budget, Fit, MemoryPlan};
use brasa_metal::Context;
use brasa_models::qwen3::{Allocated, Limits, Qwen3, model_shape};
use brasa_tokenizer::{StreamDecoder, Tokenizer};

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

#[derive(Debug)]
pub struct Session {
    ctx: Context,
    model: Qwen3,
    tok: Tokenizer,
    logits: Vec<f32>,
    /// Tokens cuyo KV está en la caché, en orden de posición.
    cached: Vec<u32>,
    /// Si es `true`, los tokens de parada se bloquean (benchmarks: generar siempre `max_new`).
    pub ignore_stop: bool,
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
        let tok = Tokenizer::from_dir(model_dir)?;
        let model = Qwen3::load(&ctx, &model_dir.join("model.brasa"), limits)?;
        let vocab = model.cfg.vocab;
        let session = Self {
            ctx,
            tok,
            logits: vec![0.0; vocab],
            cached: Vec::with_capacity(limits.ctx),
            ignore_stop: false,
            model,
        };
        Ok((session, plan))
    }

    /// Olvida la secuencia en caché (el próximo `generate` hace prefill completo).
    pub fn reset(&mut self) {
        self.cached.clear();
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
