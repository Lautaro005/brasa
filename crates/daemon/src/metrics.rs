//! Contadores y estadísticas del daemon (U1). Se miden alrededor de los eventos del engine, en
//! el daemon, sin tocar el camino caliente de la inferencia.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::time::Instant;

use brasa_core::chat::{ChatEvent, ErrorKind, Usage};
use serde::Serialize;

/// Tamaño de la ventana de las estadísticas móviles (TTFT y tok/s de decode).
const WINDOW: usize = 100;

#[derive(Debug, Default)]
struct Inner {
    endpoints: BTreeMap<&'static str, u64>,
    prompt_tokens: u64,
    generated_tokens: u64,
    cached_tokens: u64,
    errors: BTreeMap<&'static str, u64>,
    last_ttft_ms: Option<f64>,
    last_decode_tok_s: Option<f64>,
    ttft: VecDeque<f64>,
    decode: VecDeque<f64>,
}

/// Contadores del daemon desde el arranque.
#[derive(Debug, Default)]
pub struct Metrics {
    inner: Mutex<Inner>,
}

/// Cronómetro de una generación, entre el pedido y el último evento.
#[derive(Debug)]
pub struct GenTimer {
    started: Instant,
    first_token: Option<Instant>,
    finished: Option<Instant>,
}

fn error_name(k: ErrorKind) -> &'static str {
    match k {
        ErrorKind::ContextLength => "context_length",
        ErrorKind::InvalidRequest => "invalid_request",
        ErrorKind::Internal => "internal",
    }
}

fn push(window: &mut VecDeque<f64>, v: f64) {
    if window.len() == WINDOW {
        window.pop_front();
    }
    window.push_back(v);
}

fn stats(window: &VecDeque<f64>, last: Option<f64>) -> Stats {
    let samples = window.len();
    if samples == 0 {
        return Stats {
            last,
            mean: None,
            p50: None,
            samples: 0,
        };
    }
    let mean = window.iter().sum::<f64>() / samples as f64;
    let mut sorted: Vec<f64> = window.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let p50 = sorted[samples / 2];
    Stats {
        last,
        mean: Some(mean),
        p50: Some(p50),
        samples,
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Cuenta el pedido y arranca el cronómetro.
    pub fn begin(&self, endpoint: &'static str) -> GenTimer {
        if let Ok(mut i) = self.inner.lock() {
            *i.endpoints.entry(endpoint).or_default() += 1;
        }
        GenTimer {
            started: Instant::now(),
            first_token: None,
            finished: None,
        }
    }

    /// Cierra el cronómetro y suma uso, latencias y errores.
    pub fn finish(&self, timer: GenTimer, usage: Option<Usage>, error: Option<ErrorKind>) {
        let ttft = timer
            .first_token
            .map(|f| f.saturating_duration_since(timer.started).as_secs_f64() * 1e3);
        let decode = match (timer.first_token, usage) {
            (Some(first), Some(u)) if u.output_tokens >= 2 => {
                let end = timer.finished.unwrap_or_else(Instant::now);
                let span = end.saturating_duration_since(first).as_secs_f64();
                (span > 0.0).then(|| (u.output_tokens - 1) as f64 / span)
            }
            _ => None,
        };
        let Ok(mut i) = self.inner.lock() else {
            return;
        };
        if let Some(u) = usage {
            i.prompt_tokens += u.input_tokens as u64;
            i.generated_tokens += u.output_tokens as u64;
            i.cached_tokens += u.cached_tokens as u64;
        }
        if let Some(v) = ttft {
            i.last_ttft_ms = Some(v);
            push(&mut i.ttft, v);
        }
        if let Some(v) = decode {
            i.last_decode_tok_s = Some(v);
            push(&mut i.decode, v);
        }
        if let Some(k) = error {
            *i.errors.entry(error_name(k)).or_default() += 1;
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let i = self.inner.lock().expect("métricas");
        Snapshot {
            endpoints: i
                .endpoints
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
            prompt_tokens: i.prompt_tokens,
            generated_tokens: i.generated_tokens,
            cached_tokens: i.cached_tokens,
            ttft_ms: stats(&i.ttft, i.last_ttft_ms),
            decode_tok_s: stats(&i.decode, i.last_decode_tok_s),
            errors: i
                .errors
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
        }
    }
}

impl GenTimer {
    /// Registra el instante del primer token generado y del cierre, para TTFT y decode tok/s.
    pub fn on_event(&mut self, e: &ChatEvent) {
        match e {
            ChatEvent::Text(_) | ChatEvent::Reasoning(_) | ChatEvent::ToolCall(_) => {
                if self.first_token.is_none() {
                    self.first_token = Some(Instant::now());
                }
            }
            ChatEvent::Done { .. } | ChatEvent::Error(..) => {
                self.finished = Some(Instant::now());
            }
        }
    }
}

/// Estadística de una ventana móvil de hasta 100 muestras.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Stats {
    pub last: Option<f64>,
    pub mean: Option<f64>,
    pub p50: Option<f64>,
    pub samples: usize,
}

/// Foto serializable de los contadores (`GET /api/metrics`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Snapshot {
    pub endpoints: BTreeMap<String, u64>,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    /// Tokens de prompt servidos desde el prefix cache.
    pub cached_tokens: u64,
    pub ttft_ms: Stats,
    pub decode_tok_s: Stats,
    pub errors: BTreeMap<String, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuenta_request_ttft_y_decode() {
        let m = Metrics::new();
        let mut t = m.begin("/v1/chat/completions");
        std::thread::sleep(std::time::Duration::from_millis(2));
        t.on_event(&ChatEvent::Text("hola".into()));
        std::thread::sleep(std::time::Duration::from_millis(2));
        t.on_event(&ChatEvent::Done {
            reason: brasa_core::chat::FinishReason::Stop,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cached_tokens: 4,
            },
        });
        m.finish(
            t,
            Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
                cached_tokens: 4,
            }),
            None,
        );
        let s = m.snapshot();
        assert_eq!(s.endpoints["/v1/chat/completions"], 1);
        assert_eq!(
            (s.prompt_tokens, s.generated_tokens, s.cached_tokens),
            (10, 5, 4)
        );
        assert!(s.ttft_ms.last.unwrap() >= 1.0);
        assert!(s.decode_tok_s.last.unwrap() > 0.0);
    }

    #[test]
    fn cuenta_errores_por_tipo() {
        let m = Metrics::new();
        let t = m.begin("/v1/responses");
        m.finish(t, None, Some(ErrorKind::ContextLength));
        let s = m.snapshot();
        assert_eq!(s.errors["context_length"], 1);
        assert_eq!(s.endpoints["/v1/responses"], 1);
        assert_eq!(s.ttft_ms.samples, 0);
    }
}
