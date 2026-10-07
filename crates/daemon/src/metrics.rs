//! Contadores y estadísticas del daemon (U1). Se miden alrededor de los eventos del engine, en
//! el daemon, sin tocar el camino caliente de la inferencia.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use brasa_core::chat::{ChatEvent, ErrorKind, Usage};
use serde::Serialize;

/// Tamaño de la ventana de las estadísticas móviles (TTFT y tok/s de decode).
const WINDOW: usize = 100;

/// Segundos de historia de actividad que guarda `/api/activity`.
pub const HISTORY_S: u64 = 120;

/// Pedidos terminados que se recuerdan en `/api/activity`.
const RECENT: usize = 12;

#[derive(Debug, Default)]
struct Inner {
    endpoints: BTreeMap<&'static str, u64>,
    prompt_tokens: u64,
    generated_tokens: u64,
    cached_tokens: u64,
    cancelled: u64,
    errors: BTreeMap<&'static str, u64>,
    last_ttft_ms: Option<f64>,
    last_decode_tok_s: Option<f64>,
    ttft: VecDeque<f64>,
    decode: VecDeque<f64>,
    next_id: u64,
    active: BTreeMap<u64, Active>,
    recent: VecDeque<Finished>,
    /// Segundo unix → actividad repartida dentro de ese segundo.
    seconds: BTreeMap<u64, Cell>,
    /// Tramo de generación del último pedido cancelado: su uso llega después, por `add_usage`.
    cancelled_span: Option<(SystemTime, SystemTime)>,
}

/// Actividad de un segundo: tokens generados, fracción ocupada en prefill y tokens de prompt
/// procesados en ese prefill (los que no vinieron del prefix cache).
#[derive(Debug, Clone, Copy, Default)]
struct Cell {
    tokens: f64,
    prefill: f64,
    prefill_tokens: f64,
}

#[derive(Debug, Clone)]
struct Active {
    endpoint: &'static str,
    client: Option<String>,
    started: SystemTime,
    first_token: Option<SystemTime>,
}

/// Contadores del daemon desde el arranque.
#[derive(Debug, Default)]
pub struct Metrics {
    inner: Mutex<Inner>,
}

/// Cronómetro de una generación, entre el pedido y el último evento.
#[derive(Debug)]
pub struct GenTimer {
    id: u64,
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

    /// Cuenta el pedido, lo anota como en curso y arranca el cronómetro. `client` es el
    /// producto del `User-Agent` (para distinguir agentes en la GUI).
    pub fn begin(&self, endpoint: &'static str, client: Option<String>) -> GenTimer {
        let mut id = 0;
        if let Ok(mut i) = self.inner.lock() {
            *i.endpoints.entry(endpoint).or_default() += 1;
            i.next_id += 1;
            id = i.next_id;
            i.active.insert(
                id,
                Active {
                    endpoint,
                    client,
                    started: SystemTime::now(),
                    first_token: None,
                },
            );
        }
        GenTimer {
            id,
            started: Instant::now(),
            first_token: None,
            finished: None,
        }
    }

    /// Pasa un evento por el cronómetro y marca el pedido en curso cuando llega el primer token.
    pub fn event(&self, timer: &mut GenTimer, e: &ChatEvent) {
        let had_first = timer.first_token.is_some();
        timer.on_event(e);
        if !had_first
            && timer.first_token.is_some()
            && let Ok(mut i) = self.inner.lock()
            && let Some(a) = i.active.get_mut(&timer.id)
        {
            a.first_token = Some(SystemTime::now());
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
        let now = SystemTime::now();
        if let Some(a) = i.active.remove(&timer.id) {
            let first = a.first_token;
            let prompt = usage.map_or(0.0, |u| {
                u.input_tokens.saturating_sub(u.cached_tokens) as f64
            });
            spread(
                &mut i.seconds,
                a.started,
                first.unwrap_or(now),
                0.0,
                1.0,
                prompt,
            );
            if let (Some(f), Some(u)) = (first, usage) {
                spread(&mut i.seconds, f, now, u.output_tokens as f64, 0.0, 0.0);
            }
            if usage.is_none()
                && error.is_none()
                && let Some(f) = first
            {
                i.cancelled_span = Some((f, now));
            }
            let finished = Finished {
                endpoint: a.endpoint.to_string(),
                client: a.client,
                started_unix_ms: unix_ms(a.started),
                ended_unix_ms: unix_ms(now),
                ttft_ms: ttft,
                decode_tok_s: decode,
                input_tokens: usage.map(|u| u.input_tokens),
                output_tokens: usage.map(|u| u.output_tokens),
                cached_tokens: usage.map(|u| u.cached_tokens),
                outcome: match (usage, error) {
                    (_, Some(k)) => error_name(k),
                    (Some(_), None) => "ok",
                    (None, None) => "cancelled",
                },
            };
            if i.recent.len() == RECENT {
                i.recent.pop_back();
            }
            i.recent.push_front(finished);
        }
        prune(&mut i.seconds, now);
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

    /// Pedido cancelado por el cliente (se desconectó durante el stream). Su uso llega aparte,
    /// desde el hilo del modelo ([`Metrics::add_usage`]), porque el `Done` ya no se pudo entregar.
    pub fn cancelled(&self, timer: GenTimer) {
        self.finish(timer, None, None);
        if let Ok(mut i) = self.inner.lock() {
            i.cancelled += 1;
        }
    }

    /// Suma tokens de un pedido cuyo `Done` no llegó al handler (cancelado).
    pub fn add_usage(&self, u: Usage) {
        if let Ok(mut i) = self.inner.lock() {
            i.prompt_tokens += u.input_tokens as u64;
            i.generated_tokens += u.output_tokens as u64;
            i.cached_tokens += u.cached_tokens as u64;
            if let Some((from, to)) = i.cancelled_span.take() {
                spread(&mut i.seconds, from, to, u.output_tokens as f64, 0.0, 0.0);
                if let Some(r) = i.recent.iter_mut().find(|r| r.outcome == "cancelled") {
                    r.input_tokens.get_or_insert(u.input_tokens);
                    r.output_tokens.get_or_insert(u.output_tokens);
                    r.cached_tokens.get_or_insert(u.cached_tokens);
                }
            }
        }
    }

    /// Pedidos en curso, terminados recientes y actividad por segundo (`GET /api/activity`).
    pub fn activity(&self) -> Activity {
        let now = SystemTime::now();
        let i = self.inner.lock().expect("métricas");
        let now_s = unix_s(now);
        // Los pedidos en curso todavía no sumaron: su prefill se ve en vivo, sus tokens al cerrar.
        let mut live = BTreeMap::new();
        for a in i.active.values() {
            spread(
                &mut live,
                a.started,
                a.first_token.unwrap_or(now),
                0.0,
                1.0,
                0.0,
            );
        }
        let seconds = (now_s.saturating_sub(HISTORY_S - 1)..=now_s)
            .map(|t| {
                let c = i.seconds.get(&t).copied().unwrap_or_default();
                let l: Cell = live.get(&t).copied().unwrap_or_default();
                Second {
                    t,
                    tokens: c.tokens,
                    prefill: (c.prefill + l.prefill).min(1.0),
                    prefill_tokens: c.prefill_tokens,
                    prefill_live: l.prefill > 0.0,
                }
            })
            .collect();
        Activity {
            now_unix_ms: unix_ms(now),
            history_s: HISTORY_S,
            active: i
                .active
                .values()
                .map(|a| Running {
                    endpoint: a.endpoint.to_string(),
                    client: a.client.clone(),
                    started_unix_ms: unix_ms(a.started),
                    phase: if a.first_token.is_some() {
                        "decode"
                    } else {
                        "prefill"
                    },
                    ttft_ms: a.first_token.map(|f| ms_between(a.started, f)),
                })
                .collect(),
            recent: i.recent.iter().cloned().collect(),
            seconds,
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
            cancelled: i.cancelled,
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

fn unix_ms(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn unix_s(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn ms_between(a: SystemTime, b: SystemTime) -> f64 {
    b.duration_since(a).map_or(0.0, |d| d.as_secs_f64() * 1e3)
}

/// Reparte `tokens` y `prefill_tokens` (y `busy` segundos de prefill por segundo) uniformemente
/// sobre `[from, to]`, en casilleros de un segundo unix.
fn spread(
    seconds: &mut BTreeMap<u64, Cell>,
    from: SystemTime,
    to: SystemTime,
    tokens: f64,
    busy: f64,
    prefill_tokens: f64,
) {
    let a = from
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let b = to
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    if b < a {
        return;
    }
    let span = b - a;
    let mut t = a.floor();
    while t < b || (span == 0.0 && t <= a) {
        let lo = a.max(t);
        let hi = b.min(t + 1.0);
        let frac = if span > 0.0 { (hi - lo) / span } else { 1.0 };
        let cell = seconds.entry(t as u64).or_default();
        cell.tokens += tokens * frac;
        cell.prefill += busy * (hi - lo);
        cell.prefill_tokens += prefill_tokens * frac;
        t += 1.0;
    }
}

fn prune(seconds: &mut BTreeMap<u64, Cell>, now: SystemTime) {
    let keep_from = unix_s(now).saturating_sub(HISTORY_S);
    *seconds = seconds.split_off(&keep_from);
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
    /// Pedidos cancelados por el cliente a mitad de la generación.
    pub cancelled: u64,
    pub ttft_ms: Stats,
    pub decode_tok_s: Stats,
    pub errors: BTreeMap<String, u64>,
}

/// Pedido en curso.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Running {
    pub endpoint: String,
    pub client: Option<String>,
    pub started_unix_ms: u64,
    /// `prefill` hasta el primer token, después `decode`.
    pub phase: &'static str,
    pub ttft_ms: Option<f64>,
}

/// Pedido terminado (ok, cancelado o con error).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Finished {
    pub endpoint: String,
    pub client: Option<String>,
    pub started_unix_ms: u64,
    pub ended_unix_ms: u64,
    pub ttft_ms: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub input_tokens: Option<usize>,
    pub output_tokens: Option<usize>,
    pub cached_tokens: Option<usize>,
    /// `ok`, `cancelled` o el tipo de error.
    pub outcome: &'static str,
}

/// Un segundo de actividad: tokens generados (repartidos sobre el tramo de decode de cada pedido
/// terminado), fracción del segundo ocupada en prefill y tokens de prompt procesados en ese
/// prefill (sin los del prefix cache; se conocen al terminar el pedido).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Second {
    pub t: u64,
    pub tokens: f64,
    pub prefill: f64,
    pub prefill_tokens: f64,
    /// Hay un prefill en curso en este segundo (sus tokens todavía no se conocen).
    pub prefill_live: bool,
}

/// `GET /api/activity`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Activity {
    pub now_unix_ms: u64,
    pub history_s: u64,
    pub active: Vec<Running>,
    pub recent: Vec<Finished>,
    pub seconds: Vec<Second>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuenta_request_ttft_y_decode() {
        let m = Metrics::new();
        let mut t = m.begin("/v1/chat/completions", None);
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
    fn cancelado_cuenta_pedido_y_tokens() {
        let m = Metrics::new();
        let mut t = m.begin("/v1/chat/completions", None);
        t.on_event(&ChatEvent::Text("hola".into()));
        m.cancelled(t);
        m.add_usage(Usage {
            input_tokens: 20,
            output_tokens: 300,
            cached_tokens: 0,
        });
        let s = m.snapshot();
        assert_eq!(s.cancelled, 1);
        assert_eq!((s.prompt_tokens, s.generated_tokens), (20, 300));
        assert!(s.errors.is_empty());
    }

    #[test]
    fn cuenta_errores_por_tipo() {
        let m = Metrics::new();
        let t = m.begin("/v1/responses", None);
        m.finish(t, None, Some(ErrorKind::ContextLength));
        let s = m.snapshot();
        assert_eq!(s.errors["context_length"], 1);
        assert_eq!(s.endpoints["/v1/responses"], 1);
        assert_eq!(s.ttft_ms.samples, 0);
    }

    #[test]
    fn actividad_en_curso_y_reciente() {
        let m = Metrics::new();
        let mut t = m.begin("/v1/messages", Some("claude-cli/2.1".into()));
        let a = m.activity();
        assert_eq!(a.active.len(), 1);
        assert_eq!(a.active[0].phase, "prefill");
        assert_eq!(a.seconds.len(), HISTORY_S as usize);
        m.event(&mut t, &ChatEvent::Text("hola".into()));
        assert_eq!(m.activity().active[0].phase, "decode");
        let u = Usage {
            input_tokens: 7,
            output_tokens: 30,
            cached_tokens: 0,
        };
        m.finish(t, Some(u), None);
        let a = m.activity();
        assert!(a.active.is_empty());
        assert_eq!(a.recent[0].outcome, "ok");
        assert_eq!(a.recent[0].client.as_deref(), Some("claude-cli/2.1"));
        let total: f64 = a.seconds.iter().map(|s| s.tokens).sum();
        assert!((total - 30.0).abs() < 1e-6, "{total}");
        let prompt: f64 = a.seconds.iter().map(|s| s.prefill_tokens).sum();
        assert!((prompt - 7.0).abs() < 1e-6, "{prompt}");
    }

    #[test]
    fn reparte_tokens_por_segundo() {
        let mut s = BTreeMap::new();
        let t0 = UNIX_EPOCH + std::time::Duration::from_millis(10_500);
        let t1 = UNIX_EPOCH + std::time::Duration::from_millis(12_500);
        spread(&mut s, t0, t1, 40.0, 0.0, 8.0);
        assert_eq!(s[&10].tokens, 10.0);
        assert_eq!(s[&11].tokens, 20.0);
        assert_eq!(s[&12].tokens, 10.0);
        assert_eq!(s[&11].prefill_tokens, 4.0);
    }

    #[test]
    fn cancelado_reparte_el_uso_tardio() {
        let m = Metrics::new();
        let mut t = m.begin("/v1/chat/completions", None);
        m.event(&mut t, &ChatEvent::Text("a".into()));
        m.cancelled(t);
        m.add_usage(Usage {
            input_tokens: 3,
            output_tokens: 12,
            cached_tokens: 0,
        });
        let a = m.activity();
        assert_eq!(a.recent[0].outcome, "cancelled");
        assert_eq!(a.recent[0].output_tokens, Some(12));
        let total: f64 = a.seconds.iter().map(|s| s.tokens).sum();
        assert!((total - 12.0).abs() < 1e-6);
    }
}
