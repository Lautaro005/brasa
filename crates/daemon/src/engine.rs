//! Hilo del modelo: recibe trabajos en orden y devuelve eventos por canal (ADR 0008).
//! El contexto Metal no es `Sync`, así que la sesión vive siempre en este hilo.
//!
//! [`Engine::simulated`] permite tests sin GPU: un handler emite eventos fijos.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use brasa_core::chat::{ChatEvent, ChatRequest};
use brasa_memory::planner::{Budget, MemoryPlan};
use brasa_quant::BrasaFile;
use brasa_runtime::{Limits, Session};
use serde::Serialize;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug)]
pub struct Job {
    pub req: ChatRequest,
    pub events: UnboundedSender<ChatEvent>,
}

/// Datos del modelo cargado que el daemon informa en `/api/status` (U1). Se completan una vez,
/// al cargar, dentro del hilo del modelo.
#[derive(Debug, Clone, Serialize)]
pub struct LoadedModel {
    pub path: String,
    /// sha256 agregado **declarado** en el encabezado del `.brasa` (ADR 0006, `data_sha256`); lo
    /// recalcula `brasa models verify`. El daemon no lo verifica al cargar.
    pub weights_sha256_declarado: String,
    /// Cifra que verifica `brasa models verify`.
    pub weights_bytes: u64,
    pub family: String,
    pub source_repo: String,
    pub source_commit: String,
    /// Contexto del perfil de memoria.
    pub ctx: usize,
    /// Tipo de la KV cache (`f32`, `f16`, `q8_0`).
    pub kv: String,
    /// Tokens por bloque de prefill.
    pub chunk: usize,
    /// Plan de memoria: pesos + KV + workspace + overhead.
    pub plan: MemoryPlan,
}

/// Registro de depuración (`BRASA_DEBUG=1`): llamadas a herramientas, texto y fin, a stderr.
fn log_event(e: &ChatEvent) {
    match e {
        ChatEvent::ToolCall(c) => eprintln!("[brasa] llamada {} {}", c.name, c.arguments),
        ChatEvent::Done { reason, usage } => eprintln!("[brasa] fin {reason:?} {usage:?}"),
        ChatEvent::Error(k, m) => eprintln!("[brasa] error {k:?}: {m}"),
        ChatEvent::Text(t) | ChatEvent::Reasoning(t) => eprint!("{t}"),
    }
}

/// Manejador para mandar trabajos al hilo del modelo.
#[derive(Clone, Debug)]
pub struct Engine {
    tx: mpsc::Sender<Job>,
    /// Encolados que todavía no empezaron.
    pending: Arc<AtomicUsize>,
    /// En ejecución (0 o 1).
    running: Arc<AtomicUsize>,
}

/// (encolados, en ejecución) para `/api/status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Queue {
    pub pending: usize,
    pub running: usize,
}

impl Engine {
    /// Carga el modelo en un hilo nuevo. Devuelve error si no se puede cargar (por ejemplo, si
    /// el planner rechaza el contexto) o si no hay dispositivo Metal.
    pub fn start(model_dir: PathBuf, limits: Limits) -> Result<(Self, LoadedModel), String> {
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<LoadedModel, String>>();
        let pending = Arc::new(AtomicUsize::new(0));
        let running = Arc::new(AtomicUsize::new(0));
        let (p, r) = (pending.clone(), running.clone());
        std::thread::Builder::new()
            .name("brasa-engine".into())
            .spawn(move || {
                let budget = match Budget::this_machine() {
                    Some(b) => b,
                    None => {
                        let _ = ready_tx.send(Err("no hay dispositivo Metal".into()));
                        return;
                    }
                };
                let (mut session, plan) =
                    match Session::load_with_budget(&model_dir, limits, &budget) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e.0));
                            return;
                        }
                    };
                let file = BrasaFile::open(&model_dir.join("model.brasa"));
                let loaded = LoadedModel {
                    path: model_dir.display().to_string(),
                    weights_sha256_declarado: file
                        .as_ref()
                        .map(|f| f.data_sha256().to_string())
                        .unwrap_or_default(),
                    weights_bytes: file.as_ref().map_or(0, |f| f.weights_bytes() as u64),
                    family: file
                        .as_ref()
                        .map(|f| f.family().to_string())
                        .unwrap_or_default(),
                    source_repo: file
                        .as_ref()
                        .map(|f| f.source().0.to_string())
                        .unwrap_or_default(),
                    source_commit: file
                        .as_ref()
                        .map(|f| f.source().1.to_string())
                        .unwrap_or_default(),
                    ctx: limits.ctx,
                    kv: limits.kv.name().to_string(),
                    chunk: limits.max_tokens,
                    plan,
                };
                // El encabezado ya se leyó: no hace falta retener el mmap de los pesos.
                drop(file);
                let _ = ready_tx.send(Ok(loaded));
                let debug = std::env::var_os("BRASA_DEBUG").is_some();
                for job in rx {
                    p.fetch_sub(1, Ordering::Relaxed);
                    r.fetch_add(1, Ordering::Relaxed);
                    // Si el cliente se fue antes de empezar, no se genera nada.
                    if !job.events.is_closed() {
                        session.chat(&job.req, |e| {
                            if debug {
                                log_event(&e);
                            }
                            job.events.send(e).is_ok()
                        });
                    }
                    r.fetch_sub(1, Ordering::Relaxed);
                }
            })
            .map_err(|e| e.to_string())?;
        let loaded = ready_rx
            .recv()
            .map_err(|_| "el hilo del modelo terminó al cargar".to_string())??;
        Ok((
            Self {
                tx,
                pending,
                running,
            },
            loaded,
        ))
    }

    /// Engine de prueba: un handler propio emite los eventos de cada trabajo, sin GPU ni pesos.
    /// La firma del handler es la de `Session::chat`: devolver `false` cancela.
    pub fn simulated<F>(mut handler: F) -> Self
    where
        F: FnMut(&ChatRequest, &mut dyn FnMut(ChatEvent) -> bool) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<Job>();
        let pending = Arc::new(AtomicUsize::new(0));
        let running = Arc::new(AtomicUsize::new(0));
        let (p, r) = (pending.clone(), running.clone());
        std::thread::Builder::new()
            .name("brasa-engine-fake".into())
            .spawn(move || {
                for job in rx {
                    p.fetch_sub(1, Ordering::Relaxed);
                    r.fetch_add(1, Ordering::Relaxed);
                    if !job.events.is_closed() {
                        let mut on_event = |e: ChatEvent| job.events.send(e).is_ok();
                        handler(&job.req, &mut on_event);
                    }
                    r.fetch_sub(1, Ordering::Relaxed);
                }
            })
            .expect("hilo del engine simulado");
        Self {
            tx,
            pending,
            running,
        }
    }

    /// Encola un trabajo.
    pub fn submit(&self, job: Job) -> Result<(), String> {
        self.pending.fetch_add(1, Ordering::Relaxed);
        self.tx.send(job).map_err(|_| {
            self.pending.fetch_sub(1, Ordering::Relaxed);
            "el hilo del modelo terminó".to_string()
        })
    }

    /// Estado de la cola: pedidos encolados y en ejecución.
    pub fn queue(&self) -> Queue {
        Queue {
            pending: self.pending.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
        }
    }
}
