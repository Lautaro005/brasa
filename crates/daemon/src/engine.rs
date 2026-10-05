//! Hilo del modelo: recibe trabajos en orden y devuelve eventos por canal (ADR 0008).
//! El contexto Metal no es `Sync`, así que la sesión vive siempre en este hilo.

use std::path::PathBuf;
use std::sync::mpsc;

use brasa_core::chat::{ChatEvent, ChatRequest};
use brasa_runtime::{Limits, Session};
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug)]
pub struct Job {
    pub req: ChatRequest,
    pub events: UnboundedSender<ChatEvent>,
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
}

impl Engine {
    /// Carga el modelo en un hilo nuevo. Devuelve error si no se puede cargar (por ejemplo, si
    /// el planner rechaza el contexto).
    pub fn start(model_dir: PathBuf, limits: Limits) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        std::thread::Builder::new()
            .name("brasa-engine".into())
            .spawn(move || {
                let mut session = match Session::load(&model_dir, limits) {
                    Ok(s) => {
                        let _ = ready_tx.send(Ok(()));
                        s
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.0));
                        return;
                    }
                };
                let debug = std::env::var_os("BRASA_DEBUG").is_some();
                for job in rx {
                    // Si el cliente se fue antes de empezar, no se genera nada.
                    if job.events.is_closed() {
                        continue;
                    }
                    session.chat(&job.req, |e| {
                        if debug {
                            log_event(&e);
                        }
                        job.events.send(e).is_ok()
                    });
                }
            })
            .map_err(|e| e.to_string())?;
        ready_rx
            .recv()
            .map_err(|_| "el hilo del modelo terminó al cargar".to_string())??;
        Ok(Self { tx })
    }

    /// Encola un trabajo.
    pub fn submit(&self, job: Job) -> Result<(), String> {
        self.tx
            .send(job)
            .map_err(|_| "el hilo del modelo terminó".to_string())
    }
}
