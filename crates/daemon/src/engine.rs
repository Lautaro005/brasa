//! Hilo del modelo: recibe trabajos en orden y devuelve eventos por canal (ADR 0008).
//! El contexto Metal no es `Sync`, así que la sesión vive siempre en este hilo.
//!
//! Desde V5 el mismo hilo atiende el Model Manager (ADR 0025): `load`, `idle`, `pause`, `resume`
//! y `stop`. El modelo se libera soltando la `Session`, sin tocar `brasa-runtime`.
//!
//! [`Engine::simulated`] permite tests sin GPU: un handler emite eventos fijos.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use brasa_core::chat::{ChatEvent, ChatRequest, ErrorKind};
use brasa_memory::planner::{Budget, MemoryPlan};
use brasa_quant::BrasaFile;
use brasa_runtime::{Limits, Session};
use serde::Serialize;
use tokio::sync::mpsc::UnboundedSender;

use crate::metrics::Metrics;

#[derive(Debug)]
pub struct Job {
    pub req: ChatRequest,
    pub events: UnboundedSender<ChatEvent>,
    /// Recibe el uso si el `Done` no se puede entregar (el cliente canceló).
    pub metrics: Arc<Metrics>,
}

impl Job {
    /// Envía un evento al handler. Si el cliente ya se fue y el evento es el `Done`, su uso
    /// se suma igual a las métricas. Devuelve `false` si hay que cancelar.
    fn send(&self, e: ChatEvent) -> bool {
        let usage = match &e {
            ChatEvent::Done { usage, .. } => Some(*usage),
            _ => None,
        };
        let ok = self.events.send(e).is_ok();
        if let (false, Some(u)) = (ok, usage) {
            self.metrics.add_usage(u);
        }
        ok
    }
}

/// Datos del modelo cargado que el daemon informa en `/api/status` (U1). Se completan al cargar,
/// dentro del hilo del modelo.
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
    /// Origen de los parámetros de lanzamiento de decode (base de tuning o valores por defecto,
    /// ADR 0029).
    pub tuning: String,
}

/// Estado del modelo (ADR 0025). Lo informa `/api/status` y lo cambian las operaciones del
/// Model Manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    /// Cargando (transitorio).
    Loading,
    /// Pesos y KV en memoria.
    Loaded,
    /// Pesos y KV liberados; el daemon sigue vivo.
    Idle,
    /// Cargado, pero la cola no saca trabajos.
    Paused,
    /// El hilo del modelo terminó.
    Stopped,
}

impl ModelState {
    fn code(self) -> u8 {
        match self {
            Self::Loading => 0,
            Self::Loaded => 1,
            Self::Idle => 2,
            Self::Paused => 3,
            Self::Stopped => 4,
        }
    }

    fn from_code(v: u8) -> Self {
        match v {
            0 => Self::Loading,
            1 => Self::Loaded,
            2 => Self::Idle,
            3 => Self::Paused,
            _ => Self::Stopped,
        }
    }

    /// Nombre que se informa por API.
    pub fn name(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Loaded => "loaded",
            Self::Idle => "idle",
            Self::Paused => "paused",
            Self::Stopped => "stopped",
        }
    }
}

impl Serialize for ModelState {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

/// Un backend de generación: la `Session` real o un handler simulado. No es `Send`: la sesión
/// de Metal vive y muere dentro del hilo del engine (se crea con la `Factory`, que sí lo es).
trait Backend {
    fn chat(&mut self, req: &ChatRequest, on: &mut dyn FnMut(ChatEvent) -> bool);
}

struct SessionBackend(Session);

impl Backend for SessionBackend {
    fn chat(&mut self, req: &ChatRequest, on: &mut dyn FnMut(ChatEvent) -> bool) {
        self.0.chat(req, on);
    }
}

/// Crea (o recrea) el backend y los datos de `/api/status`. Se llama al arrancar y en cada `load`.
type Factory = Box<dyn FnMut() -> Result<(Box<dyn Backend>, LoadedModel), String> + Send>;

enum Cmd {
    Load,
    Idle,
    Pause,
    Resume,
    Stop,
}

struct Control {
    cmd: Cmd,
    reply: mpsc::Sender<Result<(), String>>,
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

/// Manejador para mandar trabajos y órdenes al hilo del modelo.
#[derive(Clone, Debug)]
pub struct Engine {
    tx: mpsc::Sender<Job>,
    control: mpsc::Sender<Control>,
    /// Encolados que todavía no empezaron.
    pending: Arc<AtomicUsize>,
    /// En ejecución (0 o 1).
    running: Arc<AtomicUsize>,
    state: Arc<AtomicU8>,
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
    /// `budget`: presupuesto de memoria para el planner; `None` es el de esta máquina (con
    /// `serve --perfil`, el del perfil simulado).
    pub fn start(
        model_dir: PathBuf,
        limits: Limits,
        budget: Option<Budget>,
    ) -> Result<(Self, LoadedModel), String> {
        let make: Factory = Box::new(move || {
            let budget = match &budget {
                Some(b) => b.clone(),
                None => {
                    Budget::this_machine().ok_or_else(|| "no hay dispositivo Metal".to_string())?
                }
            };
            let (session, plan) =
                Session::load_with_budget(&model_dir, limits, &budget).map_err(|e| e.0)?;
            let tuning = session.tuning().to_string();
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
                tuning,
            };
            // El encabezado ya se leyó: no hace falta retener el mmap de los pesos.
            drop(file);
            Ok((
                Box::new(SessionBackend(session)) as Box<dyn Backend>,
                loaded,
            ))
        });
        Self::spawn(make)
    }

    /// Engine de prueba: un handler propio emite los eventos de cada trabajo, sin GPU ni pesos,
    /// y sostiene las transiciones del Model Manager. La firma del handler es la de
    /// `Session::chat`: devolver `false` cancela.
    pub fn simulated<F>(handler: F) -> Self
    where
        F: FnMut(&ChatRequest, &mut dyn FnMut(ChatEvent) -> bool) + Send + 'static,
    {
        let shared = Arc::new(Mutex::new(handler));
        let make: Factory = Box::new(move || {
            let backend = Box::new(FakeBackend {
                handler: shared.clone(),
            }) as Box<dyn Backend>;
            Ok((backend, fake_model()))
        });
        Self::spawn(make).expect("hilo del engine simulado").0
    }

    fn spawn(mut make: Factory) -> Result<(Self, LoadedModel), String> {
        let (tx, jobs) = mpsc::channel::<Job>();
        let (ctrltx, control) = mpsc::channel::<Control>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<LoadedModel, String>>();
        let pending = Arc::new(AtomicUsize::new(0));
        let running = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(AtomicU8::new(ModelState::Loading.code()));
        let (p, r, st) = (pending.clone(), running.clone(), state.clone());
        std::thread::Builder::new()
            .name("brasa-engine".into())
            .spawn(move || {
                let (backend, loaded) = match make() {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                st.store(ModelState::Loaded.code(), Ordering::Relaxed);
                let _ = ready_tx.send(Ok(loaded));
                run(Some(backend), &mut make, &jobs, &control, &p, &r, &st);
            })
            .map_err(|e| e.to_string())?;
        let loaded = ready_rx
            .recv()
            .map_err(|_| "el hilo del modelo terminó al cargar".to_string())??;
        Ok((
            Self {
                tx,
                control: ctrltx,
                pending,
                running,
                state,
            },
            loaded,
        ))
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

    /// Estado del modelo (ADR 0025).
    pub fn state(&self) -> ModelState {
        ModelState::from_code(self.state.load(Ordering::Relaxed))
    }

    fn order(&self, cmd: Cmd) -> Result<(), String> {
        let (reply, rx) = mpsc::channel();
        self.control
            .send(Control { cmd, reply })
            .map_err(|_| "el hilo del modelo terminó".to_string())?;
        rx.recv()
            .map_err(|_| "el hilo del modelo terminó".to_string())?
    }

    /// Vuelve a cargar el modelo con el plan de memoria (ADR 0025).
    pub fn load(&self) -> Result<(), String> {
        self.order(Cmd::Load)
    }

    /// Libera los pesos y la KV. El daemon sigue vivo.
    pub fn idle(&self) -> Result<(), String> {
        self.order(Cmd::Idle)
    }

    /// Frena la cola sin descargar el modelo.
    pub fn pause(&self) -> Result<(), String> {
        self.order(Cmd::Pause)
    }

    /// Vuelve a sacar trabajos de la cola.
    pub fn resume(&self) -> Result<(), String> {
        self.order(Cmd::Resume)
    }

    /// Suelta el modelo y termina el hilo del engine.
    pub fn stop(&self) -> Result<(), String> {
        self.order(Cmd::Stop)
    }
}

struct FakeBackend<F> {
    handler: Arc<Mutex<F>>,
}

impl<F> Backend for FakeBackend<F>
where
    F: FnMut(&ChatRequest, &mut dyn FnMut(ChatEvent) -> bool) + Send + 'static,
{
    fn chat(&mut self, req: &ChatRequest, on: &mut dyn FnMut(ChatEvent) -> bool) {
        let mut h = self.handler.lock().unwrap_or_else(|e| e.into_inner());
        h(req, on);
    }
}

fn fake_model() -> LoadedModel {
    LoadedModel {
        path: "(simulado)".into(),
        weights_sha256_declarado: String::new(),
        weights_bytes: 0,
        family: "qwen3".into(),
        source_repo: String::new(),
        source_commit: String::new(),
        ctx: 2048,
        kv: "f16".into(),
        chunk: 512,
        plan: MemoryPlan {
            weights: 0,
            kv: 0,
            workspace: 0,
            overhead: 0,
            total: 0,
        },
        tuning: "valores por defecto (engine simulado)".into(),
    }
}

/// Atiende órdenes de control con prioridad sobre la cola. Devuelve `true` si hay que terminar.
fn control_msg(
    c: Control,
    backend: &mut Option<Box<dyn Backend>>,
    make: &mut Factory,
    paused: &mut bool,
    state: &AtomicU8,
) -> bool {
    match c.cmd {
        Cmd::Load => {
            state.store(ModelState::Loading.code(), Ordering::Relaxed);
            let r = match make() {
                Ok((b, _m)) => {
                    *backend = Some(b);
                    *paused = false;
                    state.store(ModelState::Loaded.code(), Ordering::Relaxed);
                    Ok(())
                }
                Err(e) => {
                    *backend = None;
                    state.store(ModelState::Idle.code(), Ordering::Relaxed);
                    Err(e)
                }
            };
            let _ = c.reply.send(r);
        }
        Cmd::Idle => {
            *backend = None;
            *paused = false;
            state.store(ModelState::Idle.code(), Ordering::Relaxed);
            let _ = c.reply.send(Ok(()));
        }
        Cmd::Pause => {
            let r = if backend.is_none() {
                Err("el modelo está en idle; cargalo con `brasa model load`".into())
            } else {
                *paused = true;
                state.store(ModelState::Paused.code(), Ordering::Relaxed);
                Ok(())
            };
            let _ = c.reply.send(r);
        }
        Cmd::Resume => {
            let r = if *paused {
                *paused = false;
                state.store(ModelState::Loaded.code(), Ordering::Relaxed);
                Ok(())
            } else {
                Err("el modelo no está en pausa".into())
            };
            let _ = c.reply.send(r);
        }
        Cmd::Stop => {
            *backend = None;
            state.store(ModelState::Stopped.code(), Ordering::Relaxed);
            let _ = c.reply.send(Ok(()));
            return true;
        }
    }
    false
}

/// Sirve un trabajo; si el modelo está en `idle`, lo vuelve a cargar antes (ADR 0025).
fn serve_job(
    job: &Job,
    backend: &mut Option<Box<dyn Backend>>,
    make: &mut Factory,
    state: &AtomicU8,
    debug: bool,
) {
    if backend.is_none() {
        state.store(ModelState::Loading.code(), Ordering::Relaxed);
        match make() {
            Ok((b, _m)) => {
                *backend = Some(b);
                state.store(ModelState::Loaded.code(), Ordering::Relaxed);
            }
            Err(e) => {
                state.store(ModelState::Idle.code(), Ordering::Relaxed);
                job.send(ChatEvent::Error(
                    ErrorKind::Internal,
                    format!("no se pudo cargar el modelo: {e}"),
                ));
                return;
            }
        }
    }
    if let Some(b) = backend.as_mut() {
        b.chat(&job.req, &mut |e| {
            if debug {
                log_event(&e);
            }
            job.send(e)
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    mut backend: Option<Box<dyn Backend>>,
    make: &mut Factory,
    jobs: &mpsc::Receiver<Job>,
    control: &mpsc::Receiver<Control>,
    pending: &AtomicUsize,
    running: &AtomicUsize,
    state: &AtomicU8,
) {
    let debug = std::env::var_os("BRASA_DEBUG").is_some();
    let mut paused = false;
    loop {
        // Las órdenes mandan: se atienden antes de sacar un trabajo.
        let mut stop = false;
        loop {
            match control.try_recv() {
                Ok(c) => stop |= control_msg(c, &mut backend, make, &mut paused, state),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if stop {
            return;
        }
        if paused {
            // En pausa solo se espera una orden: los trabajos quedan encolados.
            match control.recv() {
                Ok(c) => {
                    if control_msg(c, &mut backend, make, &mut paused, state) {
                        return;
                    }
                }
                Err(_) => return,
            }
            continue;
        }
        match jobs.recv_timeout(Duration::from_millis(25)) {
            Ok(job) => {
                pending.fetch_sub(1, Ordering::Relaxed);
                running.fetch_add(1, Ordering::Relaxed);
                if !job.events.is_closed() {
                    serve_job(&job, &mut backend, make, state, debug);
                }
                running.fetch_sub(1, Ordering::Relaxed);
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}
