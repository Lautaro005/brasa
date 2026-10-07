//! Planner de memoria v0 (ADR 0007): pesos + KV + workspace + overhead contra un presupuesto,
//! calculado antes de cargar el modelo.

use serde::Serialize;

const MIB: u64 = 1 << 20;
const PAGE: u64 = 16 * 1024;
/// Holgura que se resta a `recommendedMaxWorkingSetSize` para el presupuesto.
pub const BUDGET_HEADROOM: u64 = 512 * MIB;
/// Memoria del proceso fuera de los buffers planificados (tokenizer, runtime de Metal, binario,
/// sampler). Medida en T1.8 y redondeada hacia arriba (ver tests de brasa-models).
pub const PROCESS_OVERHEAD: u64 = 256 * MIB;
/// Alineación de la capacidad de la KV cache por capa (`brasa_kernels::KV_ALIGN`).
pub const KV_ALIGN: usize = 64;
/// Claves por tramo de la atención de decode (`brasa_kernels::DECODE_CHUNK`).
pub const DECODE_CHUNK: usize = 128;
/// Elementos por suma parcial de `add_norm_prep` (`brasa_kernels::norm_partials`).
pub const NORM_PREP_TG: usize = 256;
/// Granularidad del contexto elegido automáticamente.
pub const CTX_STEP: usize = 256;

/// Dimensiones del modelo que determinan su memoria.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelShape {
    /// Bytes de cada tensor de pesos (cada uno va en su propio buffer).
    pub tensor_bytes: Vec<u64>,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
}

/// Tamaños de la sesión.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SessionShape {
    pub ctx: usize,
    /// Tokens por bloque de prefill.
    pub max_tokens: usize,
    pub max_logit_rows: usize,
    /// Bytes de la KV cache por bloque de 32 elementos: 128 en f32, 64 en f16, 34 en Q8 (ADR 0009).
    pub kv_block_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MemoryPlan {
    pub weights: u64,
    pub kv: u64,
    pub workspace: u64,
    pub overhead: u64,
    pub total: u64,
}

/// Bytes que ocupa un buffer de Metal de `n` bytes (se asigna por páginas).
pub fn buffer_bytes(n: u64) -> u64 {
    n.max(1).div_ceil(PAGE) * PAGE
}

/// Bytes de los buffers de workspace de una sesión; debe coincidir con `Qwen3::load`.
pub fn workspace_bytes(m: &ModelShape, s: &SessionShape) -> u64 {
    let f = |n: usize| buffer_bytes(4 * n as u64);
    let (t, h, qd) = (s.max_tokens, m.hidden, m.heads * m.head_dim);
    3 * f(t * h) // x, h, norm_out
        + 2 * f(t * qd) // q, attn
        + 2 * f(t * m.kv_heads * m.head_dim) // k_new, v_new
        + 2 * f(t * m.ffn) // gate, up
        + f(t) // ids
        + f(h.div_ceil(NORM_PREP_TG)) // sumas parciales de RMSNorm en decode
        + f(m.heads * s.ctx.div_ceil(DECODE_CHUNK) * (m.head_dim + 2)) // parciales de decode
        + f(s.max_logit_rows * m.vocab) // logits
        + 2 * f(s.ctx * m.head_dim / 2) // tabla RoPE cos y sin
}

pub fn kv_bytes(m: &ModelShape, s: &SessionShape) -> u64 {
    let cap = s.ctx.next_multiple_of(KV_ALIGN);
    let elems = m.layers * cap * m.kv_heads * m.head_dim;
    2 * buffer_bytes((elems.div_ceil(32) * s.kv_block_bytes) as u64)
}

pub fn plan(m: &ModelShape, s: &SessionShape) -> MemoryPlan {
    let weights = m.tensor_bytes.iter().map(|b| buffer_bytes(*b)).sum();
    let kv = kv_bytes(m, s);
    let workspace = workspace_bytes(m, s);
    let overhead = PROCESS_OVERHEAD;
    MemoryPlan {
        weights,
        kv,
        workspace,
        overhead,
        total: weights + kv + workspace + overhead,
    }
}

/// Presupuesto de memoria para el modelo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Budget {
    pub bytes: u64,
    /// De dónde sale, para el mensaje: "medido: ..." o "estimado: ...".
    pub source: String,
}

impl Budget {
    /// Desde el `recommendedMaxWorkingSetSize` medido de esta máquina.
    pub fn from_working_set(recommended: u64, chip: &str, ram: u64) -> Self {
        Self {
            bytes: recommended.saturating_sub(BUDGET_HEADROOM),
            source: format!(
                "medido en {chip} {} GB: working set recomendado {:.2} GiB − holgura {} MiB",
                ram >> 30,
                gib(recommended),
                BUDGET_HEADROOM / MIB
            ),
        }
    }

    /// Perfil simulado para una Mac de `ram_gib` (estimación si no hay medición).
    pub fn profile(ram_gib: u64) -> Self {
        let ram = ram_gib << 30;
        // Medido: 16 GB -> 74 % de la RAM. Sin medición para 8 GB: 2/3 de la RAM.
        let recommended = if ram_gib >= 16 {
            ram / 100 * 74
        } else {
            ram / 3 * 2
        };
        Self {
            bytes: recommended.saturating_sub(BUDGET_HEADROOM),
            source: format!(
                "estimado para el perfil {ram_gib} GB: working set {:.2} GiB − holgura {} MiB",
                gib(recommended),
                BUDGET_HEADROOM / MIB
            ),
        }
    }

    /// Presupuesto de la máquina actual (requiere GPU Metal).
    pub fn this_machine() -> Option<Self> {
        let info = brasa_metal::device_info()?;
        let ram = brasa_core::sys::sysctl_u64("hw.memsize").unwrap_or(0);
        Some(Self::from_working_set(
            info.recommended_max_working_set,
            &info.name,
            ram,
        ))
    }
}

/// Perfil de memoria (ADR 0029): fija el presupuesto simulado y los valores por defecto del modo
/// agente (`serve`). El contexto por defecto es 16K en los dos (CLAUDE.md: los agentes mandan
/// prompts de 15K a 30K tokens); la KV es Q8 en 8 GB, donde hace falta la memoria, y f16 en 16 GB,
/// donde entra con margen y es más precisa (ADR 0009).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Profile {
    #[serde(rename = "8gb")]
    G8,
    #[serde(rename = "16gb")]
    G16,
}

/// Contexto por defecto del modo agente en todos los perfiles.
pub const AGENT_CTX: usize = 16_384;

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Profile::G8 => "8gb",
            Profile::G16 => "16gb",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "8gb" | "8" => Some(Profile::G8),
            "16gb" | "16" => Some(Profile::G16),
            _ => None,
        }
    }

    /// Perfil de una Mac con `ram` bytes: 8 GB hasta 12 GiB, 16 GB desde ahí.
    pub fn for_ram(ram: u64) -> Self {
        if ram < 12 << 30 {
            Profile::G8
        } else {
            Profile::G16
        }
    }

    /// Perfil de esta Mac (16 GB si no se puede leer la RAM).
    pub fn this_machine() -> Self {
        brasa_core::sys::sysctl_u64("hw.memsize").map_or(Profile::G16, Self::for_ram)
    }

    pub fn ram_gib(self) -> u64 {
        match self {
            Profile::G8 => 8,
            Profile::G16 => 16,
        }
    }

    /// Presupuesto simulado del perfil ([`Budget::profile`]; estimado si no hay medición).
    pub fn budget(self) -> Budget {
        Budget::profile(self.ram_gib())
    }

    /// Contexto por defecto del modo agente.
    pub fn agent_ctx(self) -> usize {
        AGENT_CTX
    }

    /// Tipo de KV por defecto del modo agente (nombre de `KvType`: `q8_0` o `f16`).
    pub fn agent_kv(self) -> &'static str {
        match self {
            Profile::G8 => "q8_0",
            Profile::G16 => "f16",
        }
    }
}

/// Presupuesto para cargar con un perfil simulado en esta máquina: el menor entre el del perfil y
/// el de la máquina (simular 8 GB en una de 16 rechaza lo que no entraría en 8; simular 16 en una
/// de 8 no promete memoria que no hay).
pub fn simulated_budget(profile: Profile, machine: Option<Budget>) -> Budget {
    let p = profile.budget();
    match machine {
        Some(m) if m.bytes < p.bytes => m,
        _ => p,
    }
}

pub fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// Contexto más grande (múltiplo de `CTX_STEP`, hasta `max_ctx`) que entra en el presupuesto.
pub fn max_context(
    m: &ModelShape,
    s: &SessionShape,
    budget: &Budget,
    max_ctx: usize,
) -> Option<usize> {
    let fits = |ctx: usize| plan(m, &SessionShape { ctx, ..*s }).total <= budget.bytes;
    let (mut lo, mut hi) = (0usize, max_ctx / CTX_STEP);
    if !fits(CTX_STEP) {
        return None;
    }
    // Búsqueda binaria sobre múltiplos de CTX_STEP (la memoria crece con ctx).
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(mid * CTX_STEP) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo * CTX_STEP)
}

/// Resultado de verificar un plan contra el presupuesto.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fit {
    Fits(MemoryPlan),
    TooBig {
        plan: MemoryPlan,
        /// Contexto máximo que entra con la misma configuración (si alguno entra).
        max_ctx: Option<usize>,
    },
}

pub fn check(m: &ModelShape, s: &SessionShape, budget: &Budget, model_max_ctx: usize) -> Fit {
    let p = plan(m, s);
    if p.total <= budget.bytes {
        Fit::Fits(p)
    } else {
        Fit::TooBig {
            plan: p,
            max_ctx: max_context(m, s, budget, model_max_ctx),
        }
    }
}

/// Descripción de un plan en una línea.
pub fn describe(p: &MemoryPlan) -> String {
    format!(
        "{:.2} GiB (pesos {:.2} + KV {:.2} + workspace {:.2} + overhead {:.2})",
        gib(p.total),
        gib(p.weights),
        gib(p.kv),
        gib(p.workspace),
        gib(p.overhead)
    )
}

/// Mensaje de rechazo claro (criterio de T1.8).
pub fn rejection_message(
    ctx: usize,
    plan: &MemoryPlan,
    budget: &Budget,
    max_ctx: Option<usize>,
) -> String {
    let mut s = format!(
        "el contexto {ctx} no entra en memoria: necesita {} y el presupuesto es {:.2} GiB ({}).",
        describe(plan),
        gib(budget.bytes),
        budget.source
    );
    match max_ctx {
        Some(c) => s.push_str(&format!(" Contexto máximo que entra: {c} (--ctx {c}).")),
        None => s.push_str(" El modelo no entra ni con el contexto mínimo."),
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_4b() -> ModelShape {
        // Formas de Qwen3-4B; pesos aproximados (2,29 GiB en un tensor) para el test.
        ModelShape {
            tensor_bytes: vec![2_459_000_000],
            layers: 36,
            hidden: 2560,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            ffn: 9728,
            vocab: 151_936,
        }
    }

    fn session(ctx: usize) -> SessionShape {
        SessionShape {
            ctx,
            max_tokens: 128,
            max_logit_rows: 1,
            kv_block_bytes: 128,
        }
    }

    #[test]
    fn kv_coincide_con_la_tabla_de_plan_md() {
        // PLAN.md: KV FP16 a 16 384 = 2,25 GiB -> f32 = 4,5 GiB.
        let kv = kv_bytes(&qwen3_4b(), &session(16_384));
        assert_eq!(kv, 2 * 36 * 16_384 * 8 * 128 * 4);
        assert!((gib(kv) - 4.5).abs() < 1e-9);
    }

    #[test]
    fn max_context_es_el_mayor_que_entra() {
        let m = qwen3_4b();
        let budget = Budget::profile(8);
        let c = max_context(&m, &session(0), &budget, 40_960).unwrap();
        assert!(plan(&m, &session(c)).total <= budget.bytes);
        assert!(plan(&m, &session(c + CTX_STEP)).total > budget.bytes);
        assert_eq!(c % CTX_STEP, 0);
    }

    fn agent(ctx: usize, kv_block_bytes: usize) -> SessionShape {
        SessionShape {
            ctx,
            max_tokens: 512,
            max_logit_rows: 1,
            kv_block_bytes,
        }
    }

    #[test]
    fn perfiles() {
        assert_eq!(Profile::for_ram(8 << 30), Profile::G8);
        assert_eq!(Profile::for_ram(7 << 30), Profile::G8);
        assert_eq!(Profile::for_ram(16 << 30), Profile::G16);
        assert_eq!(Profile::for_ram(32 << 30), Profile::G16);
        for p in [Profile::G8, Profile::G16] {
            assert_eq!(Profile::parse(p.name()), Some(p));
            assert_eq!(p.agent_ctx(), 16_384);
            assert!(p.budget().source.starts_with("estimado"));
        }
        assert_eq!(Profile::G8.agent_kv(), "q8_0");
        assert_eq!(Profile::G16.agent_kv(), "f16");
        assert_eq!(Profile::parse("4gb"), None);
    }

    #[test]
    fn el_modo_agente_entra_en_cada_perfil() {
        // Pesos de Qwen3-4B Q4 como en `brasa plan` (2,20 GiB); el plan real con el archivo lo
        // verifica tests/planner.rs de brasa-runtime. Presupuestos estimados (ADR 0007).
        let m = ModelShape {
            tensor_bytes: vec![2_362_232_013],
            ..qwen3_4b()
        };
        let (q8, f16) = (34, 64);
        let margin =
            |p: Profile, kv| p.budget().bytes as i64 - plan(&m, &agent(AGENT_CTX, kv)).total as i64;
        // 8 GB con Q8: más de 1 GiB de margen. 16 GB con f16: más de 6 GiB.
        assert!(margin(Profile::G8, q8) > 1 << 30);
        assert!(margin(Profile::G16, f16) > 6 << 30);
        // 8 GB con f16 entra por menos de 256 MiB sobre un presupuesto estimado: por eso Q8 ahí.
        let m8 = margin(Profile::G8, f16);
        assert!((0..256 << 20).contains(&m8), "{m8}");
    }

    #[test]
    fn presupuesto_simulado_es_el_menor() {
        let m16 = Budget::from_working_set(12 << 30, "M1 Pro", 16 << 30);
        let b = simulated_budget(Profile::G8, Some(m16.clone()));
        assert_eq!(b, Profile::G8.budget());
        let chico = Budget::from_working_set(4 << 30, "M2", 8 << 30);
        assert_eq!(simulated_budget(Profile::G16, Some(chico.clone())), chico);
        assert_eq!(simulated_budget(Profile::G16, None), Profile::G16.budget());
    }

    #[test]
    fn rechaza_con_mensaje_claro() {
        let m = qwen3_4b();
        let budget = Budget::profile(8);
        match check(&m, &session(32_768), &budget, 40_960) {
            Fit::TooBig { plan, max_ctx } => {
                let msg = rejection_message(32_768, &plan, &budget, max_ctx);
                assert!(msg.contains("no entra") && msg.contains("--ctx"), "{msg}");
            }
            Fit::Fits(_) => panic!("32K con KV f32 no debería entrar en 8 GB"),
        }
        assert!(matches!(
            check(&m, &session(2048), &budget, 40_960),
            Fit::Fits(_)
        ));
    }
}
