//! De la base de tuning a los parámetros de lanzamiento que usa el engine (ADR 0029).
//!
//! Al cargar un modelo, el runtime abre la base del fingerprint actual y la traduce a un
//! [`Launch`]. Sin base, con una base inválida o con `BRASA_TUNING=off`, quedan los valores por
//! defecto (los fijados a mano), con un estado que explica por qué. Nunca entra en pánico.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use brasa_kernels::{GemvOp, KvType, Launch};

use crate::db::{TuningDb, TuningKey, default_dir};
use crate::fingerprint::Fingerprint;

/// Nombre de la clave `kernel` de la atención de decode en la base.
pub const ATTN_LANES: &str = "attn_decode_lanes";
/// Nombre del único parámetro que se tunea: simdgroups por threadgroup.
pub const SG: &str = "SG";

/// Forma canónica de un GEMV en la base.
pub fn gemv_shape(rows: usize, cols: usize) -> String {
    format!("rows={rows},cols={cols}")
}

/// Forma canónica de la atención de decode en la base.
pub fn lanes_shape(hkv: usize, lk: usize) -> String {
    format!("hkv={hkv},lk={lk}")
}

/// Variante de compilación de la atención de decode (tipo de KV y grupo GQA).
pub fn lanes_variant(kv: KvType, group: usize) -> String {
    format!("kv={},g={group}", kv.name())
}

/// Lee `a=1,b=2` en un mapa.
fn fields(s: &str) -> Option<BTreeMap<&str, &str>> {
    if s.is_empty() {
        return Some(BTreeMap::new());
    }
    s.split(',')
        .map(|kv| kv.split_once('='))
        .collect::<Option<BTreeMap<_, _>>>()
}

fn num(m: &BTreeMap<&str, &str>, k: &str) -> Option<usize> {
    m.get(k)?.parse().ok()
}

/// Aplica una entrada de la base a `launch`. `false` si la clave no es de un parámetro que este
/// binario sepa usar o el valor no es admisible (la entrada se ignora).
fn apply_entry(launch: &mut Launch, key: &TuningKey, params: &BTreeMap<String, u32>) -> bool {
    let Some(&sg) = params.get(SG) else {
        return false;
    };
    let sg = sg as usize;
    let Some(shape) = fields(&key.shape) else {
        return false;
    };
    if let Some(op) = GemvOp::from_kernel_name(&key.kernel) {
        let (Some(rows), Some(cols)) = (num(&shape, "rows"), num(&shape, "cols")) else {
            return false;
        };
        return launch.set_gemv(op, rows, cols, sg).is_ok();
    }
    if key.kernel == ATTN_LANES {
        let Some(variant) = fields(&key.variant) else {
            return false;
        };
        let (Some(kv), Some(group), Some(lk)) = (
            variant.get("kv").and_then(|k| KvType::parse(k)),
            num(&variant, "g"),
            num(&shape, "lk"),
        ) else {
            return false;
        };
        return launch.set_attn_lanes(kv, group, lk, sg).is_ok();
    }
    false
}

/// Traduce una base a parámetros de lanzamiento. Devuelve también cuántas entradas se ignoraron.
pub fn launch_from_db(db: &TuningDb) -> (Launch, usize) {
    let mut launch = Launch::default();
    let mut ignored = 0;
    for (k, v) in db.iter() {
        if !apply_entry(&mut launch, k, &v.params) {
            ignored += 1;
        }
    }
    (launch, ignored)
}

/// De dónde salieron los parámetros de lanzamiento.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuningStatus {
    /// Base del fingerprint actual, con `entries` entradas (`ignored` no aplicables).
    Tuned {
        fingerprint_id: String,
        entries: usize,
        ignored: usize,
    },
    /// No hay base para este fingerprint: valores por defecto.
    Missing { fingerprint_id: String },
    /// La base existe pero no se puede usar (dañada, de otro esquema o de otro fingerprint).
    Invalid {
        fingerprint_id: String,
        error: String,
    },
    /// `BRASA_TUNING=off`: valores por defecto a pedido.
    Disabled,
    /// No hay carpeta de bases (sin `HOME` ni `BRASA_TUNING_DIR`).
    NoDir,
}

impl TuningStatus {
    /// `true` si se usan parámetros de una base.
    pub fn is_tuned(&self) -> bool {
        matches!(self, TuningStatus::Tuned { .. })
    }
}

impl fmt::Display for TuningStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TuningStatus::Tuned {
                fingerprint_id,
                entries,
                ignored,
            } => {
                write!(f, "base {fingerprint_id} con {entries} entradas")?;
                if *ignored > 0 {
                    write!(f, " ({ignored} ignoradas)")?;
                }
                Ok(())
            }
            TuningStatus::Missing { fingerprint_id } => write!(
                f,
                "sin base para el fingerprint {fingerprint_id}: valores por defecto \
                 (`brasa tune` la crea en menos de un minuto)"
            ),
            TuningStatus::Invalid { error, .. } => {
                write!(f, "base inválida, valores por defecto: {error}")
            }
            TuningStatus::Disabled => {
                write!(f, "desactivado (BRASA_TUNING=off): valores por defecto")
            }
            TuningStatus::NoDir => {
                write!(f, "sin carpeta de bases (falta HOME): valores por defecto")
            }
        }
    }
}

/// Parámetros de lanzamiento y su origen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub launch: Launch,
    pub status: TuningStatus,
}

impl Resolved {
    fn defaults(status: TuningStatus) -> Self {
        Self {
            launch: Launch::default(),
            status,
        }
    }
}

/// Parámetros de la base de `fingerprint` en `dir` (`None`: no hay carpeta).
pub fn resolve(dir: Option<&Path>, fingerprint: &Fingerprint) -> Resolved {
    let Some(dir) = dir else {
        return Resolved::defaults(TuningStatus::NoDir);
    };
    let id = fingerprint.id();
    if !dir.join(TuningDb::file_name(fingerprint)).exists() {
        return Resolved::defaults(TuningStatus::Missing { fingerprint_id: id });
    }
    match TuningDb::open(dir, fingerprint) {
        Ok(db) => {
            let (launch, ignored) = launch_from_db(&db);
            Resolved {
                launch,
                status: TuningStatus::Tuned {
                    fingerprint_id: id,
                    entries: db.len(),
                    ignored,
                },
            }
        }
        Err(e) => Resolved::defaults(TuningStatus::Invalid {
            fingerprint_id: id,
            error: e.to_string(),
        }),
    }
}

/// Si `BRASA_TUNING` pide no usar la base (`off`, `0`, `false`, `no`).
pub fn disabled_by_env() -> bool {
    std::env::var("BRASA_TUNING").is_ok_and(|v| matches!(v.trim(), "off" | "0" | "false" | "no"))
}

/// Parámetros para esta máquina y estos kernels: la base de [`default_dir`] para
/// [`Fingerprint::current`], salvo `BRASA_TUNING=off`.
pub fn resolve_current() -> Resolved {
    if disabled_by_env() {
        return Resolved::defaults(TuningStatus::Disabled);
    }
    resolve(default_dir().as_deref(), &Fingerprint::current())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::TuningValue;
    use brasa_kernels::WeightType;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fp() -> Fingerprint {
        Fingerprint {
            chip: "Apple M1 Pro".into(),
            gpu_cores: Some(16),
            cpu_performance_cores: 8,
            cpu_efficiency_cores: 2,
            memory_bytes: 16 << 30,
            metal_family: Some("apple7".into()),
            macos_version: "27.0".into(),
            macos_build: "27A1".into(),
            kernels_version: "0123456789abcdef".into(),
        }
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let p = std::env::temp_dir().join(format!(
                "brasa-apply-test-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn val(sg: u32) -> TuningValue {
        TuningValue {
            params: BTreeMap::from([(SG.to_string(), sg)]),
            gpu_us: 10.0,
            default_us: 11.0,
            samples: 5,
        }
    }

    fn sample_db() -> TuningDb {
        let mut db = TuningDb::new(fp());
        db.insert(
            TuningKey::new("gemv_scaled3_q4_0_f32", gemv_shape(6144, 2560), ""),
            val(4),
        );
        db.insert(
            TuningKey::new("gemv_fast_q4_0_f32", gemv_shape(2560, 9728), ""),
            val(8),
        );
        db.insert(
            TuningKey::new(
                ATTN_LANES,
                lanes_shape(8, 16384),
                lanes_variant(KvType::Q8_0, 4),
            ),
            val(2),
        );
        db
    }

    #[test]
    fn base_inexistente_da_valores_por_defecto() {
        let dir = TempDir::new();
        let r = resolve(Some(&dir.0), &fp());
        assert_eq!(r.launch, Launch::default());
        assert_eq!(
            r.status,
            TuningStatus::Missing {
                fingerprint_id: fp().id()
            }
        );
        assert!(r.status.to_string().contains("brasa tune"));
        let r = resolve(None, &fp());
        assert_eq!(
            (r.launch, r.status),
            (Launch::default(), TuningStatus::NoDir)
        );
    }

    #[test]
    fn base_del_fingerprint_se_aplica() {
        let dir = TempDir::new();
        sample_db().save_in(&dir.0).unwrap();
        let r = resolve(Some(&dir.0), &fp());
        assert!(r.status.is_tuned(), "{}", r.status);
        assert_eq!(r.launch.len(), 3);
        assert_eq!(r.launch.gemv_sg(GemvOp::Scaled3, 6144, 2560), 4);
        assert_eq!(
            r.launch.gemv_sg(GemvOp::Fast(WeightType::Q4_0), 2560, 9728),
            8
        );
        assert_eq!(r.launch.attn_lanes_sg(KvType::Q8_0, 4, 16000), 2);
        // Lo que no está en la base sigue con los valores por defecto.
        assert_eq!(r.launch.attn_lanes_sg(KvType::F16, 4, 16000), 4);
        assert_eq!(
            r.launch.gemv_sg(GemvOp::Fast(WeightType::Q4_0), 2560, 4096),
            2
        );
    }

    #[test]
    fn base_de_otro_fingerprint_se_ignora() {
        let dir = TempDir::new();
        // Otro fingerprint: otro nombre de archivo, así que para este no hay base.
        let mut otro = fp();
        otro.kernels_version = "fedcba9876543210".into();
        let mut db = TuningDb::new(otro.clone());
        db.insert(
            TuningKey::new("gemv_scaled3_q4_0_f32", gemv_shape(6144, 2560), ""),
            val(8),
        );
        db.save_in(&dir.0).unwrap();
        let r = resolve(Some(&dir.0), &fp());
        assert_eq!(r.launch, Launch::default());
        assert!(matches!(r.status, TuningStatus::Missing { .. }));
        // Un archivo con el nombre de este fingerprint pero de otro: inválido, sin pánico.
        std::fs::rename(
            dir.0.join(TuningDb::file_name(&otro)),
            dir.0.join(TuningDb::file_name(&fp())),
        )
        .unwrap();
        let r = resolve(Some(&dir.0), &fp());
        assert_eq!(r.launch, Launch::default());
        assert!(
            matches!(&r.status, TuningStatus::Invalid { error, .. } if error.contains("otro fingerprint")),
            "{}",
            r.status
        );
    }

    #[test]
    fn base_danada_da_valores_por_defecto() {
        let dir = TempDir::new();
        std::fs::write(dir.0.join(TuningDb::file_name(&fp())), "{").unwrap();
        let r = resolve(Some(&dir.0), &fp());
        assert_eq!(r.launch, Launch::default());
        assert!(matches!(r.status, TuningStatus::Invalid { .. }));
    }

    #[test]
    fn entradas_desconocidas_o_invalidas_se_ignoran() {
        let mut db = sample_db();
        // Kernel que no se tunea en este binario, sg no admitido, filas no divisibles, forma rota.
        db.insert(
            TuningKey::new("gemm_tiled_q4_0_f32", "m=9728,k=2560,t=512", ""),
            val(2),
        );
        db.insert(
            TuningKey::new("gemv_scaled_swiglu_q4_0_f32", gemv_shape(9728, 2560), ""),
            val(3),
        );
        db.insert(
            TuningKey::new("gemv_fast_q8_0_f32", gemv_shape(100, 2560), ""),
            val(8),
        );
        db.insert(TuningKey::new("gemv_fast_q6_0_f32", "rows", ""), val(2));
        db.insert(
            TuningKey::new(ATTN_LANES, lanes_shape(8, 2048), "kv=f12,g=4"),
            val(2),
        );
        let (launch, ignored) = launch_from_db(&db);
        assert_eq!(ignored, 5);
        assert_eq!(launch.len(), 3);
    }
}
