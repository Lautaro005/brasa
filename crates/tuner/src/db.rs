//! Base de tuning: un JSON versionado por fingerprint (ADR 0026).
//!
//! Clave = (kernel, forma, variante); valor = parámetros elegidos y la medición que los respalda.
//! Un archivo de otra versión de esquema o de otro fingerprint se rechaza con un error, nunca se
//! usa a medias. La escritura es atómica (archivo temporal + `rename`).

use crate::fingerprint::Fingerprint;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Versión del esquema del archivo. Un archivo con otra versión se rechaza.
pub const SCHEMA_VERSION: u32 = 1;

/// Identifica una medición: qué kernel, con qué forma y en qué variante de compilación.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuningKey {
    /// Nombre de la función MSL, por ejemplo "flash_attn_gqa".
    pub kernel: String,
    /// Forma canónica de la llamada, por ejemplo "t=512,hq=32,hkv=8,d=128".
    pub shape: String,
    /// Variante de compilación que no se tunea, por ejemplo "kv=f16,g=4".
    pub variant: String,
}

impl TuningKey {
    pub fn new(
        kernel: impl Into<String>,
        shape: impl Into<String>,
        variant: impl Into<String>,
    ) -> Self {
        Self {
            kernel: kernel.into(),
            shape: shape.into(),
            variant: variant.into(),
        }
    }
}

/// Resultado del tuning para una clave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuningValue {
    /// Parámetros elegidos, por ejemplo `{"FA_ROWS": 16, "BC": 64}`.
    pub params: BTreeMap<String, u32>,
    /// Tiempo de GPU (mediana) con esos parámetros, en µs.
    pub gpu_us: f64,
    /// Tiempo de GPU (mediana) con los parámetros por defecto, en µs, medido en la misma corrida.
    pub default_us: f64,
    /// Repeticiones de cada medición.
    pub samples: u32,
}

/// Una entrada tal como se guarda (campos planos; `flatten` no admite `deny_unknown_fields`).
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    kernel: String,
    shape: String,
    variant: String,
    params: BTreeMap<String, u32>,
    gpu_us: f64,
    default_us: f64,
    samples: u32,
}

impl Entry {
    fn new(k: &TuningKey, v: &TuningValue) -> Self {
        Self {
            kernel: k.kernel.clone(),
            shape: k.shape.clone(),
            variant: k.variant.clone(),
            params: v.params.clone(),
            gpu_us: v.gpu_us,
            default_us: v.default_us,
            samples: v.samples,
        }
    }

    fn split(self) -> (TuningKey, TuningValue) {
        (
            TuningKey {
                kernel: self.kernel,
                shape: self.shape,
                variant: self.variant,
            },
            TuningValue {
                params: self.params,
                gpu_us: self.gpu_us,
                default_us: self.default_us,
                samples: self.samples,
            },
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DbFile {
    schema: u32,
    fingerprint_id: String,
    fingerprint: Fingerprint,
    entries: Vec<Entry>,
}

/// Errores de la base de tuning. Ninguno entra en pánico: el llamador decide si re-tunear.
#[derive(Debug)]
pub enum TuningError {
    Io {
        path: PathBuf,
        error: std::io::Error,
    },
    /// El archivo no es un JSON válido de la base o le faltan campos.
    Corrupt { path: PathBuf, detail: String },
    /// El archivo es de otra versión de esquema.
    Schema { path: PathBuf, found: u64 },
    /// El archivo es de otro fingerprint (otra máquina, otro macOS u otros kernels).
    Fingerprint {
        path: PathBuf,
        found: String,
        expected: String,
    },
}

impl fmt::Display for TuningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Corrupt { path, detail } => write!(
                f,
                "{}: base de tuning dañada ({detail}); borrala para volver a tunear",
                path.display()
            ),
            Self::Schema { path, found } => write!(
                f,
                "{}: base de tuning con esquema {found}, este binario usa {SCHEMA_VERSION}; \
                 hay que volver a tunear",
                path.display()
            ),
            Self::Fingerprint {
                path,
                found,
                expected,
            } => write!(
                f,
                "{}: base de tuning de otro fingerprint ({found}, se esperaba {expected}); \
                 hay que volver a tunear",
                path.display()
            ),
        }
    }
}

impl std::error::Error for TuningError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Base de tuning de un fingerprint, en memoria.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningDb {
    fingerprint: Fingerprint,
    entries: BTreeMap<TuningKey, TuningValue>,
}

impl TuningDb {
    /// Base vacía para `fingerprint`.
    pub fn new(fingerprint: Fingerprint) -> Self {
        Self {
            fingerprint,
            entries: BTreeMap::new(),
        }
    }

    pub fn fingerprint(&self) -> &Fingerprint {
        &self.fingerprint
    }

    /// Nombre del archivo de la base dentro de su carpeta: `<fingerprint id>.json`.
    pub fn file_name(fingerprint: &Fingerprint) -> String {
        format!("{}.json", fingerprint.id())
    }

    /// Abre la base de `fingerprint` en `dir`. Si el archivo no existe devuelve una base vacía;
    /// si existe pero es inválido, de otro esquema o de otro fingerprint, devuelve el error.
    pub fn open(dir: &Path, fingerprint: &Fingerprint) -> Result<Self, TuningError> {
        let path = dir.join(Self::file_name(fingerprint));
        match Self::load(&path, fingerprint) {
            Err(TuningError::Io { error, .. }) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(Self::new(fingerprint.clone()))
            }
            r => r,
        }
    }

    /// Carga `path` y verifica que sea del esquema actual y de `expected`.
    pub fn load(path: &Path, expected: &Fingerprint) -> Result<Self, TuningError> {
        let text = std::fs::read_to_string(path).map_err(|error| TuningError::Io {
            path: path.to_path_buf(),
            error,
        })?;
        let corrupt = |detail: String| TuningError::Corrupt {
            path: path.to_path_buf(),
            detail,
        };
        // Primero el esquema, así un archivo de otra versión no se informa como dañado.
        let raw: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| corrupt(e.to_string()))?;
        let schema = raw
            .get("schema")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| corrupt("falta el campo `schema`".into()))?;
        if schema != u64::from(SCHEMA_VERSION) {
            return Err(TuningError::Schema {
                path: path.to_path_buf(),
                found: schema,
            });
        }
        let file: DbFile = serde_json::from_value(raw).map_err(|e| corrupt(e.to_string()))?;
        if file.fingerprint != *expected {
            return Err(TuningError::Fingerprint {
                path: path.to_path_buf(),
                found: file.fingerprint.id(),
                expected: expected.id(),
            });
        }
        if file.fingerprint_id != expected.id() {
            return Err(corrupt(format!(
                "fingerprint_id {} no coincide con el fingerprint ({})",
                file.fingerprint_id,
                expected.id()
            )));
        }
        let mut entries = BTreeMap::new();
        for e in file.entries {
            let (key, value) = e.split();
            if !(value.gpu_us.is_finite() && value.default_us.is_finite()) {
                return Err(corrupt(format!("tiempo no finito en {key:?}")));
            }
            if entries.insert(key.clone(), value).is_some() {
                return Err(corrupt(format!("clave repetida {key:?}")));
            }
        }
        Ok(Self {
            fingerprint: file.fingerprint,
            entries,
        })
    }

    /// Guarda en `dir/<fingerprint id>.json` (crea la carpeta) y devuelve la ruta.
    pub fn save_in(&self, dir: &Path) -> Result<PathBuf, TuningError> {
        std::fs::create_dir_all(dir).map_err(|error| TuningError::Io {
            path: dir.to_path_buf(),
            error,
        })?;
        let path = dir.join(Self::file_name(&self.fingerprint));
        self.save(&path)?;
        Ok(path)
    }

    /// Guarda en `path` de forma atómica: escribe `<path>.tmp.<pid>`, hace `fsync` y lo renombra.
    /// Un corte a mitad de camino deja el archivo anterior intacto.
    pub fn save(&self, path: &Path) -> Result<(), TuningError> {
        let file = DbFile {
            schema: SCHEMA_VERSION,
            fingerprint_id: self.fingerprint.id(),
            fingerprint: self.fingerprint.clone(),
            entries: self.entries.iter().map(|(k, v)| Entry::new(k, v)).collect(),
        };
        let mut text = serde_json::to_string_pretty(&file).map_err(|e| TuningError::Corrupt {
            path: path.to_path_buf(),
            detail: e.to_string(),
        })?;
        text.push('\n');
        let mut tmp_name = path.as_os_str().to_owned();
        tmp_name.push(format!(".tmp.{}", std::process::id()));
        let tmp = PathBuf::from(tmp_name);
        let io = |p: &Path| {
            let p = p.to_path_buf();
            move |error| TuningError::Io { path: p, error }
        };
        let write = || -> std::io::Result<()> {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(text.as_bytes())?;
            f.sync_all()
        };
        if let Err(e) = write() {
            let _ = std::fs::remove_file(&tmp);
            return Err(io(&tmp)(e));
        }
        std::fs::rename(&tmp, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            io(path)(e)
        })
    }

    pub fn get(&self, key: &TuningKey) -> Option<&TuningValue> {
        self.entries.get(key)
    }

    /// Inserta o reemplaza; devuelve el valor anterior.
    pub fn insert(&mut self, key: TuningKey, value: TuningValue) -> Option<TuningValue> {
        self.entries.insert(key, value)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entradas en orden de clave.
    pub fn iter(&self) -> impl Iterator<Item = (&TuningKey, &TuningValue)> {
        self.entries.iter()
    }
}

/// Carpeta por defecto de las bases: `$BRASA_TUNING_DIR` si está definida, si no
/// `~/Library/Application Support/brasa/tuning`. `None` si no hay `HOME`.
pub fn default_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("BRASA_TUNING_DIR").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d));
    }
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(PathBuf::from(home).join("Library/Application Support/brasa/tuning"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fp() -> Fingerprint {
        Fingerprint {
            chip: "Apple M1 Pro".into(),
            gpu_cores: Some(14),
            cpu_performance_cores: 6,
            cpu_efficiency_cores: 2,
            memory_bytes: 16 << 30,
            metal_family: Some("apple7".into()),
            macos_version: "15.5".into(),
            macos_build: "24F74".into(),
            kernels_version: "0123456789abcdef".into(),
        }
    }

    /// Carpeta temporal única, borrada al salir del scope.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let p = std::env::temp_dir().join(format!(
                "brasa-tuner-test-{}-{}",
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

    fn value(us: f64) -> TuningValue {
        TuningValue {
            params: BTreeMap::from([("FA_ROWS".to_string(), 16), ("BC".to_string(), 64)]),
            gpu_us: us,
            default_us: 2.0 * us,
            samples: 20,
        }
    }

    fn sample_db() -> TuningDb {
        let mut db = TuningDb::new(fp());
        db.insert(
            TuningKey::new("flash_attn_gqa", "t=512,hq=32,hkv=8,d=128", "kv=f16,g=4"),
            value(100.0),
        );
        db.insert(
            TuningKey::new("gemm_tiled_q4_0_f32", "m=9728,k=2560,t=512", ""),
            value(2500.5),
        );
        db
    }

    #[test]
    fn roundtrip() {
        let dir = TempDir::new();
        let db = sample_db();
        let path = db.save_in(&dir.0).unwrap();
        assert_eq!(path, dir.0.join(format!("{}.json", fp().id())));
        let back = TuningDb::open(&dir.0, &fp()).unwrap();
        assert_eq!(back, db);
        assert_eq!(back.len(), 2);
        let k = TuningKey::new("flash_attn_gqa", "t=512,hq=32,hkv=8,d=128", "kv=f16,g=4");
        assert_eq!(back.get(&k), Some(&value(100.0)));
        // Guardar dos veces da el mismo archivo (salida determinista) y no deja temporales.
        let a = std::fs::read(&path).unwrap();
        back.save_in(&dir.0).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), a);
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
    }

    #[test]
    fn insert_reemplaza_y_consulta() {
        let mut db = TuningDb::new(fp());
        assert!(db.is_empty());
        let k = TuningKey::new("gemv_fast_q4_0_f32", "rows=2560,cols=9728", "");
        assert!(db.insert(k.clone(), value(10.0)).is_none());
        assert_eq!(db.insert(k.clone(), value(8.0)), Some(value(10.0)));
        assert_eq!(db.get(&k).unwrap().gpu_us, 8.0);
        assert!(
            db.get(&TuningKey::new(
                "gemv_fast_q4_0_f32",
                "rows=2560,cols=9728",
                "x"
            ))
            .is_none()
        );
    }

    #[test]
    fn archivo_ausente_da_base_vacia() {
        let dir = TempDir::new();
        let db = TuningDb::open(&dir.0, &fp()).unwrap();
        assert!(db.is_empty());
        assert_eq!(db.fingerprint(), &fp());
    }

    #[test]
    fn otro_fingerprint_se_rechaza() {
        let dir = TempDir::new();
        let path = dir.0.join("db.json");
        sample_db().save(&path).unwrap();
        let mut otro = fp();
        otro.kernels_version = "fedcba9876543210".into();
        let err = TuningDb::load(&path, &otro).unwrap_err();
        assert!(matches!(err, TuningError::Fingerprint { .. }), "{err}");
        let mut otro = fp();
        otro.macos_build = "24G84".into();
        assert!(matches!(
            TuningDb::load(&path, &otro),
            Err(TuningError::Fingerprint { .. })
        ));
        // Con otro fingerprint `open` busca otro archivo: base vacía, la vieja queda intacta.
        let mut otro = fp();
        otro.memory_bytes = 8 << 30;
        sample_db().save_in(&dir.0).unwrap();
        assert!(TuningDb::open(&dir.0, &otro).unwrap().is_empty());
    }

    #[test]
    fn otro_esquema_se_rechaza() {
        let dir = TempDir::new();
        let path = dir.0.join("db.json");
        sample_db().save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let text = text.replacen(
            &format!("\"schema\": {SCHEMA_VERSION}"),
            &format!("\"schema\": {}", SCHEMA_VERSION + 1),
            1,
        );
        std::fs::write(&path, text).unwrap();
        let err = TuningDb::load(&path, &fp()).unwrap_err();
        assert!(
            matches!(err, TuningError::Schema { found, .. } if found == u64::from(SCHEMA_VERSION) + 1),
            "{err}"
        );
        assert!(err.to_string().contains("volver a tunear"));
    }

    #[test]
    fn archivo_corrupto_da_error_claro() {
        let dir = TempDir::new();
        let path = dir.0.join("db.json");
        let good = {
            sample_db().save(&path).unwrap();
            std::fs::read_to_string(&path).unwrap()
        };
        let casos = [
            "".to_string(),
            "{".to_string(),
            "\u{0}\u{1}basura".to_string(),
            "[]".to_string(),
            "{\"schema\": \"1\"}".to_string(),
            format!("{{\"schema\": {SCHEMA_VERSION}}}"),
            good[..good.len() / 2].to_string(),
            good.replacen("\"gpu_us\"", "\"gpu_uss\"", 1),
            good.replacen(&fp().id(), "0000000000000000", 1),
        ];
        for texto in casos {
            std::fs::write(&path, &texto).unwrap();
            let err = TuningDb::load(&path, &fp()).unwrap_err();
            assert!(
                matches!(err, TuningError::Corrupt { .. }),
                "{texto:?}: {err}"
            );
            assert!(err.to_string().contains("dañada"), "{err}");
        }
    }

    #[test]
    fn clave_repetida_es_corrupta() {
        let dir = TempDir::new();
        let path = dir.0.join("db.json");
        let mut db = TuningDb::new(fp());
        db.insert(TuningKey::new("a", "s", "v"), value(1.0));
        db.save(&path).unwrap();
        let mut raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let e = raw["entries"][0].clone();
        raw["entries"].as_array_mut().unwrap().push(e);
        std::fs::write(&path, raw.to_string()).unwrap();
        assert!(matches!(
            TuningDb::load(&path, &fp()),
            Err(TuningError::Corrupt { .. })
        ));
    }

    #[test]
    fn escritura_atomica_no_pisa_si_falla() {
        let dir = TempDir::new();
        let path = dir.0.join("db.json");
        sample_db().save(&path).unwrap();
        let antes = std::fs::read(&path).unwrap();
        // Un directorio en lugar del temporal hace fallar la escritura antes del rename.
        let tmp = dir.0.join(format!("db.json.tmp.{}", std::process::id()));
        std::fs::create_dir(&tmp).unwrap();
        let mut db = sample_db();
        db.insert(TuningKey::new("x", "y", "z"), value(1.0));
        assert!(matches!(db.save(&path), Err(TuningError::Io { .. })));
        assert_eq!(std::fs::read(&path).unwrap(), antes);
    }

    #[test]
    fn default_dir_en_application_support() {
        // Solo se verifica la forma sin tocar el entorno (los tests corren en paralelo).
        if std::env::var_os("BRASA_TUNING_DIR").is_none() {
            let d = default_dir().unwrap();
            assert!(
                d.ends_with("Library/Application Support/brasa/tuning"),
                "{d:?}"
            );
        }
    }
}
