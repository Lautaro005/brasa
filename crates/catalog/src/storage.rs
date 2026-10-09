//! Almacenamiento de la carpeta de modelos (ADR 0034): qué ocupa cada cosa, cuánto queda libre en
//! el volumen, una reserva que las descargas no pueden comerse y la limpieza de descargas a medias
//! viejas.
//!
//! Reglas:
//! - Lo que no se reconoce solo se informa; nunca se borra.
//! - Lo único que borra este módulo son archivos `.part` (descargas a medias, ADR 0020) más viejos
//!   que el umbral, y las carpetas que quedan vacías después. Borrar un modelo entero es `brasa rm`
//!   (o `DELETE /api/storage/models/{name}`), que pasa por [`crate::local::resolve_child`].
//! - El recorrido no sigue symlinks (ni los cuenta). Antes de borrar se vuelve a mirar el archivo:
//!   tiene que ser regular, seguir siendo viejo y su ruta real tiene que ser exactamente la que se
//!   recorrió dentro de la carpeta de modelos. Si la carpeta de modelos misma pasa por un symlink,
//!   la limpieza se niega salvo que se pida seguirlo (la misma regla que `brasa rm`).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::Serialize;

use crate::dirs::ModelsDir;
use crate::manifest::FileSpec;
use crate::{Error, Result};

pub const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
/// Reserva por defecto: lo que una descarga tiene que dejar libre en el volumen.
pub const DEFAULT_RESERVE_BYTES: u64 = 2 * GIB;
/// Una descarga a medias con más días que esto se considera vieja (la limpia `storage clean`).
pub const DEFAULT_PARTIAL_MAX_AGE_DAYS: u64 = 7;
/// Tope de la reserva configurable, para atajar errores de tipeo (por ejemplo, bytes en vez de GiB).
pub const MAX_RESERVE_GIB: f64 = 1024.0;
/// Sufijo de los archivos de una descarga a medias (`pull::download_file`).
pub const PART_SUFFIX: &str = ".part";
const DAY: u64 = 86_400;

/// Ajustes de almacenamiento (sección `[storage]` del archivo de configuración).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Settings {
    /// Bytes del volumen que una descarga no puede usar.
    pub reserve_bytes: u64,
    /// Días a partir de los cuales una descarga a medias es vieja.
    pub partial_max_age_days: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            reserve_bytes: DEFAULT_RESERVE_BYTES,
            partial_max_age_days: DEFAULT_PARTIAL_MAX_AGE_DAYS,
        }
    }
}

impl Settings {
    /// Valida los valores del archivo (`None`: el valor por defecto).
    pub fn new(reserve_gib: Option<f64>, partial_max_age_days: Option<u64>) -> Result<Self> {
        let mut s = Self::default();
        if let Some(g) = reserve_gib {
            if !g.is_finite() || !(0.0..=MAX_RESERVE_GIB).contains(&g) {
                return Err(Error(format!(
                    "storage.reserve_gib = {g}: tiene que estar entre 0 y {MAX_RESERVE_GIB} (GiB)"
                )));
            }
            s.reserve_bytes = (g * GIB as f64).round() as u64;
        }
        if let Some(d) = partial_max_age_days {
            // Una descarga en curso escribe su `.part` todo el tiempo: con al menos un día de
            // umbral, la limpieza nunca la toca aunque no sepa que está corriendo.
            if !(1..=36_500).contains(&d) {
                return Err(Error(format!(
                    "storage.partial_max_age_days = {d}: tiene que estar entre 1 y 36500 días"
                )));
            }
            s.partial_max_age_days = d;
        }
        Ok(s)
    }

    pub fn reserve_gib(&self) -> f64 {
        self.reserve_bytes as f64 / GIB as f64
    }
}

/// Bytes en forma legible (`2.21 GiB`, `14.0 MiB`, `9.8 KiB`, `512 B`).
pub fn human(b: u64) -> String {
    if b >= GIB {
        format!("{:.2} GiB", b as f64 / GIB as f64)
    } else if b >= MIB {
        format!("{:.1} MiB", b as f64 / MIB as f64)
    } else if b >= 1024 {
        format!("{:.1} KiB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

/// Antigüedad en forma legible (`3 días`, `5 h`, `12 min`).
pub fn human_age(secs: u64) -> String {
    match secs {
        s if s >= 2 * DAY => format!("{} días", s / DAY),
        s if s >= DAY => "1 día".into(),
        s if s >= 3600 => format!("{} h", s / 3600),
        s if s >= 60 => format!("{} min", s / 60),
        s => format!("{s} s"),
    }
}

/// Tamaño y espacio libre del volumen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Volume {
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// Volumen de `path`, o del ancestro existente más cercano si `path` todavía no existe (la carpeta
/// predeterminada se crea con la primera descarga). `None` si no se puede leer.
pub fn volume(path: &Path) -> Option<Volume> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    let mut cur: &Path = &abs;
    loop {
        if cur.exists() {
            let (total_bytes, free_bytes) = crate::dirs::volume_bytes(cur)?;
            return Some(Volume {
                total_bytes,
                free_bytes,
            });
        }
        cur = cur.parent()?;
    }
}

/// Mensaje de error si `need` bytes no entran en `free` sin tocar la reserva; `None` si entran.
pub fn space_error(dir: &Path, need: u64, free: u64, reserve: u64) -> Option<String> {
    let available = free.saturating_sub(reserve);
    if need <= available {
        return None;
    }
    Some(format!(
        "no hay espacio para la descarga en {}: hacen falta {}; hay {} libres y la reserva que \
         las descargas no usan es de {}, así que quedan {} disponibles; faltan {}. Liberá espacio \
         (`brasa storage clean`, `brasa rm <modelo>`) o bajá la reserva (`reserve_gib` en la \
         sección [storage] de la configuración)",
        dir.display(),
        human(need),
        human(free),
        human(reserve),
        human(available),
        human(need - available)
    ))
}

/// Chequeo previo a descargar `need` bytes en `dir`. Si el volumen no se puede leer, no bloquea
/// (la descarga fallaría igual al escribir, con el error del sistema).
pub fn check_space(dir: &Path, need: u64, settings: &Settings) -> Result<()> {
    match volume(dir) {
        Some(v) => match space_error(dir, need, v.free_bytes, settings.reserve_bytes) {
            Some(msg) => Err(Error(msg)),
            None => Ok(()),
        },
        None => Ok(()),
    }
}

fn len_of(p: &Path) -> u64 {
    std::fs::metadata(p).map_or(0, |m| m.len())
}

/// Bytes ya en disco en `dest` de una descarga de `files` (archivos completos y `.part`).
pub fn bytes_on_disk(dest: &Path, files: &[FileSpec]) -> u64 {
    files.iter().map(|f| on_disk(dest, f)).sum()
}

fn on_disk(dest: &Path, f: &FileSpec) -> u64 {
    let done = len_of(&dest.join(&f.path));
    let part = len_of(&dest.join(format!("{}{PART_SUFFIX}", f.path)));
    done.max(part)
}

/// Bytes que faltan bajar para completar `files` en `dest` (lo que ya está no se vuelve a pedir).
pub fn remaining_bytes(dest: &Path, files: &[FileSpec]) -> u64 {
    files
        .iter()
        .map(|f| f.size.unwrap_or(0).saturating_sub(on_disk(dest, f)))
        .sum()
}

fn is_part(p: &Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.len() > PART_SUFFIX.len() && n.ends_with(PART_SUFFIX))
}

/// Recorrido sin seguir symlinks: bytes de los archivos regulares que no son `.part` y la lista de
/// `.part` con su tamaño y fecha de modificación.
#[derive(Debug, Default)]
struct Walk {
    bytes: u64,
    parts: Vec<(PathBuf, u64, SystemTime)>,
}

fn walk(dir: &Path, out: &mut Walk) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let Ok(md) = std::fs::symlink_metadata(&p) else {
            continue;
        };
        let ft = md.file_type();
        if ft.is_symlink() {
            continue;
        }
        if ft.is_dir() {
            walk(&p, out);
        } else if ft.is_file() {
            if is_part(&p) {
                let mtime = md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                out.parts.push((p, md.len(), mtime));
            } else {
                out.bytes += md.len();
            }
        }
    }
}

/// Bytes de los archivos regulares bajo `dir` (incluidos los `.part`), sin seguir symlinks.
pub fn dir_bytes(dir: &Path) -> u64 {
    let mut w = Walk::default();
    walk(dir, &mut w);
    w.bytes + w.parts.iter().map(|p| p.1).sum::<u64>()
}

fn age_secs(now: SystemTime, mtime: SystemTime) -> u64 {
    now.duration_since(mtime).map_or(0, |d| d.as_secs())
}

fn rel(base: &Path, p: &Path) -> String {
    p.strip_prefix(base).unwrap_or(p).display().to_string()
}

/// Modelo completo (carpeta con `model.brasa`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelEntry {
    pub name: String,
    /// Todo lo de la carpeta menos los `.part` (que van en `partials`).
    pub bytes: u64,
}

/// Archivo de una descarga a medias.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Partial {
    /// Ruta relativa a la carpeta de modelos.
    pub path: String,
    /// Entrada de primer nivel que lo contiene (normalmente el nombre del modelo).
    pub entry: String,
    pub bytes: u64,
    pub age_secs: u64,
    /// Más viejo que el umbral: `storage clean` lo borraría.
    pub old: bool,
}

/// Lo que no es un modelo completo: se informa, nunca se borra.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OtherKind {
    /// Safetensors de origen de un manifiesto (`pull --desde-fuente`, carpeta `hf_dir`).
    Source,
    /// Carpeta de una descarga que no terminó (lo ya bajado; los `.part` van aparte).
    Download,
    Dir,
    File,
    /// Symlink: no se sigue ni se cuenta.
    Symlink,
    Other,
}

impl OtherKind {
    pub fn describe(self) -> &'static str {
        match self {
            OtherKind::Source => "safetensors de origen",
            OtherKind::Download => "descarga sin terminar",
            OtherKind::Dir => "carpeta no reconocida",
            OtherKind::File => "archivo no reconocido",
            OtherKind::Symlink => "symlink (no se sigue)",
            OtherKind::Other => "no reconocido",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Other {
    pub name: String,
    pub kind: OtherKind,
    pub label: &'static str,
    pub bytes: u64,
    /// Destino, si es un symlink.
    pub target: Option<String>,
}

/// Inventario de la carpeta de modelos y del volumen.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub dir: String,
    pub dir_source: &'static str,
    pub dir_source_label: &'static str,
    pub exists: bool,
    pub volume: Option<Volume>,
    pub reserve_bytes: u64,
    /// Libre menos la reserva: lo que puede usar una descarga.
    pub available_bytes: Option<u64>,
    pub partial_max_age_days: u64,
    pub models: Vec<ModelEntry>,
    pub partials: Vec<Partial>,
    pub other: Vec<Other>,
    pub models_bytes: u64,
    pub partial_bytes: u64,
    pub old_partial_bytes: u64,
    pub other_bytes: u64,
}

/// Inventario de `md` sin seguir symlinks. `sources` son las carpetas de safetensors de origen
/// del catálogo (`hf_dir` de cada manifiesto), que se informan como tales.
pub fn report(md: &ModelsDir, settings: &Settings, sources: &[String], now: SystemTime) -> Report {
    let base = md.path.as_path();
    let max_age = settings.partial_max_age_days * DAY;
    let mut models = Vec::new();
    let mut partials = Vec::new();
    let mut other = Vec::new();
    let mut push_parts = |entry: &str, parts: Vec<(PathBuf, u64, SystemTime)>| {
        for (p, bytes, mtime) in parts {
            let age = age_secs(now, mtime);
            partials.push(Partial {
                path: rel(base, &p),
                entry: entry.to_string(),
                bytes,
                age_secs: age,
                old: age >= max_age,
            });
        }
    };
    if let Ok(rd) = std::fs::read_dir(base) {
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if name == ".DS_Store" {
                continue;
            }
            let Ok(meta) = std::fs::symlink_metadata(&p) else {
                continue;
            };
            let ft = meta.file_type();
            if ft.is_symlink() {
                let target = std::fs::read_link(&p).ok().map(|t| t.display().to_string());
                other.push(Other {
                    name,
                    kind: OtherKind::Symlink,
                    label: OtherKind::Symlink.describe(),
                    bytes: 0,
                    target,
                });
            } else if ft.is_file() {
                if is_part(&p) {
                    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    push_parts(&name, vec![(p, meta.len(), mtime)]);
                } else {
                    other.push(Other {
                        name,
                        kind: OtherKind::File,
                        label: OtherKind::File.describe(),
                        bytes: meta.len(),
                        target: None,
                    });
                }
            } else if ft.is_dir() {
                let mut w = Walk::default();
                walk(&p, &mut w);
                let has_parts = !w.parts.is_empty();
                push_parts(&name, std::mem::take(&mut w.parts));
                let weights = std::fs::symlink_metadata(p.join("model.brasa"));
                if weights.is_ok_and(|m| m.is_file()) {
                    models.push(ModelEntry {
                        name,
                        bytes: w.bytes,
                    });
                } else {
                    let kind = if sources.contains(&name) {
                        OtherKind::Source
                    } else if has_parts {
                        OtherKind::Download
                    } else {
                        OtherKind::Dir
                    };
                    other.push(Other {
                        name,
                        kind,
                        label: kind.describe(),
                        bytes: w.bytes,
                        target: None,
                    });
                }
            } else {
                other.push(Other {
                    name,
                    kind: OtherKind::Other,
                    label: OtherKind::Other.describe(),
                    bytes: 0,
                    target: None,
                });
            }
        }
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    partials.sort_by(|a, b| a.path.cmp(&b.path));
    other.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));
    let vol = volume(base);
    Report {
        dir: base.display().to_string(),
        dir_source: md.source.as_str(),
        dir_source_label: md.source.describe(),
        exists: base.is_dir(),
        volume: vol,
        reserve_bytes: settings.reserve_bytes,
        available_bytes: vol.map(|v| v.free_bytes.saturating_sub(settings.reserve_bytes)),
        partial_max_age_days: settings.partial_max_age_days,
        models_bytes: models.iter().map(|m| m.bytes).sum(),
        partial_bytes: partials.iter().map(|p| p.bytes).sum(),
        old_partial_bytes: partials.iter().filter(|p| p.old).map(|p| p.bytes).sum(),
        other_bytes: other.iter().map(|o| o.bytes).sum(),
        models,
        partials,
        other,
    }
}

/// Qué limpiar y cómo.
#[derive(Debug, Clone)]
pub struct CleanOptions<'a> {
    /// Se borran los `.part` con al menos esta antigüedad.
    pub max_age_days: u64,
    /// `false`: solo informa qué borraría (dry-run).
    pub apply: bool,
    /// Permitir que la carpeta de modelos pase por un symlink (como `brasa rm
    /// --seguir-symlink-base`).
    pub follow_base_symlink: bool,
    /// Carpetas que no se tocan (por ejemplo, la de una descarga en curso).
    pub exclude: &'a [PathBuf],
    pub now: SystemTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CleanItem {
    /// Ruta relativa a la carpeta de modelos.
    pub path: String,
    pub bytes: u64,
    pub age_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Skipped {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CleanReport {
    pub dir: String,
    pub dry_run: bool,
    pub older_than_days: u64,
    /// Lo que se borró (o, en dry-run, lo que se borraría).
    pub items: Vec<CleanItem>,
    pub bytes: u64,
    /// Descargas a medias más nuevas que el umbral (quedan para reanudar).
    pub kept: usize,
    pub kept_bytes: u64,
    /// Lo que no se tocó por un motivo (descarga en curso, cambió, error al borrar).
    pub skipped: Vec<Skipped>,
    /// Carpetas que quedaron vacías y se borraron.
    pub removed_dirs: Vec<String>,
}

/// Borra (o con `apply = false`, lista) los `.part` de la carpeta de modelos más viejos que el
/// umbral. Ver las reglas del módulo.
pub fn clean_partials(base: &Path, opts: &CleanOptions<'_>) -> Result<CleanReport> {
    if opts.max_age_days == 0 {
        return Err(Error(
            "el umbral de limpieza tiene que ser de al menos 1 día".into(),
        ));
    }
    if !opts.follow_base_symlink {
        if let Some(link) = crate::local::primer_symlink_del_usuario(base) {
            let real = std::fs::canonicalize(base).unwrap_or_else(|_| link.clone());
            return Err(Error(format!(
                "la carpeta de modelos {} pasa por el symlink {} -> {}; la limpieza no lo sigue \
                 (usá --seguir-symlink-base para limpiar en la ruta real)",
                base.display(),
                link.display(),
                real.display()
            )));
        }
    }
    let mut out = CleanReport {
        dir: base.display().to_string(),
        dry_run: !opts.apply,
        older_than_days: opts.max_age_days,
        items: Vec::new(),
        bytes: 0,
        kept: 0,
        kept_bytes: 0,
        skipped: Vec::new(),
        removed_dirs: Vec::new(),
    };
    if !base.is_dir() {
        return Ok(out);
    }
    let base_canon =
        std::fs::canonicalize(base).map_err(|e| Error(format!("{}: {e}", base.display())))?;
    let exclude: Vec<PathBuf> = opts
        .exclude
        .iter()
        .map(|p| std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()))
        .collect();
    let max_age = opts.max_age_days * DAY;

    // Mismo recorrido que el inventario, desde la ruta real: sin seguir symlinks.
    let mut w = Walk::default();
    walk(&base_canon, &mut w);
    w.parts.sort_by(|a, b| a.0.cmp(&b.0));
    for (p, bytes, mtime) in w.parts {
        let path = rel(&base_canon, &p);
        let age = age_secs(opts.now, mtime);
        if age < max_age {
            out.kept += 1;
            out.kept_bytes += bytes;
            continue;
        }
        if exclude.iter().any(|ex| p.starts_with(ex)) {
            out.skipped.push(Skipped {
                path,
                reason: "descarga en curso".into(),
            });
            continue;
        }
        if opts.apply {
            if let Err(reason) = remove_part(&base_canon, &p, max_age, opts.now) {
                out.skipped.push(Skipped { path, reason });
                continue;
            }
            // Si la carpeta quedó vacía (y no es la carpeta de modelos), se borra; `remove_dir`
            // falla si tiene algo, así que nunca borra contenido.
            let mut dir = p.parent().map(Path::to_path_buf);
            while let Some(d) = dir {
                if d == base_canon || !d.starts_with(&base_canon) {
                    break;
                }
                let empty = std::fs::read_dir(&d).is_ok_and(|mut r| r.next().is_none());
                if !empty || std::fs::remove_dir(&d).is_err() {
                    break;
                }
                out.removed_dirs.push(rel(&base_canon, &d));
                dir = d.parent().map(Path::to_path_buf);
            }
        }
        out.bytes += bytes;
        out.items.push(CleanItem {
            path,
            bytes,
            age_secs: age,
        });
    }
    Ok(out)
}

/// Vuelve a verificar un `.part` justo antes de borrarlo.
fn remove_part(
    base_canon: &Path,
    p: &Path,
    max_age: u64,
    now: SystemTime,
) -> std::result::Result<(), String> {
    let meta = std::fs::symlink_metadata(p).map_err(|e| e.to_string())?;
    if !meta.file_type().is_file() || !is_part(p) {
        return Err("ya no es un archivo .part regular".into());
    }
    // Si se reanudó entre el recorrido y ahora, ya no es viejo.
    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    if age_secs(now, mtime) < max_age {
        return Err("se modificó hace poco (¿descarga reanudada?)".into());
    }
    let real = std::fs::canonicalize(p).map_err(|e| e.to_string())?;
    let expect = p
        .strip_prefix(base_canon)
        .map(|r| base_canon.join(r))
        .map_err(|_| "fuera de la carpeta de modelos".to_string())?;
    if real != expect || !real.starts_with(base_canon) {
        return Err("su ruta real sale de la carpeta de modelos".into());
    }
    std::fs::remove_file(&real).map_err(|e| e.to_string())
}

/// Umbral de [`CleanOptions`] como [`Duration`].
pub fn days(d: u64) -> Duration {
    Duration::from_secs(d * DAY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dirs::DirSource;

    fn md(p: &Path) -> ModelsDir {
        ModelsDir {
            path: p.to_path_buf(),
            source: DirSource::Default,
        }
    }

    fn write(p: &Path, n: usize) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, vec![7u8; n]).unwrap();
    }

    /// Le pone a `p` una fecha de modificación de hace `d` días.
    fn age(p: &Path, d: u64) {
        let f = std::fs::File::options().write(true).open(p).unwrap();
        f.set_modified(SystemTime::now() - days(d)).unwrap();
    }

    #[test]
    fn valida_los_ajustes() {
        let d = Settings::new(None, None).unwrap();
        assert_eq!(d.reserve_bytes, 2 * GIB);
        assert_eq!(d.partial_max_age_days, 7);
        let s = Settings::new(Some(0.5), Some(30)).unwrap();
        assert_eq!(s.reserve_bytes, GIB / 2);
        assert_eq!(s.partial_max_age_days, 30);
        assert_eq!(Settings::new(Some(0.0), None).unwrap().reserve_bytes, 0);
        for g in [-1.0, f64::NAN, f64::INFINITY, 5000.0] {
            assert!(Settings::new(Some(g), None).is_err(), "aceptó {g}");
        }
        assert!(Settings::new(None, Some(0)).is_err());
    }

    #[test]
    fn la_reserva_no_se_toca() {
        let d = Path::new("/m");
        // Entra justo: libre 10 GiB, reserva 2, hace falta 8.
        assert!(space_error(d, 8 * GIB, 10 * GIB, 2 * GIB).is_none());
        // Un byte más ya no.
        let e = space_error(d, 8 * GIB + 1, 10 * GIB, 2 * GIB).unwrap();
        assert!(e.contains("faltan 1 B"), "{e}");
        // Menos libre que la reserva: falta todo lo que se pide.
        let e = space_error(d, 3 * GIB, GIB, 2 * GIB).unwrap();
        assert!(e.contains("faltan 3.00 GiB"), "{e}");
        assert!(e.contains("reserva"), "{e}");
        // Sin nada que bajar, siempre entra.
        assert!(space_error(d, 0, 0, 2 * GIB).is_none());
    }

    #[test]
    fn check_space_usa_el_volumen_real() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Settings::default();
        assert!(check_space(tmp.path(), 1, &s).is_ok());
        // Una carpeta que todavía no existe usa el volumen de su ancestro.
        assert!(check_space(&tmp.path().join("a/b"), 1, &s).is_ok());
        let enorme = Settings::new(Some(MAX_RESERVE_GIB), None).unwrap();
        let v = volume(tmp.path()).unwrap();
        if v.free_bytes < enorme.reserve_bytes {
            let e = check_space(tmp.path(), 1, &enorme).unwrap_err();
            assert!(e.0.contains("no hay espacio"), "{}", e.0);
        }
    }

    #[test]
    fn bytes_en_disco_y_faltantes() {
        let tmp = tempfile::tempdir().unwrap();
        let files = vec![
            FileSpec {
                path: "tokenizer.json".into(),
                sha256: String::new(),
                size: Some(100),
            },
            FileSpec {
                path: "model.brasa".into(),
                sha256: String::new(),
                size: Some(1000),
            },
        ];
        assert_eq!(remaining_bytes(tmp.path(), &files), 1100);
        write(&tmp.path().join("tokenizer.json"), 100);
        write(&tmp.path().join("model.brasa.part"), 400);
        assert_eq!(bytes_on_disk(tmp.path(), &files), 500);
        assert_eq!(remaining_bytes(tmp.path(), &files), 600);
    }

    /// Carpeta de modelos de prueba: un modelo, una descarga a medias vieja, una nueva, safetensors
    /// de origen, un archivo suelto y un symlink a una carpeta de afuera con un `.part` viejo.
    fn fixture(tmp: &Path) -> PathBuf {
        let base = tmp.join("modelos");
        write(&base.join("m1/model.brasa"), 1000);
        write(&base.join("m1/tokenizer.json"), 10);
        write(&base.join("bajando/tokenizer.json"), 20);
        write(&base.join("bajando/model.brasa.part"), 300);
        age(&base.join("bajando/model.brasa.part"), 10);
        write(&base.join("reciente/model.brasa.part"), 50);
        write(&base.join("qwen-hf/model.safetensors"), 700);
        write(&base.join("suelto.gguf"), 40);
        let afuera = tmp.join("afuera");
        write(&afuera.join("x.part"), 999);
        age(&afuera.join("x.part"), 30);
        std::os::unix::fs::symlink(&afuera, base.join("enlace")).unwrap();
        // Un `.part` que es un symlink a un archivo de afuera: no se sigue ni se borra.
        std::os::unix::fs::symlink(afuera.join("x.part"), base.join("m1/trampa.part")).unwrap();
        base
    }

    #[test]
    fn inventario_sin_seguir_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        let base = fixture(tmp.path());
        let r = report(
            &md(&base),
            &Settings::default(),
            &["qwen-hf".to_string()],
            SystemTime::now(),
        );
        assert!(r.exists);
        assert_eq!(
            r.models,
            [ModelEntry {
                name: "m1".into(),
                bytes: 1010
            }]
        );
        let parts: Vec<(&str, bool)> = r
            .partials
            .iter()
            .map(|p| (p.path.as_str(), p.old))
            .collect();
        assert_eq!(
            parts,
            [
                ("bajando/model.brasa.part", true),
                ("reciente/model.brasa.part", false)
            ]
        );
        assert_eq!(r.partial_bytes, 350);
        assert_eq!(dir_bytes(&base.join("bajando")), 320);
        // El symlink `m1/trampa.part` no se cuenta.
        assert_eq!(dir_bytes(&base.join("m1")), 1010);
        assert_eq!(r.old_partial_bytes, 300);
        let kinds: Vec<(&str, OtherKind, u64)> = r
            .other
            .iter()
            .map(|o| (o.name.as_str(), o.kind, o.bytes))
            .collect();
        assert_eq!(
            kinds,
            [
                ("qwen-hf", OtherKind::Source, 700),
                ("suelto.gguf", OtherKind::File, 40),
                ("bajando", OtherKind::Download, 20),
                ("enlace", OtherKind::Symlink, 0),
                ("reciente", OtherKind::Download, 0),
            ]
        );
        assert!(r.volume.is_some());
        assert_eq!(
            r.available_bytes,
            r.volume
                .map(|v| v.free_bytes.saturating_sub(DEFAULT_RESERVE_BYTES))
        );
        // Una carpeta que no existe: inventario vacío, sin error.
        let r = report(
            &md(&tmp.path().join("no-existe")),
            &Settings::default(),
            &[],
            SystemTime::now(),
        );
        assert!(!r.exists && r.models.is_empty() && r.partials.is_empty());
    }

    #[test]
    fn limpieza_en_dry_run_no_borra() {
        let tmp = tempfile::tempdir().unwrap();
        let base = fixture(tmp.path());
        let opts = CleanOptions {
            max_age_days: 7,
            apply: false,
            follow_base_symlink: false,
            exclude: &[],
            now: SystemTime::now(),
        };
        let r = clean_partials(&base, &opts).unwrap();
        assert!(r.dry_run);
        let paths: Vec<&str> = r.items.iter().map(|i| i.path.as_str()).collect();
        assert_eq!(paths, ["bajando/model.brasa.part"]);
        assert_eq!(r.bytes, 300);
        assert_eq!((r.kept, r.kept_bytes), (1, 50));
        assert!(base.join("bajando/model.brasa.part").is_file());
    }

    #[test]
    fn limpieza_borra_solo_parciales_viejos_de_adentro() {
        let tmp = tempfile::tempdir().unwrap();
        let base = fixture(tmp.path());
        // Otra descarga vieja, sola en su carpeta: la carpeta queda vacía y se va.
        write(&base.join("abandonada/sub/model.brasa.part"), 5);
        age(&base.join("abandonada/sub/model.brasa.part"), 9);
        let opts = CleanOptions {
            max_age_days: 7,
            apply: true,
            follow_base_symlink: false,
            exclude: &[],
            now: SystemTime::now(),
        };
        let r = clean_partials(&base, &opts).unwrap();
        assert!(!r.dry_run);
        let paths: Vec<&str> = r.items.iter().map(|i| i.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "abandonada/sub/model.brasa.part",
                "bajando/model.brasa.part"
            ]
        );
        assert_eq!(r.removed_dirs, ["abandonada/sub", "abandonada"]);
        assert!(!base.join("bajando/model.brasa.part").exists());
        assert!(!base.join("abandonada").exists());
        // Lo demás queda: lo ya bajado, el parcial nuevo, el modelo, lo no reconocido y lo de
        // afuera (aunque sea un .part viejo alcanzable por symlinks).
        for p in [
            "bajando/tokenizer.json",
            "reciente/model.brasa.part",
            "m1/model.brasa",
            "qwen-hf/model.safetensors",
            "suelto.gguf",
        ] {
            assert!(base.join(p).is_file(), "borró {p}");
        }
        assert!(tmp.path().join("afuera/x.part").is_file());
        assert!(
            std::fs::symlink_metadata(base.join("m1/trampa.part"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[test]
    fn limpieza_respeta_la_descarga_en_curso() {
        let tmp = tempfile::tempdir().unwrap();
        let base = fixture(tmp.path());
        let exclude = [base.join("bajando")];
        let opts = CleanOptions {
            max_age_days: 7,
            apply: true,
            follow_base_symlink: false,
            exclude: &exclude,
            now: SystemTime::now(),
        };
        let r = clean_partials(&base, &opts).unwrap();
        assert!(r.items.is_empty());
        assert_eq!(r.skipped[0].path, "bajando/model.brasa.part");
        assert!(base.join("bajando/model.brasa.part").is_file());
    }

    #[test]
    fn limpieza_no_sigue_una_base_que_es_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let real = fixture(tmp.path());
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let mut opts = CleanOptions {
            max_age_days: 7,
            apply: true,
            follow_base_symlink: false,
            exclude: &[],
            now: SystemTime::now(),
        };
        let e = clean_partials(&link, &opts).unwrap_err();
        assert!(e.0.contains("symlink"), "{}", e.0);
        assert!(real.join("bajando/model.brasa.part").is_file());
        // Con el permiso explícito, limpia en la ruta real.
        opts.follow_base_symlink = true;
        let r = clean_partials(&link, &opts).unwrap();
        assert_eq!(r.items.len(), 1);
        assert!(!real.join("bajando/model.brasa.part").exists());
        // Umbral cero: rechazado (tocaría una descarga en curso).
        opts.max_age_days = 0;
        assert!(clean_partials(&link, &opts).is_err());
    }

    #[test]
    fn formatos_legibles() {
        assert_eq!(human(512), "512 B");
        assert_eq!(human(10_000), "9.8 KiB");
        assert_eq!(human(3 * MIB / 2), "1.5 MiB");
        assert_eq!(human(2 * GIB), "2.00 GiB");
        assert_eq!(human_age(30), "30 s");
        assert_eq!(human_age(DAY), "1 día");
        assert_eq!(human_age(9 * DAY + 5), "9 días");
    }
}
