//! Fingerprint de la máquina para la base de tuning (ADR 0026).
//!
//! Solo entran datos que comparten todas las unidades del mismo modelo de chip y la misma versión
//! de software: nada de número de serie, UUID de plataforma, hostname ni usuario.

use crate::hardware::HardwareInfo;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Versión del formato canónico que se hashea. Cambiarla invalida todas las bases existentes.
pub const FINGERPRINT_VERSION: u32 = 1;

/// Lo que determina que un resultado de tuning siga siendo válido.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fingerprint {
    /// Nombre comercial del chip, por ejemplo "Apple M1 Pro".
    pub chip: String,
    /// Núcleos de GPU (`None` si IOKit no los informa).
    pub gpu_cores: Option<u32>,
    pub cpu_performance_cores: u32,
    pub cpu_efficiency_cores: u32,
    /// RAM física en bytes.
    pub memory_bytes: u64,
    /// Familia Metal más alta, por ejemplo "apple7".
    pub metal_family: Option<String>,
    /// Versión de macOS, por ejemplo "15.5".
    pub macos_version: String,
    /// Build de macOS, por ejemplo "24F74". Fija el compilador MSL y el driver de la GPU, que
    /// vienen con el sistema (ADR 0026).
    pub macos_build: String,
    /// Hash de las fuentes MSL embebidas ([`kernels_version`]).
    pub kernels_version: String,
}

impl Fingerprint {
    /// Fingerprint de la máquina actual con los kernels de este binario.
    pub fn current() -> Self {
        Self::from_hardware(&crate::hardware::hardware_info(), &kernels_version())
    }

    pub fn from_hardware(hw: &HardwareInfo, kernels_version: &str) -> Self {
        Self {
            chip: hw.chip.clone(),
            gpu_cores: hw.gpu_cores,
            cpu_performance_cores: hw.cpu_performance_cores,
            cpu_efficiency_cores: hw.cpu_efficiency_cores,
            memory_bytes: hw.memory_bytes,
            metal_family: hw.metal.as_ref().and_then(|m| m.apple_family.clone()),
            macos_version: hw.macos_version.clone(),
            macos_build: hw.macos_build.clone(),
            kernels_version: kernels_version.to_string(),
        }
    }

    /// Texto canónico que se hashea: una línea `campo=largo:valor` por campo, en orden fijo. El
    /// largo hace que ningún valor pueda imitar a otro campo.
    pub fn canonical(&self) -> String {
        let opt_u32 = |v: Option<u32>| v.map_or_else(|| "-".to_string(), |n| n.to_string());
        let fields: [(&str, String); 9] = [
            ("chip", self.chip.clone()),
            ("gpu_cores", opt_u32(self.gpu_cores)),
            (
                "cpu_performance_cores",
                self.cpu_performance_cores.to_string(),
            ),
            (
                "cpu_efficiency_cores",
                self.cpu_efficiency_cores.to_string(),
            ),
            ("memory_bytes", self.memory_bytes.to_string()),
            (
                "metal_family",
                self.metal_family.clone().unwrap_or_else(|| "-".to_string()),
            ),
            ("macos_version", self.macos_version.clone()),
            ("macos_build", self.macos_build.clone()),
            ("kernels_version", self.kernels_version.clone()),
        ];
        let mut s = format!("brasa-fingerprint v{FINGERPRINT_VERSION}\n");
        for (k, v) in fields {
            s.push_str(&format!("{k}={}:{v}\n", v.len()));
        }
        s
    }

    /// Identificador estable: los primeros 16 dígitos hex del SHA-256 de [`Self::canonical`].
    /// Se usa como nombre del archivo de la base.
    pub fn id(&self) -> String {
        hex16(Sha256::digest(self.canonical().as_bytes()).as_slice())
    }
}

/// Versión de los kernels: los primeros 16 dígitos hex del SHA-256 de las fuentes MSL embebidas
/// (nombre y contenido de cada una, en el orden de `brasa_kernels::sources::ALL`).
pub fn kernels_version() -> String {
    let mut h = Sha256::new();
    for (name, src) in brasa_kernels::sources::ALL {
        h.update((name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update((src.len() as u64).to_le_bytes());
        h.update(src.as_bytes());
    }
    hex16(h.finalize().as_slice())
}

fn hex16(bytes: &[u8]) -> String {
    bytes[..8].iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn sample() -> Fingerprint {
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

    #[test]
    fn hash_estable() {
        // Valor fijo (verificado con `shasum -a 256` sobre el texto canónico): si cambia, cambió el
        // formato canónico y hay que subir FINGERPRINT_VERSION.
        assert_eq!(sample().id(), "7726bfac4fc995ef");
        assert_eq!(sample().id(), sample().id());
        assert_eq!(sample().id().len(), 16);
    }

    #[test]
    fn cada_campo_cambia_el_hash() {
        let base = sample().id();
        let variantes: [fn(&mut Fingerprint); 10] = [
            |f| f.chip = "Apple M2".into(),
            |f| f.gpu_cores = Some(16),
            |f| f.gpu_cores = None,
            |f| f.cpu_performance_cores = 8,
            |f| f.cpu_efficiency_cores = 4,
            |f| f.memory_bytes = 8 << 30,
            |f| f.metal_family = Some("apple8".into()),
            |f| f.macos_version = "15.6".into(),
            |f| f.macos_build = "24G84".into(),
            |f| f.kernels_version = "fedcba9876543210".into(),
        ];
        for cambiar in variantes {
            let mut f = sample();
            cambiar(&mut f);
            assert_ne!(f.id(), base, "{f:?}");
        }
    }

    #[test]
    fn valores_no_imitan_otros_campos() {
        let mut a = sample();
        a.macos_version = "15.5\nmacos_build=5:24F74".into();
        a.macos_build = String::new();
        assert_ne!(a.id(), sample().id());
    }

    #[test]
    fn serde_roundtrip() {
        let f = sample();
        let s = serde_json::to_string(&f).unwrap();
        assert_eq!(serde_json::from_str::<Fingerprint>(&s).unwrap(), f);
    }

    #[test]
    fn sin_identificadores_de_la_unidad() {
        let s = serde_json::to_string(&Fingerprint::current())
            .unwrap()
            .to_lowercase();
        for prohibido in ["serial", "uuid", "host", "user"] {
            assert!(!s.contains(prohibido), "{prohibido} en {s}");
        }
    }

    #[test]
    fn kernels_version_estable_y_completa() {
        assert_eq!(kernels_version(), kernels_version());
        assert_eq!(kernels_version().len(), 16);
        // Toda fuente .metal del crate de kernels tiene que estar en sources::ALL.
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels/src/metal");
        let mut en_disco: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".metal"))
            .collect();
        en_disco.sort();
        let mut listadas: Vec<String> = brasa_kernels::sources::ALL
            .iter()
            .map(|(n, _)| n.to_string())
            .collect();
        listadas.sort();
        assert_eq!(en_disco, listadas);
    }

    #[test]
    fn current_en_apple_silicon() {
        let f = Fingerprint::current();
        assert!(f.chip.starts_with("Apple M"), "{f:?}");
        assert_eq!(f.kernels_version, kernels_version());
        assert_eq!(f, Fingerprint::current());
    }
}
