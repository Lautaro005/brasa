//! Parámetros de lanzamiento de decode elegibles en tiempo de ejecución (ADR 0029).
//!
//! Solo entran parámetros que no cambian los resultados: cuántos simdgroups van en cada
//! threadgroup de los GEMV de decode y de `attn_decode_lanes` (constante de compilación; las
//! variantes que no son la de por defecto las compila `Kernels::set_launch` al cargar). Cada fila (GEMV) o tramo de claves
//! (atención) lo calcula un simdgroup con el mismo código cualquiera sea ese número, así que la
//! salida es la misma bit a bit (lo verifican los tests de `tests/launch.rs`). Sin entradas, los
//! valores son los fijados a mano en la M1 Pro.
//!
//! La consulta corre en cada dispatch del loop de decode: recorre vectores armados al cargar, sin
//! asignar (regla 4).

use crate::{KvType, WeightType};

/// Filas por simdgroup de los GEMV rápidos (`NR` en `matmul.metal`).
pub const GEMV_NR: usize = 4;
/// Simdgroups por threadgroup de los GEMV rápidos sin base de tuning.
pub const GEMV_SG_DEFAULT: usize = 2;
/// Simdgroups (tramos) por threadgroup de `attn_decode_lanes` sin base de tuning.
pub const LANES_SG_DEFAULT: usize = 4;
/// Valores admitidos para los dos parámetros (hasta 256 hilos por threadgroup).
pub const SG_CANDIDATES: [usize; 4] = [1, 2, 4, 8];

/// GEMV de decode cuyo lanzamiento se puede elegir.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvOp {
    /// `gemv_fast_*` (o, down y el lm_head q8_0).
    Fast(WeightType),
    /// `gemv_scaled_*` (lm_head q6_0).
    Scaled(WeightType),
    /// `gemv_scaled3_q4_0` (q, k y v en un dispatch; `rows` es la suma de las tres).
    Scaled3,
    /// `gemv_scaled_swiglu_q4_0` (gate, up y SwiGLU).
    ScaledSwiglu,
}

impl GemvOp {
    /// Nombre de la función MSL (clave `kernel` de la base de tuning).
    pub fn kernel_name(self) -> &'static str {
        match self {
            GemvOp::Fast(WeightType::Q4_0) => "gemv_fast_q4_0_f32",
            GemvOp::Fast(WeightType::Q8_0) => "gemv_fast_q8_0_f32",
            GemvOp::Fast(WeightType::Q6_0) => "gemv_fast_q6_0_f32",
            GemvOp::Scaled(WeightType::Q4_0) => "gemv_scaled_q4_0_f32",
            GemvOp::Scaled(WeightType::Q6_0) => "gemv_scaled_q6_0_f32",
            GemvOp::Scaled(WeightType::Q8_0) => "gemv_scaled_q8_0_f32",
            GemvOp::Scaled3 => "gemv_scaled3_q4_0_f32",
            GemvOp::ScaledSwiglu => "gemv_scaled_swiglu_q4_0_f32",
        }
    }

    /// Inverso de [`GemvOp::kernel_name`] (solo las variantes que existen).
    pub fn from_kernel_name(name: &str) -> Option<Self> {
        Some(match name {
            "gemv_fast_q4_0_f32" => GemvOp::Fast(WeightType::Q4_0),
            "gemv_fast_q8_0_f32" => GemvOp::Fast(WeightType::Q8_0),
            "gemv_fast_q6_0_f32" => GemvOp::Fast(WeightType::Q6_0),
            "gemv_scaled_q4_0_f32" => GemvOp::Scaled(WeightType::Q4_0),
            "gemv_scaled_q6_0_f32" => GemvOp::Scaled(WeightType::Q6_0),
            "gemv_scaled3_q4_0_f32" => GemvOp::Scaled3,
            "gemv_scaled_swiglu_q4_0_f32" => GemvOp::ScaledSwiglu,
            _ => return None,
        })
    }
}

/// Error al armar un [`Launch`] con un valor no admitido.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchError(pub String);

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LaunchError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GemvEntry {
    op: GemvOp,
    rows: usize,
    cols: usize,
    sg: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LanesEntry {
    kv: KvType,
    group: usize,
    lk: usize,
    sg: usize,
}

/// Parámetros de lanzamiento de decode. `Launch::default()` reproduce los valores de siempre.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Launch {
    gemv: Vec<GemvEntry>,
    lanes: Vec<LanesEntry>,
}

fn check_sg(sg: usize) -> Result<(), LaunchError> {
    if SG_CANDIDATES.contains(&sg) {
        Ok(())
    } else {
        Err(LaunchError(format!(
            "simdgroups por threadgroup {sg}: se admite {SG_CANDIDATES:?}"
        )))
    }
}

impl Launch {
    /// Fija los simdgroups por threadgroup del GEMV `op` con forma `[rows, cols]`. `rows` tiene
    /// que ser múltiplo de las filas por threadgroup (`4 · sg`).
    pub fn set_gemv(
        &mut self,
        op: GemvOp,
        rows: usize,
        cols: usize,
        sg: usize,
    ) -> Result<(), LaunchError> {
        check_sg(sg)?;
        if rows == 0 || rows % (GEMV_NR * sg) != 0 {
            return Err(LaunchError(format!(
                "{}: {rows} filas no son múltiplo de {} (sg {sg})",
                op.kernel_name(),
                GEMV_NR * sg
            )));
        }
        let e = GemvEntry { op, rows, cols, sg };
        match self
            .gemv
            .iter_mut()
            .find(|g| g.op == op && g.rows == rows && g.cols == cols)
        {
            Some(g) => *g = e,
            None => self.gemv.push(e),
        }
        Ok(())
    }

    /// Fija los tramos por threadgroup de `attn_decode_lanes` para la KV `kv`, el grupo GQA
    /// `group`, medidos con `lk` claves. Para otra longitud se usa la entrada de `lk` más cercana
    /// en escala logarítmica.
    pub fn set_attn_lanes(
        &mut self,
        kv: KvType,
        group: usize,
        lk: usize,
        sg: usize,
    ) -> Result<(), LaunchError> {
        check_sg(sg)?;
        if lk == 0 {
            return Err(LaunchError("attn_decode_lanes: lk 0".into()));
        }
        let e = LanesEntry { kv, group, lk, sg };
        match self
            .lanes
            .iter_mut()
            .find(|l| l.kv == kv && l.group == group && l.lk == lk)
        {
            Some(l) => *l = e,
            None => self.lanes.push(e),
        }
        Ok(())
    }

    /// Simdgroups por threadgroup para el GEMV `op` `[rows, cols]` (por defecto 2).
    pub fn gemv_sg(&self, op: GemvOp, rows: usize, cols: usize) -> usize {
        self.gemv
            .iter()
            .find(|g| g.op == op && g.rows == rows && g.cols == cols)
            .map_or(GEMV_SG_DEFAULT, |g| g.sg)
    }

    /// Tramos por threadgroup de `attn_decode_lanes` con `lk` claves (por defecto 4).
    pub fn attn_lanes_sg(&self, kv: KvType, group: usize, lk: usize) -> usize {
        let dist = |e: &LanesEntry| (e.lk.max(1) as f64 / lk.max(1) as f64).ln().abs();
        self.lanes
            .iter()
            .filter(|e| e.kv == kv && e.group == group)
            .min_by(|a, b| dist(a).total_cmp(&dist(b)))
            .map_or(LANES_SG_DEFAULT, |e| e.sg)
    }

    /// (GEMV, simdgroups) de cada entrada de GEMV.
    pub fn gemv_entries(&self) -> impl Iterator<Item = (GemvOp, usize)> + '_ {
        self.gemv.iter().map(|g| (g.op, g.sg))
    }

    /// (KV, grupo GQA, simdgroups) de cada entrada de atención.
    pub fn lanes_entries(&self) -> impl Iterator<Item = (KvType, usize, usize)> + '_ {
        self.lanes.iter().map(|l| (l.kv, l.group, l.sg))
    }

    /// Cantidad de entradas (GEMV más atención).
    pub fn len(&self) -> usize {
        self.gemv.len() + self.lanes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn por_defecto_los_valores_de_siempre() {
        let l = Launch::default();
        assert!(l.is_empty());
        assert_eq!(l.gemv_sg(GemvOp::Scaled3, 6144, 2560), 2);
        assert_eq!(l.attn_lanes_sg(KvType::F16, 4, 2048), 4);
    }

    #[test]
    fn consulta_por_forma_exacta() {
        let mut l = Launch::default();
        l.set_gemv(GemvOp::Fast(WeightType::Q4_0), 2560, 9728, 8)
            .unwrap();
        assert_eq!(l.gemv_sg(GemvOp::Fast(WeightType::Q4_0), 2560, 9728), 8);
        assert_eq!(l.gemv_sg(GemvOp::Fast(WeightType::Q4_0), 2560, 4096), 2);
        assert_eq!(l.gemv_sg(GemvOp::Fast(WeightType::Q8_0), 2560, 9728), 2);
        // Reemplaza, no duplica.
        l.set_gemv(GemvOp::Fast(WeightType::Q4_0), 2560, 9728, 1)
            .unwrap();
        assert_eq!(l.len(), 1);
        assert_eq!(l.gemv_sg(GemvOp::Fast(WeightType::Q4_0), 2560, 9728), 1);
    }

    #[test]
    fn rechaza_valores_invalidos() {
        let mut l = Launch::default();
        assert!(l.set_gemv(GemvOp::Scaled3, 6144, 2560, 3).is_err());
        assert!(l.set_gemv(GemvOp::Scaled3, 6144, 2560, 16).is_err());
        // 1000 filas no son múltiplo de 32.
        assert!(l.set_gemv(GemvOp::ScaledSwiglu, 1000, 2560, 8).is_err());
        assert!(l.set_attn_lanes(KvType::F16, 4, 2048, 0).is_err());
        assert!(l.is_empty());
    }

    #[test]
    fn atencion_usa_la_longitud_mas_cercana() {
        let mut l = Launch::default();
        l.set_attn_lanes(KvType::F16, 4, 2048, 2).unwrap();
        l.set_attn_lanes(KvType::F16, 4, 16384, 8).unwrap();
        assert_eq!(l.attn_lanes_sg(KvType::F16, 4, 100), 2);
        assert_eq!(l.attn_lanes_sg(KvType::F16, 4, 5000), 2);
        assert_eq!(l.attn_lanes_sg(KvType::F16, 4, 6000), 8);
        assert_eq!(l.attn_lanes_sg(KvType::F16, 4, 40000), 8);
        // Otra KV u otro grupo: valor por defecto.
        assert_eq!(l.attn_lanes_sg(KvType::Q8_0, 4, 2048), 4);
        assert_eq!(l.attn_lanes_sg(KvType::F16, 8, 2048), 4);
    }

    #[test]
    fn nombres_de_kernel_ida_y_vuelta() {
        for op in [
            GemvOp::Fast(WeightType::Q4_0),
            GemvOp::Fast(WeightType::Q8_0),
            GemvOp::Fast(WeightType::Q6_0),
            GemvOp::Scaled(WeightType::Q4_0),
            GemvOp::Scaled(WeightType::Q6_0),
            GemvOp::Scaled3,
            GemvOp::ScaledSwiglu,
        ] {
            assert_eq!(GemvOp::from_kernel_name(op.kernel_name()), Some(op));
        }
        assert_eq!(GemvOp::from_kernel_name("gemm_tiled_q4_0_f32"), None);
    }
}
