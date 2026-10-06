//! Modelos conocidos por el harness y dónde están sus pesos para cada engine.
//!
//! Provisorio hasta que exista `brasa-catalog`: rutas relativas a la raíz del repo, generadas
//! por `tools/make_gguf.sh` y `mlx_lm.convert` (ver ADR 0002).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    pub name: &'static str,
    pub source_repo: &'static str,
    pub source_commit: &'static str,
    pub gguf_path: &'static str,
    pub gguf_quant: &'static str,
    pub mlx_path: &'static str,
    pub mlx_quant: &'static str,
    pub brasa_dir: &'static str,
    pub brasa_quant: &'static str,
}

pub const MODELS: &[ModelSpec] = &[ModelSpec {
    name: "qwen3-4b-q4",
    source_repo: "Qwen/Qwen3-4B",
    source_commit: "1cfa9a7208912126459214e8b04321603b3df60c",
    gguf_path: "models/qwen3-4b-q4_0.gguf",
    gguf_quant: "Q4_0 (bloques de 32, escala FP16)",
    mlx_path: "models/qwen3-4b-mlx-q4g32",
    mlx_quant: "MLX 4 bits, grupos de 32 (escala y bias BF16)",
    brasa_dir: "models/qwen3-4b-q4",
    brasa_quant: "q4_0 g32 + q6_0 embeddings (.brasa, ADR 0006 y 0012)",
}];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.name == name)
}
