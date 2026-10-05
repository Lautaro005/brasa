//! Límite de hilos por threadgroup de los kernels de atención y GEMM. Baja cuando un kernel usa
//! muchos registros, así que sirve para detectar presión de registros:
//!   cargo run --release -p brasa-kernels --example pipeline_limits
use brasa_kernels::sources::*;
use brasa_metal::Context;

fn main() {
    let ctx = Context::new().unwrap();
    let f16 = "#define KV_F16 1\n";
    for (name, src, f) in [
        (
            "flash_attn_f32 kv f32",
            FLASH_ATTENTION.to_string(),
            "flash_attn_f32",
        ),
        (
            "flash_attn_f32 kv f16",
            format!("{f16}{FLASH_ATTENTION}"),
            "flash_attn_f32",
        ),
        (
            "flash_attn_gqa G4 kv f32",
            format!("#define GQA_G 4\n{FLASH_ATTENTION}"),
            "flash_attn_gqa",
        ),
        (
            "flash_attn_gqa G4 kv f16",
            format!("#define GQA_G 4\n{f16}{FLASH_ATTENTION}"),
            "flash_attn_gqa",
        ),
        (
            "attn_decode_lanes G4 kv f16",
            format!("#define GQA_G 4\n{f16}{DECODE_ATTENTION}"),
            "attn_decode_lanes",
        ),
        (
            "gemm_tiled_q4_0",
            MATMUL_TILED.to_string(),
            "gemm_tiled_q4_0_f32",
        ),
    ] {
        let p = ctx.pipeline(&src, f).unwrap();
        println!(
            "{name:30} máx. hilos por threadgroup {}",
            p.max_threads_per_threadgroup()
        );
    }
}
