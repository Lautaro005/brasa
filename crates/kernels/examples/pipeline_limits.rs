//! Límite de hilos por threadgroup de los kernels de atención y GEMM. Baja cuando un kernel usa
//! muchos registros, así que sirve para detectar presión de registros:
//!   cargo run --release -p brasa-kernels --example pipeline_limits
use brasa_kernels::sources::*;
use brasa_kernels::{KvType, kv_source};
use brasa_metal::Context;

fn main() {
    let ctx = Context::new().unwrap();
    let g4 = |src: &str| format!("#define GQA_G 4\n{src}");
    let mut cases = Vec::new();
    for kv in [KvType::F32, KvType::F16, KvType::Q8_0] {
        let n = kv.name();
        cases.push((
            format!("flash_attn_f32 kv {n}"),
            kv_source(kv, FLASH_ATTENTION),
            "flash_attn_f32",
        ));
        cases.push((
            format!("flash_attn_gqa G4 kv {n}"),
            kv_source(kv, &g4(FLASH_ATTENTION)),
            "flash_attn_gqa",
        ));
        cases.push((
            format!("attn_decode_lanes G4 kv {n}"),
            kv_source(kv, &g4(DECODE_ATTENTION)),
            "attn_decode_lanes",
        ));
    }
    cases.push((
        "gemm_tiled_q4_0".into(),
        MATMUL_TILED.to_string(),
        "gemm_tiled_q4_0_f32",
    ));
    for (name, src, f) in cases {
        let p = ctx.pipeline(&src, f).unwrap();
        println!(
            "{name:30} máx. hilos por threadgroup {}",
            p.max_threads_per_threadgroup()
        );
    }
}
