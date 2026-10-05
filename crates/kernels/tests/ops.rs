//! Equivalencia GPU vs CPU de normas, softmax, RoPE, embedding y matmul cuantizado
//! (tolerancias en `brasa_kernels`). Formas de Qwen3-4B: H = 2560, FFN = 9728, head_dim = 128.

use brasa_kernels::testutil::{Rng, max_rel};
use brasa_kernels::{Kernels, QMatrix, RopeTable, WeightType, reference};
use brasa_metal::{Arg, Context};
use brasa_quant::QType;

fn setup() -> (Context, Kernels) {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    (ctx, k)
}

#[test]
fn rms_norm() {
    let (ctx, k) = setup();
    let mut rng = Rng::new(11);
    // (filas, n): estado oculto en decode y prefill, y QK-norm por cabeza.
    let mut worst = 0f32;
    for (rows, n) in [(1, 2560), (37, 2560), (64 * 32, 128), (3, 100)] {
        let x = rng.vec(rows * n, 30.0);
        let w = rng.vec(n, 2.0);
        let mut expected = vec![0.0; rows * n];
        reference::rms_norm(&x, &w, n, 1e-6, &mut expected);
        let (gx, gw) = (ctx.buffer_from(&x).unwrap(), ctx.buffer_from(&w).unwrap());
        let mut out = ctx.buffer::<f32>(rows * n).unwrap();
        let mut cmd = ctx.command().unwrap();
        k.rms_norm(
            &mut cmd,
            Arg::buf(&gx),
            Arg::buf(&gw),
            Arg::buf(&out),
            rows,
            n,
            1e-6,
        );
        cmd.commit_and_wait().unwrap();
        worst = worst.max(max_rel(out.as_mut_slice(), &expected, 1e-3));
    }
    eprintln!("rms_norm: error relativo máximo {worst:.2e}");
    assert!(worst <= 1e-5, "{worst}");
}

#[test]
fn softmax() {
    let (ctx, k) = setup();
    let mut rng = Rng::new(12);
    let mut worst = 0f32;
    for (rows, n) in [(1, 151_936), (32, 2048), (5, 3), (2, 1)] {
        let x = rng.vec(rows * n, 15.0);
        let mut expected = vec![0.0; rows * n];
        reference::softmax(&x, n, &mut expected);
        let gx = ctx.buffer_from(&x).unwrap();
        let mut out = ctx.buffer::<f32>(rows * n).unwrap();
        let mut cmd = ctx.command().unwrap();
        k.softmax(&mut cmd, &gx, &out, rows, n);
        cmd.commit_and_wait().unwrap();
        worst = worst.max(max_rel(out.as_mut_slice(), &expected, 1e-30));
    }
    eprintln!("softmax: error relativo máximo {worst:.2e}");
    assert!(worst <= 1e-5, "{worst}");
}

#[test]
fn rope_neox() {
    let (ctx, k) = setup();
    let mut rng = Rng::new(13);
    let (heads, dim, max_pos) = (32, 128, 40_960);
    let table = RopeTable::new(&ctx, 1_000_000.0, dim, max_pos).unwrap();
    let (cos, sin) = brasa_kernels::rope_table(1_000_000.0, dim, max_pos);
    let mut worst = 0f32;
    // Posiciones bajas y altas (cerca del máximo nativo de Qwen3).
    for (tokens, pos0) in [(1, 0), (7, 3), (16, 40_944)] {
        let x = rng.vec(tokens * heads * dim, 3.0);
        let mut expected = x.clone();
        reference::rope_neox(&mut expected, heads, dim, pos0, &cos, &sin);
        let mut gx = ctx.buffer_from(&x).unwrap();
        let mut cmd = ctx.command().unwrap();
        k.rope_neox(&mut cmd, Arg::buf(&gx), &table, tokens, heads, dim, pos0);
        cmd.commit_and_wait().unwrap();
        let got = gx.as_mut_slice();
        let half = dim / 2;
        for (v, (g, e)) in x.chunks(dim).zip(got.chunks(dim).zip(expected.chunks(dim))) {
            for i in 0..half {
                let scale = v[i].abs() + v[i + half].abs();
                for j in [i, i + half] {
                    worst = worst.max((g[j] - e[j]).abs() / scale.max(1e-30));
                }
            }
        }
    }
    eprintln!("rope: error máximo / (|a|+|b|) {worst:.2e}");
    assert!(worst <= 1e-6, "{worst}");
}

#[test]
fn embed_q6_0_exacto() {
    let (ctx, k) = setup();
    let mut rng = Rng::new(15);
    let (vocab, h) = (1000, 2560);
    let table = rng.q6_0(vocab, h);
    let ids: Vec<u32> = vec![0, 999, 17, 17, 500];
    let mut expected = vec![0.0; ids.len() * h];
    reference::embed(QType::Q6_0, &table, h, &ids, &mut expected);
    let gt = ctx.buffer_from(&table).unwrap();
    let gids = ctx.buffer_from(&ids).unwrap();
    let mut out = ctx.buffer::<f32>(ids.len() * h).unwrap();
    let w = QMatrix {
        data: &gt,
        qtype: WeightType::Q6_0,
        rows: vocab,
        cols: h,
    };
    let mut cmd = ctx.command().unwrap();
    k.embed(&mut cmd, w, &gids, Arg::buf(&out), ids.len());
    cmd.commit_and_wait().unwrap();
    assert_eq!(out.as_mut_slice(), expected.as_slice());
}

#[test]
fn embed_q8_0_exacto() {
    let (ctx, k) = setup();
    let mut rng = Rng::new(14);
    let (vocab, h) = (1000, 2560);
    let table = rng.q8_0(vocab, h);
    let ids: Vec<u32> = vec![0, 999, 17, 17, 500];
    let mut expected = vec![0.0; ids.len() * h];
    reference::embed(QType::Q8_0, &table, h, &ids, &mut expected);
    let gt = ctx.buffer_from(&table).unwrap();
    let gids = ctx.buffer_from(&ids).unwrap();
    let mut out = ctx.buffer::<f32>(ids.len() * h).unwrap();
    let w = QMatrix {
        data: &gt,
        qtype: WeightType::Q8_0,
        rows: vocab,
        cols: h,
    };
    let mut cmd = ctx.command().unwrap();
    k.embed(&mut cmd, w, &gids, Arg::buf(&out), ids.len());
    cmd.commit_and_wait().unwrap();
    assert_eq!(out.as_mut_slice(), expected.as_slice());
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    Gemv,
    GemvSimple,
    Naive,
    Tiled,
}

fn check_matmul(qtype: WeightType, path: Path, rows: usize, cols: usize, tokens: usize) -> f32 {
    let (ctx, k) = setup();
    let mut rng = Rng::new((rows * 31 + cols + tokens) as u64);
    let (w, q) = match qtype {
        WeightType::Q4_0 => (rng.q4_0(rows, cols), QType::Q4_0),
        WeightType::Q8_0 => (rng.q8_0(rows, cols), QType::Q8_0),
        WeightType::Q6_0 => (rng.q6_0(rows, cols), QType::Q6_0),
    };
    let x = rng.vec(tokens * cols, 4.0);
    let mut expected = vec![0.0; tokens * rows];
    let abs_sum = reference::matmul(q, &w, rows, cols, &x, &mut expected);
    let (gw, gx) = (ctx.buffer_from(&w).unwrap(), ctx.buffer_from(&x).unwrap());
    let mut y = ctx.buffer::<f32>(tokens * rows).unwrap();
    let m = QMatrix {
        data: &gw,
        qtype,
        rows,
        cols,
    };
    let mut cmd = ctx.command().unwrap();
    match path {
        Path::Gemv => k.gemv(&mut cmd, m, Arg::buf(&gx), Arg::buf(&y), tokens),
        Path::GemvSimple => k.gemv_simple(&mut cmd, m, Arg::buf(&gx), Arg::buf(&y), tokens),
        Path::Naive => k.gemm_naive(&mut cmd, m, Arg::buf(&gx), Arg::buf(&y), tokens),
        Path::Tiled => k.gemm(&mut cmd, m, Arg::buf(&gx), Arg::buf(&y), tokens),
    }
    cmd.commit_and_wait().unwrap();
    y.as_mut_slice()
        .iter()
        .zip(&expected)
        .zip(&abs_sum)
        .map(|((g, e), a)| (g - e).abs() / a.max(1e-30))
        .fold(0.0, f32::max)
}

#[test]
fn matmul_q4_0_y_q8_0() {
    let mut worst = 0f32;
    // (filas, columnas): q/o proj, gate/up, down, y una forma chica (sin tiled: 5 % 64 != 0).
    for (rows, cols) in [(4096, 2560), (9728, 2560), (2560, 9728), (5, 64)] {
        for tokens in [1, 3] {
            for path in [Path::Gemv, Path::GemvSimple, Path::Naive, Path::Tiled] {
                worst = worst.max(check_matmul(WeightType::Q4_0, path, rows, cols, tokens));
            }
        }
    }
    // lm_head q8_0 (filas reducidas para que el test sea rápido) y una forma chica.
    for (rows, cols) in [(8192, 2560), (6, 32)] {
        for path in [Path::Gemv, Path::GemvSimple, Path::Naive, Path::Tiled] {
            worst = worst.max(check_matmul(WeightType::Q8_0, path, rows, cols, 2));
        }
    }
    // Tabla de embeddings q6_0 (ADR 0012): lm_head con 1 y 2 filas de logits, y una forma chica.
    for (rows, cols) in [(8192, 2560), (6, 32)] {
        for tokens in [1, 2] {
            for path in [Path::Gemv, Path::GemvSimple, Path::Naive, Path::Tiled] {
                worst = worst.max(check_matmul(WeightType::Q6_0, path, rows, cols, tokens));
            }
        }
    }
    eprintln!("matmul: error máximo / Σ|w·x| {worst:.2e}");
    assert!(worst <= 1e-5, "{worst}");
}

#[test]
fn gemm_tiled_bordes_de_tokens() {
    // Tokens que no son múltiplo del bloque de 32 y bloques completos.
    let mut worst = 0f32;
    for tokens in [31, 32, 33, 70, 128] {
        worst = worst.max(check_matmul(
            WeightType::Q4_0,
            Path::Tiled,
            1024,
            2560,
            tokens,
        ));
        worst = worst.max(check_matmul(
            WeightType::Q8_0,
            Path::Tiled,
            512,
            2560,
            tokens,
        ));
    }
    eprintln!("gemm tiled (bordes): error máximo / Σ|w·x| {worst:.2e}");
    assert!(worst <= 1e-5, "{worst}");
}
