//! `GET /api/bench`: lista los reportes JSON de `docs/bench/` (o `$BRASA_BENCH_DIR`) y extrae un
//! resumen para la tabla y los gráficos de la GUI. Los reportes ilegibles o marcados como no
//! válidos se informan aparte.

use std::path::{Path, PathBuf};

use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

fn bench_dir() -> PathBuf {
    std::env::var_os("BRASA_BENCH_DIR").map_or_else(|| PathBuf::from("docs/bench"), PathBuf::from)
}

fn json_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            json_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "json") {
            out.push(p);
        }
    }
}

fn num(v: &Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for k in path {
        cur = &cur[*k];
    }
    cur.as_f64()
}

fn one(dir: &Path, path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let rel = path.strip_prefix(dir).unwrap_or(path).display().to_string();
    Ok(json!({
        "file": rel,
        "timestamp": v["timestamp"],
        "chip": v["machine"]["chip"],
        "engine": v["engine"]["name"],
        "label": v["engine"]["label"],
        "model": v["model"]["name"],
        "quant": v["model"]["quant"],
        "ctx": v["ctx"],
        "prompt_tokens": v["prompt_tokens"],
        "gen_tokens": v["gen_tokens"],
        "ttft_ms": num(&v, &["summary", "ttft_ms", "median"]),
        "prefill_tok_s": num(&v, &["summary", "prefill_tok_s", "median"]),
        "decode_tok_s": num(&v, &["summary", "decode_tok_s", "median"]),
        "peak_footprint_bytes": num(&v, &["summary", "peak_footprint_bytes", "median"]),
        "valid": v["valid"].as_bool().unwrap_or(false),
        "invalid_reasons": v["invalid_reasons"],
    }))
}

pub async fn bench() -> Response {
    let dir = bench_dir();
    if !dir.is_dir() {
        return axum::Json(json!({
            "dir": dir.display().to_string(),
            "exists": false,
            "reports": [],
            "errors": [],
        }))
        .into_response();
    }
    let mut files = Vec::new();
    json_files(&dir, &mut files);
    files.sort();
    let mut reports = Vec::new();
    let mut errors = Vec::new();
    for f in &files {
        match one(&dir, f) {
            Ok(v) => reports.push(v),
            Err(e) => errors.push(json!({
                "file": f.strip_prefix(&dir).unwrap_or(f).display().to_string(),
                "error": e
            })),
        }
    }
    axum::Json(json!({
        "dir": dir.display().to_string(),
        "exists": true,
        "reports": reports,
        "errors": errors,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrae_resumen_de_un_reporte() {
        let dir = Path::new("/tmp");
        let p = Path::new("/tmp/x.json");
        let json = r#"{"timestamp":"t","machine":{"chip":"M1 Pro"},"engine":{"name":"brasa","label":""},
            "model":{"name":"qwen3-4b-q4","quant":"q4_0"},"ctx":2048,"prompt_tokens":1920,"gen_tokens":128,
            "summary":{"ttft_ms":{"median":900.0,"min":1,"max":2},"prefill_tok_s":{"median":500.0},
            "decode_tok_s":{"median":40.0},"peak_footprint_bytes":{"median":3.0e9}},
            "valid":false,"invalid_reasons":["swap"]}"#;
        let v: Value = serde_json::from_str(json).unwrap();
        let text = v.to_string();
        std::fs::write(p, text).unwrap();
        let r = one(dir, p).unwrap();
        assert_eq!(r["decode_tok_s"], 40.0);
        assert_eq!(r["engine"], "brasa");
        assert_eq!(r["valid"], false);
        let _ = std::fs::remove_file(p);
    }
}
