//! `brasa ps`: consulta `/api/status` y `/api/metrics` de un `serve` corriendo y los muestra
//! legibles (o crudos con `--json`).

use clap::Args;
use serde_json::Value;

use crate::http;

#[derive(Debug, Args)]
pub struct PsArgs {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Imprime las respuestas crudas en JSON.
    #[arg(long)]
    json: bool,
}

fn gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
}

fn take(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn mib(b: u64) -> String {
    format!("{:.0} MiB", b as f64 / (1u64 << 20) as f64)
}

fn str_at<'a>(v: &'a Value, path: &[&str]) -> &'a str {
    path.iter()
        .fold(v, |acc, k| &acc[*k])
        .as_str()
        .unwrap_or("?")
}

fn u64_at(v: &Value, path: &[&str]) -> u64 {
    path.iter().fold(v, |acc, k| &acc[*k]).as_u64().unwrap_or(0)
}

fn f64_at(v: &Value, path: &[&str]) -> Option<f64> {
    path.iter()
        .fold(v, |acc, k| &acc[*k])
        .as_f64()
        .filter(|x| x.is_finite())
}

fn duration(secs: f64) -> String {
    let s = secs as u64;
    let (h, m, s) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}h {m}m {s}s")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

pub fn run(a: PsArgs) -> Result<(), String> {
    let status = http::get_json(&a.host, a.port, "/api/status")?;
    let metrics = http::get_json(&a.host, a.port, "/api/metrics")?;
    if a.json {
        let both = serde_json::json!({"status": status, "metrics": metrics});
        println!(
            "{}",
            serde_json::to_string_pretty(&both).expect("json serializable")
        );
    } else {
        print_status(&status);
        print_metrics(&metrics);
    }
    Ok(())
}

fn print_status(s: &Value) {
    println!(
        "brasa {} ({}) — arriba {}",
        str_at(s, &["version"]),
        str_at(s, &["commit"]),
        duration(s["uptime_s"].as_f64().unwrap_or(0.0))
    );
    println!(
        "modelo      {} ({})",
        str_at(s, &["model", "id"]),
        str_at(s, &["model", "family"])
    );
    println!("ruta        {}", str_at(s, &["model", "path"]));
    let sha = str_at(s, &["model", "weights_sha256"]);
    let sha = if sha.len() > 16 { &sha[..16] } else { sha };
    println!(
        "pesos       sha256 {sha}…  {}",
        gib(u64_at(s, &["model", "weights_bytes"]))
    );
    println!(
        "origen      {} @ {}",
        str_at(s, &["model", "source_repo"]),
        take(str_at(s, &["model", "source_commit"]), 12)
    );
    println!(
        "contexto    {} (kv {}, bloque {})",
        u64_at(s, &["context", "ctx"]),
        str_at(s, &["context", "kv"]),
        u64_at(s, &["context", "chunk"])
    );
    let plan = &s["plan"];
    println!(
        "plan        {} (pesos {} + KV {} + workspace {} + overhead {})",
        gib(u64_at(plan, &["total"])),
        gib(u64_at(plan, &["weights"])),
        gib(u64_at(plan, &["kv"])),
        gib(u64_at(plan, &["workspace"])),
        gib(u64_at(plan, &["overhead"]))
    );
    println!(
        "proceso     huella {} (residente {}, pico {})",
        gib(u64_at(s, &["process", "footprint"])),
        gib(u64_at(s, &["process", "resident"])),
        gib(u64_at(s, &["process", "lifetime_peak_footprint"]))
    );
    println!(
        "sistema     presión {}, disponible {}, swap {}",
        str_at(s, &["system", "pressure"]),
        s["system"]["available_percent"]
            .as_u64()
            .map_or("?".to_string(), |p| format!("{p}%")),
        mib(u64_at(s, &["system", "swap_used"]))
    );
    println!(
        "cola        {} encolados, {} en ejecución",
        u64_at(s, &["queue", "pending"]),
        u64_at(s, &["queue", "running"])
    );
}

fn line(label: &str, st: &Value) {
    let f = |k: &str| {
        f64_at(st, &[k])
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "—".into())
    };
    println!(
        "{label:<12}último {}, media {}, p50 {} (n={})",
        f("last"),
        f("mean"),
        f("p50"),
        u64_at(st, &["samples"])
    );
}

fn print_metrics(m: &Value) {
    println!();
    println!("Pedidos");
    if let Some(map) = m["endpoints"].as_object() {
        if map.is_empty() {
            println!("  (ninguno)");
        }
        for (k, v) in map {
            println!("  {k:<26}{}", v.as_u64().unwrap_or(0));
        }
    }
    println!(
        "Tokens      prompt {}, generados {}, reutilizados {}",
        u64_at(m, &["prompt_tokens"]),
        u64_at(m, &["generated_tokens"]),
        u64_at(m, &["cached_tokens"])
    );
    line("TTFT", &m["ttft_ms"]);
    println!(
        "{:<12}último {}, media {}, p50 {} (n={})",
        "decode tok/s",
        f64_at(&m["decode_tok_s"], &["last"])
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "—".into()),
        f64_at(&m["decode_tok_s"], &["mean"])
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "—".into()),
        f64_at(&m["decode_tok_s"], &["p50"])
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "—".into()),
        u64_at(&m["decode_tok_s"], &["samples"])
    );
    let errors = &m["errors"];
    if errors.as_object().is_none_or(|o| o.is_empty()) {
        println!("errores     ninguno");
    } else if let Some(map) = errors.as_object() {
        for (k, v) in map {
            println!("errores     {k}: {}", v.as_u64().unwrap_or(0));
        }
    }
}
