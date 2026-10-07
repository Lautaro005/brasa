//! `brasa plan <modelo>`: plan de memoria (pesos + KV + workspace + overhead) contra el
//! presupuesto de esta Mac o de un perfil simulado, sin cargar el modelo (ADR 0007).

use brasa_memory::planner::{Budget, Fit, gib, rejection_message};
use brasa_runtime::{Limits, Session};
use clap::Args;

use crate::config::Perfil;
use crate::run::resolve_model;

#[derive(Debug, Args)]
pub struct PlanArgs {
    model: String,
    /// Contexto a evaluar.
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Tokens por bloque de prefill.
    #[arg(long, default_value_t = brasa_runtime::DEFAULT_CHUNK)]
    chunk: usize,
    /// Tipo de la KV cache: f16 (por defecto), q8_0 o f32 (ADR 0009).
    #[arg(long, default_value = "f16", value_parser = crate::parse_kv)]
    kv: brasa_runtime::KvType,
    /// Simular el presupuesto de otra Mac en lugar de medir esta.
    #[arg(long, value_enum)]
    perfil: Option<Perfil>,
    /// Salida en JSON.
    #[arg(long)]
    json: bool,
}

/// Devuelve `Ok(true)` si entra, `Ok(false)` si no (el proceso sale con código 2).
pub fn run(args: PlanArgs) -> Result<bool, String> {
    let dir = resolve_model(&args.model)?;
    let budget = match args.perfil {
        Some(p) => p.profile().budget(),
        None => Budget::this_machine().ok_or("no hay dispositivo Metal")?,
    };
    let limits = Limits {
        ctx: args.ctx,
        max_tokens: args.chunk,
        max_logit_rows: 1,
        kv: args.kv,
    };
    let (fit, model_max) = Session::plan(&dir, limits, &budget).map_err(|e| e.to_string())?;
    let (fits, plan, max_ctx, message) = match &fit {
        Fit::Fits(p) => (
            true,
            serde_json::to_value(p).unwrap_or(serde_json::Value::Null),
            Some(args.ctx),
            format!(
                "contexto {} -> entra: {}",
                args.ctx,
                brasa_memory::planner::describe(p)
            ),
        ),
        Fit::TooBig { plan, max_ctx } => (
            false,
            serde_json::to_value(plan).unwrap_or(serde_json::Value::Null),
            *max_ctx,
            rejection_message(args.ctx, plan, &budget, *max_ctx),
        ),
    };
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": dir.display().to_string(),
                "model_max_ctx": model_max,
                "ctx": args.ctx,
                "kv": args.kv.name(),
                "budget": {"bytes": budget.bytes, "gib": gib(budget.bytes), "source": budget.source},
                "fits": fits,
                "plan": plan,
                "max_ctx": max_ctx,
                "message": message,
            }))
            .unwrap()
        );
        return Ok(fits);
    }
    println!(
        "modelo       {} (contexto máximo del modelo {model_max})",
        dir.display()
    );
    println!(
        "presupuesto  {:.2} GiB ({})",
        gib(budget.bytes),
        budget.source
    );
    if fits {
        println!(
            "contexto     {} -> entra: {}",
            args.ctx,
            describe_plan(&plan)
        );
    } else {
        println!("contexto     {} -> NO entra", args.ctx);
        println!("{message}");
    }
    Ok(fits)
}

fn describe_plan(plan: &serde_json::Value) -> String {
    let g = |k: &str| plan[k].as_u64().unwrap_or(0) as f64 / (1u64 << 30) as f64;
    format!(
        "{:.2} GiB (pesos {:.2} + KV {:.2} + workspace {:.2} + overhead {:.2})",
        g("total"),
        g("weights"),
        g("kv"),
        g("workspace"),
        g("overhead")
    )
}
