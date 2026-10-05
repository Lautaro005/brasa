//! `brasa plan <modelo>`: plan de memoria (pesos + KV + workspace + overhead) contra el
//! presupuesto de esta Mac o de un perfil simulado, sin cargar el modelo (ADR 0007).

use brasa_memory::planner::{Budget, Fit, describe, gib, rejection_message};
use brasa_runtime::{Limits, Session};
use clap::{Args, ValueEnum};

use crate::run::resolve_model;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Perfil {
    #[value(name = "8gb")]
    G8,
    #[value(name = "16gb")]
    G16,
}

#[derive(Debug, Args)]
pub struct PlanArgs {
    model: String,
    /// Contexto a evaluar.
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Tokens por bloque de prefill.
    #[arg(long, default_value_t = 128)]
    chunk: usize,
    /// Tipo de la KV cache: f16 (por defecto) o f32 (ADR 0009).
    #[arg(long, default_value = "f16", value_parser = crate::parse_kv)]
    kv: brasa_runtime::KvType,
    /// Simular el presupuesto de otra Mac en lugar de medir esta.
    #[arg(long, value_enum)]
    perfil: Option<Perfil>,
}

/// Devuelve `Ok(true)` si entra, `Ok(false)` si no (el proceso sale con código 2).
pub fn run(args: PlanArgs) -> Result<bool, String> {
    let dir = resolve_model(&args.model)?;
    let budget = match args.perfil {
        Some(Perfil::G8) => Budget::profile(8),
        Some(Perfil::G16) => Budget::profile(16),
        None => Budget::this_machine().ok_or("no hay dispositivo Metal")?,
    };
    let limits = Limits {
        ctx: args.ctx,
        max_tokens: args.chunk,
        max_logit_rows: 1,
        kv: args.kv,
    };
    let (fit, model_max) = Session::plan(&dir, limits, &budget).map_err(|e| e.to_string())?;
    println!(
        "modelo       {} (contexto máximo del modelo {model_max})",
        dir.display()
    );
    println!(
        "presupuesto  {:.2} GiB ({})",
        gib(budget.bytes),
        budget.source
    );
    match fit {
        Fit::Fits(p) => {
            println!("contexto     {} -> entra: {}", args.ctx, describe(&p));
            Ok(true)
        }
        Fit::TooBig { plan, max_ctx } => {
            println!("contexto     {} -> NO entra", args.ctx);
            println!("{}", rejection_message(args.ctx, &plan, &budget, max_ctx));
            Ok(false)
        }
    }
}
