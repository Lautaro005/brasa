//! Graba en el binario el commit de git y si había cambios sin commitear al compilar. `brasa
//! benchmark` lo compara con el estado del repo para no medir un binario desactualizado.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    // Recompilar ante cualquier cambio de código o de git.
    for p in [
        "../../crates",
        "../../.git/HEAD",
        "../../.git/index",
        "../../.git/refs",
    ] {
        println!("cargo:rerun-if-changed={p}");
    }
    let commit = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "desconocido".into());
    let dirty =
        git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    let id = if dirty {
        format!("{commit}-dirty")
    } else {
        commit
    };
    println!("cargo:rustc-env=BRASA_BUILD_COMMIT={id}");
}
