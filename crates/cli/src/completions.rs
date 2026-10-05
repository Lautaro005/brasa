//! `brasa completions <shell>`: scripts de autocompletado con `clap_complete` (U5).

use clap::{Args, CommandFactory};
use clap_complete::{Shell, generate};

#[derive(Debug, Args)]
pub struct CompletionsArgs {
    /// Shell para el que se genera el script.
    #[arg(value_enum)]
    shell: Shell,
}

pub fn run(a: CompletionsArgs) {
    let mut cmd = crate::Cli::command();
    let name = cmd.get_name().to_string();
    generate(a.shell, &mut cmd, name, &mut std::io::stdout());
}
