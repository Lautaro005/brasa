//! Tokenizer BPE byte-level y chat template (ADR 0005).

pub mod bpe;
pub mod template;

use std::fmt;
use std::path::Path;

pub use bpe::{AddedToken, Bpe, StreamDecoder};
pub use template::{ChatTemplate, RenderOptions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Tokenizer y chat template de un modelo, cargados de su carpeta (`tokenizer.json` y
/// `tokenizer_config.json`).
#[derive(Debug)]
pub struct Tokenizer {
    pub bpe: Bpe,
    pub template: ChatTemplate,
    /// Tokens que terminan la generación (`eos_token` del config más `<|endoftext|>`).
    pub stop_ids: Vec<u32>,
}

impl Tokenizer {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name))
                .map_err(|e| Error(format!("{}: {e}", dir.join(name).display())))
        };
        let bpe = Bpe::from_tokenizer_json(&read("tokenizer.json")?)?;
        let config = read("tokenizer_config.json")?;
        let template = ChatTemplate::from_tokenizer_config(&config)?;
        let cfg: serde_json::Value =
            serde_json::from_str(&config).map_err(|e| Error(e.to_string()))?;
        let mut stop_ids: Vec<u32> = [cfg["eos_token"].as_str(), Some("<|endoftext|>")]
            .into_iter()
            .flatten()
            .filter_map(|t| bpe.added_token_id(t))
            .collect();
        stop_ids.dedup();
        Ok(Self {
            bpe,
            template,
            stop_ids,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode(text)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.bpe.decode(ids)
    }
}
