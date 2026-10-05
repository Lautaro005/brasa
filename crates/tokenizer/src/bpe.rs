//! BPE byte-level compatible con `tokenizer.json` de Hugging Face (familia Qwen2/Qwen3).
//!
//! Pasos de `encode`: separar tokens agregados (coincidencia literal) → NFC → pre-tokenizar con la
//! regex del modelo → BPE sobre los bytes de cada pieza.

use std::collections::HashMap;

use fancy_regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::{Error, Result};

/// Token agregado (`added_tokens` de `tokenizer.json`): se reconoce literalmente en el texto.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
}

#[derive(Debug)]
pub struct Bpe {
    /// Bytes de cada token por id (incluye los agregados, con su contenido UTF-8).
    id_to_bytes: Vec<Vec<u8>>,
    /// Id de cada byte suelto: el punto de partida del BPE.
    byte_to_id: [u32; 256],
    /// (izquierda, derecha) -> (rango del merge, id resultante).
    merges: HashMap<(u32, u32), (u32, u32)>,
    added: Vec<AddedToken>,
    pretokenizer: Regex,
    nfc: bool,
}

/// Tabla de GPT-2: cada byte se representa como un carácter Unicode imprimible.
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..=255u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    table
}

fn bad(msg: impl Into<String>) -> Error {
    Error(format!("tokenizer.json: {}", msg.into()))
}

impl Bpe {
    /// Carga desde el contenido de `tokenizer.json`.
    pub fn from_tokenizer_json(json: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(json).map_err(|e| bad(e.to_string()))?;
        let model = &v["model"];
        if model["type"] != "BPE" {
            return Err(bad("solo se soporta model.type = BPE"));
        }
        if model["byte_fallback"] == true {
            return Err(bad("byte_fallback no soportado"));
        }

        let unicode = byte_to_unicode();
        let mut char_to_byte = HashMap::new();
        for (b, c) in unicode.iter().enumerate() {
            char_to_byte.insert(*c, b as u8);
        }
        let to_bytes = |s: &str| -> Result<Vec<u8>> {
            s.chars()
                .map(|c| {
                    char_to_byte.get(&c).copied().ok_or_else(|| {
                        bad(format!("carácter fuera del alfabeto byte-level: {c:?}"))
                    })
                })
                .collect()
        };

        let vocab = model["vocab"]
            .as_object()
            .ok_or_else(|| bad("falta model.vocab"))?;
        let added: Vec<AddedToken> = v["added_tokens"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|t| AddedToken {
                        id: t["id"].as_u64().unwrap_or(0) as u32,
                        content: t["content"].as_str().unwrap_or("").to_string(),
                        special: t["special"].as_bool().unwrap_or(false),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let max_id = vocab
            .values()
            .filter_map(Value::as_u64)
            .chain(added.iter().map(|t| t.id as u64))
            .max()
            .ok_or_else(|| bad("vocabulario vacío"))?;
        let mut id_to_bytes = vec![Vec::new(); max_id as usize + 1];
        let mut bytes_to_id = HashMap::with_capacity(vocab.len());
        for (tok, id) in vocab {
            let id = id.as_u64().ok_or_else(|| bad("id no numérico"))? as u32;
            let bytes = to_bytes(tok)?;
            bytes_to_id.insert(bytes.clone(), id);
            id_to_bytes[id as usize] = bytes;
        }
        for t in &added {
            id_to_bytes[t.id as usize] = t.content.as_bytes().to_vec();
        }

        let mut byte_to_id = [0u32; 256];
        for b in 0..=255u8 {
            byte_to_id[b as usize] = *bytes_to_id
                .get(&vec![b])
                .ok_or_else(|| bad(format!("falta el token del byte {b}")))?;
        }

        let raw_merges = model["merges"]
            .as_array()
            .ok_or_else(|| bad("falta model.merges"))?;
        let mut merges = HashMap::with_capacity(raw_merges.len());
        for (rank, m) in raw_merges.iter().enumerate() {
            // Dos formatos: "a b" (viejo) o ["a", "b"] (nuevo).
            let (l, r) = match m {
                Value::String(s) => s.split_once(' ').ok_or_else(|| bad("merge mal formado"))?,
                Value::Array(a) if a.len() == 2 => (
                    a[0].as_str().ok_or_else(|| bad("merge mal formado"))?,
                    a[1].as_str().ok_or_else(|| bad("merge mal formado"))?,
                ),
                _ => return Err(bad("merge mal formado")),
            };
            let (lb, rb) = (to_bytes(l)?, to_bytes(r)?);
            let mut joined = lb.clone();
            joined.extend_from_slice(&rb);
            let get = |b: &Vec<u8>| {
                bytes_to_id
                    .get(b)
                    .copied()
                    .ok_or_else(|| bad("merge con token fuera del vocabulario"))
            };
            merges.insert((get(&lb)?, get(&rb)?), (rank as u32, get(&joined)?));
        }

        let pattern = find_split_regex(&v["pre_tokenizer"])
            .ok_or_else(|| bad("no se encontró la regex de pre-tokenización"))?;
        let pretokenizer = Regex::new(&pattern).map_err(|e| bad(format!("regex: {e}")))?;
        let nfc = match &v["normalizer"] {
            Value::Null => false,
            n if n["type"] == "NFC" => true,
            n => return Err(bad(format!("normalizer no soportado: {n}"))),
        };

        let mut added = added;
        // Coincidencia más larga primero cuando dos tokens empiezan igual.
        added.sort_by_key(|t| std::cmp::Reverse(t.content.len()));
        Ok(Self {
            id_to_bytes,
            byte_to_id,
            merges,
            added,
            pretokenizer,
            nfc,
        })
    }

    /// Tamaño del vocabulario (ids válidos: `0..vocab_size`).
    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Id de un token agregado por su contenido, por ejemplo `<|im_end|>`.
    pub fn added_token_id(&self, content: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|t| t.content == content)
            .map(|t| t.id)
    }

    pub fn added_tokens(&self) -> &[AddedToken] {
        &self.added
    }

    /// Bytes crudos de un token (UTF-8 parcial posible).
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.id_to_bytes.get(id as usize).map(Vec::as_slice)
    }

    /// Tokeniza `text`. Los tokens agregados que aparezcan en el texto se reconocen literalmente.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3);
        let mut rest = text;
        while !rest.is_empty() {
            match self.next_added(rest) {
                Some((pos, tok)) => {
                    self.encode_plain(&rest[..pos], &mut out);
                    out.push(tok.id);
                    rest = &rest[pos + tok.content.len()..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    /// Primer token agregado en `s` (el más largo si hay varios en la misma posición).
    fn next_added<'s>(&'s self, s: &str) -> Option<(usize, &'s AddedToken)> {
        let mut best: Option<(usize, &AddedToken)> = None;
        for t in &self.added {
            if let Some(pos) = s.find(&t.content)
                && best.is_none_or(|(b, _)| pos < b)
            {
                best = Some((pos, t));
            }
        }
        best
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let normalized: String;
        let text = if self.nfc {
            normalized = text.nfc().collect();
            normalized.as_str()
        } else {
            text
        };
        // Comportamiento "Isolated": cada coincidencia es una pieza; los huecos también.
        let mut last = 0;
        for m in self.pretokenizer.find_iter(text) {
            let m = m.expect("la regex de pre-tokenización no debería fallar");
            if m.start() > last {
                self.bpe_piece(&text.as_bytes()[last..m.start()], out);
            }
            self.bpe_piece(m.as_str().as_bytes(), out);
            last = m.end();
        }
        if last < text.len() {
            self.bpe_piece(&text.as_bytes()[last..], out);
        }
    }

    /// BPE sobre los bytes de una pieza: une repetidamente el par adyacente de menor rango.
    fn bpe_piece(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let mut syms: Vec<u32> = bytes.iter().map(|b| self.byte_to_id[*b as usize]).collect();
        loop {
            let best = syms
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.merges
                        .get(&(w[0], w[1]))
                        .map(|&(rank, id)| (rank, i, id))
                })
                .min();
            let Some((rank, _, merged)) = best else { break };
            // Une todas las apariciones de ese par, de izquierda a derecha.
            let mut i = 0;
            let mut next = Vec::with_capacity(syms.len());
            while i < syms.len() {
                if i + 1 < syms.len()
                    && self.merges.get(&(syms[i], syms[i + 1])).map(|m| m.0) == Some(rank)
                {
                    next.push(merged);
                    i += 2;
                } else {
                    next.push(syms[i]);
                    i += 1;
                }
            }
            syms = next;
        }
        out.extend_from_slice(&syms);
    }

    /// Decodifica a texto (UTF-8 inválido se reemplaza por U+FFFD).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(b) = self.token_bytes(*id) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Busca la regex de un pre-tokenizer `Split` (directo o dentro de un `Sequence`).
fn find_split_regex(p: &Value) -> Option<String> {
    match p["type"].as_str()? {
        "Split" => p["pattern"]["Regex"].as_str().map(str::to_string),
        "Sequence" => p["pretokenizers"]
            .as_array()?
            .iter()
            .find_map(find_split_regex),
        _ => None,
    }
}

/// Decodificador incremental para streaming: devuelve solo caracteres UTF-8 completos y retiene
/// los bytes de un carácter partido entre tokens.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Agrega un token y devuelve el texto nuevo que ya se puede emitir.
    pub fn push(&mut self, bpe: &Bpe, id: u32) -> String {
        if let Some(b) = bpe.token_bytes(id) {
            self.pending.extend_from_slice(b);
        }
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            // Si el error es un carácter incompleto al final, se espera al próximo token.
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            // Bytes inválidos en el medio: se emiten con reemplazo.
            Err(_) => self.pending.len(),
        };
        let text = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        text
    }

    /// Vacía lo pendiente al terminar la generación.
    pub fn finish(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        text
    }
}
