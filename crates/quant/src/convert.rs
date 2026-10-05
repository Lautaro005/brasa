//! Conversión nativa de un modelo Qwen3 en safetensors (BF16/F16 de Hugging Face) al formato
//! `.brasa` (U4, ADR 0006), sin Python. Replica el layout y la cuantización de
//! `tools/convert_brasa.py`: proyecciones en q4_0, embeddings (= lm_head) en q8_0, normas en f32.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::qtype::{BLOCK, f16_to_f32, f32_to_f16};
use crate::safetensors::SafeTensors;
use crate::{Error, Result};

const PAGE: u64 = 16384;
const CONVERTER: &str = "brasa-convert v1";
const MAGIC: &[u8; 4] = b"BRSA";
const VERSION: u32 = 1;
/// Esquema de cuantización de cada tensor del formato (ADR 0006).
const Q4_BYTES: usize = 18;
const Q8_BYTES: usize = 34;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertReport {
    pub tensors: usize,
    pub bytes: u64,
    pub data_sha256: String,
}

fn align(n: u64) -> u64 {
    n.div_ceil(PAGE) * PAGE
}

fn dtype_for(name: &str, ndim: usize) -> &'static str {
    if ndim == 1 {
        "f32"
    } else if name == "model.embed_tokens.weight" {
        "q8_0"
    } else {
        "q4_0"
    }
}

fn nbytes(dtype: &str, shape: &[usize]) -> Result<usize> {
    let n: usize = shape.iter().product();
    Ok(match dtype {
        "f32" => 4 * n,
        "q4_0" | "q8_0" => {
            let last = shape.last().copied().unwrap_or(0);
            if last % BLOCK != 0 {
                return Err(Error(format!(
                    "la última dimensión {last} no es múltiplo de {BLOCK}"
                )));
            }
            n / BLOCK * if dtype == "q4_0" { Q4_BYTES } else { Q8_BYTES }
        }
        other => return Err(Error(format!("dtype {other} no soportado"))),
    })
}

/// q4_0 con el sesgo de 8 y la escala del mayor módulo (como `quant_q4_0` del conversor Python).
fn quant_q4_0(w: &[f32], out: &mut Vec<u8>) {
    out.clear();
    for block in w.chunks_exact(BLOCK) {
        let mut idx = 0;
        let mut best = block[0].abs();
        for (i, v) in block.iter().enumerate() {
            let a = v.abs();
            if a > best {
                best = a;
                idx = i;
            }
        }
        let d16 = f32_to_f16(block[idx] / -8.0);
        let d = f16_to_f32(d16);
        let mut q = [8u8; BLOCK];
        if d != 0.0 {
            for (j, &v) in block.iter().enumerate() {
                q[j] = ((v / d).round_ties_even() + 8.0).clamp(0.0, 15.0) as u8;
            }
        }
        out.extend_from_slice(&d16.to_le_bytes());
        for j in 0..16 {
            out.push(q[j] | (q[j + 16] << 4));
        }
    }
}

/// q8_0 (como `quant_q8_0` del conversor Python).
fn quant_q8_0(w: &[f32], out: &mut Vec<u8>) {
    out.clear();
    for block in w.chunks_exact(BLOCK) {
        let amax = block.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d16 = f32_to_f16(amax / 127.0);
        let d = f16_to_f32(d16);
        out.extend_from_slice(&d16.to_le_bytes());
        for &v in block {
            let q = if d == 0.0 {
                0i32
            } else {
                (v / d).round_ties_even().clamp(-127.0, 127.0) as i32
            };
            out.push(q as i8 as u8);
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn f32_bytes(w: &[f32], out: &mut Vec<u8>) {
    out.clear();
    for v in w {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

struct Plan {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    nbytes: usize,
    offset: u64,
    sha256: String,
}

fn meta_json(plans: &[Plan], config: &Value, repo: &str, commit: &str, data_sha256: &str) -> Value {
    let tensors: Vec<Value> = plans
        .iter()
        .map(|p| {
            json!({
                "name": p.name,
                "dtype": p.dtype,
                "shape": p.shape,
                "nbytes": p.nbytes,
                "offset": p.offset,
                "sha256": p.sha256,
            })
        })
        .collect();
    json!({
        "format": "brasa",
        "version": VERSION,
        "converter": CONVERTER,
        "model": {
            "family": "qwen3",
            "source_repo": repo,
            "source_commit": commit,
            "config": config,
        },
        "quant": {"block": BLOCK, "scheme": "q4_0 lineales, q8_0 embeddings/lm_head, f32 normas (ADR 0006)"},
        "page": PAGE,
        "tensors": tensors,
        "data_sha256": data_sha256,
    })
}

/// Convierte `hf_dir` (con `config.json`, `model.safetensors.index.json` y sus shards) a
/// `out_dir/model.brasa` más las copias del tokenizer.
pub fn convert(hf_dir: &Path, out_dir: &Path, repo: &str, commit: &str) -> Result<ConvertReport> {
    let config: Value = serde_json::from_str(
        &std::fs::read_to_string(hf_dir.join("config.json"))
            .map_err(|e| Error(format!("{}: {e}", hf_dir.join("config.json").display())))?,
    )
    .map_err(|e| Error(format!("config.json: {e}")))?;
    if config["model_type"] != "qwen3" {
        return Err(Error(format!(
            "model_type {} no es qwen3",
            config["model_type"]
        )));
    }
    let index: Value = serde_json::from_str(
        &std::fs::read_to_string(hf_dir.join("model.safetensors.index.json"))
            .map_err(|e| Error(format!("model.safetensors.index.json: {e}")))?,
    )
    .map_err(|e| Error(format!("index.json: {e}")))?;
    let weight_map: BTreeMap<String, String> = index["weight_map"]
        .as_object()
        .ok_or_else(|| Error("index.json: falta weight_map".into()))?
        .iter()
        .map(|(k, v)| {
            v.as_str()
                .map(|s| (k.clone(), s.to_string()))
                .ok_or_else(|| Error(format!("weight_map[{k}] no es string")))
        })
        .collect::<Result<_>>()?;

    let files: BTreeSet<&String> = weight_map.values().collect();
    let mut shards: BTreeMap<&str, SafeTensors> = BTreeMap::new();
    for f in files {
        shards.insert(f.as_str(), SafeTensors::open(&hf_dir.join(f))?);
    }

    // Plan de tensores (orden alfabético, como el conversor Python).
    let mut plans = Vec::with_capacity(weight_map.len());
    for (name, file) in &weight_map {
        let st = &shards[file.as_str()];
        let shape = st
            .shape(name)
            .ok_or_else(|| Error(format!("no existe el tensor {name}")))?
            .to_vec();
        let dtype = dtype_for(name, shape.len());
        let nb = nbytes(dtype, &shape)?;
        plans.push(Plan {
            name: name.clone(),
            dtype,
            shape,
            nbytes: nb,
            offset: 0,
            sha256: "0".repeat(64),
        });
    }

    // Offsets absolutos alineados a página; el encabezado tiene que entrar antes de los datos.
    let mut data_start = PAGE;
    let (mut total, mut header_len);
    loop {
        let mut off = data_start;
        for p in &mut plans {
            p.offset = off;
            off = align(off + p.nbytes as u64);
        }
        total = off;
        let meta = meta_json(&plans, &config, repo, commit, &"0".repeat(64));
        header_len = serde_json::to_vec(&meta)
            .map_err(|e| Error(e.to_string()))?
            .len() as u64
            + 16;
        if header_len <= data_start {
            break;
        }
        data_start += PAGE;
    }

    std::fs::create_dir_all(out_dir).map_err(|e| Error(format!("{}: {e}", out_dir.display())))?;
    let out_path = out_dir.join("model.brasa");
    let mut file =
        File::create(&out_path).map_err(|e| Error(format!("{}: {e}", out_path.display())))?;
    file.set_len(total)
        .map_err(|e| Error(format!("{}: {e}", out_path.display())))?;

    let mut all_hashes = String::with_capacity(64 * plans.len());
    let mut buf = Vec::new();
    for p in &mut plans {
        let w = shards[weight_map[&p.name].as_str()].to_f32(&p.name)?;
        match p.dtype {
            "f32" => f32_bytes(&w, &mut buf),
            "q4_0" => quant_q4_0(&w, &mut buf),
            _ => quant_q8_0(&w, &mut buf),
        }
        if buf.len() != p.nbytes {
            return Err(Error(format!(
                "{}: {} bytes, se esperaban {}",
                p.name,
                buf.len(),
                p.nbytes
            )));
        }
        p.sha256 = hex(&Sha256::digest(&buf));
        all_hashes.push_str(&p.sha256);
        file.seek(SeekFrom::Start(p.offset))
            .map_err(|e| Error(format!("{}: {e}", out_path.display())))?;
        file.write_all(&buf)
            .map_err(|e| Error(format!("{}: {e}", out_path.display())))?;
    }
    file.flush()
        .map_err(|e| Error(format!("{}: {e}", out_path.display())))?;

    let data_sha256 = hex(&Sha256::digest(all_hashes.as_bytes()));
    let meta = meta_json(&plans, &config, repo, commit, &data_sha256);
    let mut json = serde_json::to_vec(&meta).map_err(|e| Error(e.to_string()))?;
    let mut header = Vec::with_capacity(16 + json.len());
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&VERSION.to_le_bytes());
    header.extend_from_slice(&(json.len() as u64).to_le_bytes());
    header.append(&mut json);
    if header.len() as u64 > data_start {
        return Err(Error("el encabezado no entra antes de los datos".into()));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|e| Error(format!("{}: {e}", out_path.display())))?;
    file.write_all(&header)
        .map_err(|e| Error(format!("{}: {e}", out_path.display())))?;
    file.flush()
        .map_err(|e| Error(format!("{}: {e}", out_path.display())))?;

    for name in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(hf_dir.join(name), out_dir.join(name))
            .map_err(|e| Error(format!("{name}: {e}")))?;
    }
    Ok(ConvertReport {
        tensors: plans.len(),
        bytes: plans.iter().map(|p| p.nbytes as u64).sum(),
        data_sha256,
    })
}
