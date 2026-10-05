//! Lector mínimo de safetensors (BF16/F16/F32) para validar contra los pesos originales.

use std::collections::HashMap;
use std::path::Path;

use crate::mmap::Mmap;
use crate::qtype::f16_to_f32;
use crate::{Error, Result};

#[derive(Debug)]
pub struct SafeTensors {
    map: Mmap,
    data_start: usize,
    entries: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let map = Mmap::open(path)?;
        let b = map.as_slice();
        let n = u64::from_le_bytes(
            b.get(..8)
                .ok_or_else(|| Error("safetensors truncado".into()))?
                .try_into()
                .unwrap(),
        ) as usize;
        let header: serde_json::Value = serde_json::from_slice(&b[8..8 + n])
            .map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let mut entries = HashMap::new();
        for (name, v) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let shape = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let off = v["data_offsets"].as_array().unwrap();
            entries.insert(
                name.clone(),
                (
                    v["dtype"].as_str().unwrap().to_string(),
                    shape,
                    off[0].as_u64().unwrap() as usize,
                    off[1].as_u64().unwrap() as usize,
                ),
            );
        }
        Ok(Self {
            map,
            data_start: 8 + n,
            entries,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.1.as_slice())
    }

    /// Tensor convertido a f32.
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, _, a, b) = self
            .entries
            .get(name)
            .ok_or_else(|| Error(format!("no existe el tensor {name}")))?;
        let raw = &self.map.as_slice()[self.data_start + a..self.data_start + b];
        Ok(match dtype.as_str() {
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            other => return Err(Error(format!("dtype {other} no soportado"))),
        })
    }
}
