//! Criterio de T1.2 (ADR 0005): template y tokens idénticos a la referencia en todas las
//! fixtures, y casos borde idénticos al tokenizer oficial de Hugging Face.

use std::path::{Path, PathBuf};
use std::time::Instant;

use brasa_tokenizer::{RenderOptions, StreamDecoder, Tokenizer};
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_dir(&root().join("fixtures/qwen3-4b/tokenizer")).unwrap()
}

fn read_i32(path: &Path) -> Vec<u32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
        .collect()
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn template_y_tokens_iguales_a_las_fixtures() {
    let tok = tokenizer();
    let spec = json(&root().join("tools/fixture_prompts.json"));
    for p in spec["prompts"].as_array().unwrap() {
        let id = p["id"].as_str().unwrap();
        let dir = root().join("fixtures/qwen3-4b").join(id);
        let expected_text = std::fs::read_to_string(dir.join("prompt.txt")).unwrap();

        let text = if let Some(t) = p["text"].as_str() {
            t.to_string()
        } else if let Some(f) = p["text_file"].as_str() {
            std::fs::read_to_string(root().join(f)).unwrap()
        } else {
            let opts = RenderOptions {
                add_generation_prompt: true,
                enable_thinking: p["enable_thinking"].as_bool(),
            };
            let tools = p.get("tools");
            tok.template.render(&p["messages"], tools, &opts).unwrap()
        };
        assert_eq!(text, expected_text, "{id}: el template renderiza distinto");

        let ids = tok.encode(&text);
        assert_eq!(
            ids,
            read_i32(&dir.join("tokens.i32")),
            "{id}: tokens distintos"
        );
        assert_eq!(
            tok.decode(&ids),
            text.nfc().collect::<String>(),
            "{id}: decode no devuelve el texto"
        );
    }
}

#[test]
fn greedy_de_referencia_decodifica_igual() {
    let tok = tokenizer();
    let manifest = json(&root().join("fixtures/qwen3-4b/manifest.json"));
    for p in manifest["prompts"].as_array().unwrap() {
        let id = p["id"].as_str().unwrap();
        let ids = read_i32(&root().join("fixtures/qwen3-4b").join(id).join("greedy.i32"));
        assert_eq!(tok.decode(&ids), p["greedy_text"].as_str().unwrap(), "{id}");
        // El decodificador incremental produce el mismo texto token a token.
        let mut d = StreamDecoder::new();
        let mut s: String = ids.iter().map(|i| d.push(&tok.bpe, *i)).collect();
        s.push_str(&d.finish());
        assert_eq!(s, p["greedy_text"].as_str().unwrap(), "{id}: streaming");
    }
}

#[test]
fn casos_borde_iguales_a_hf() {
    let tok = tokenizer();
    let cases = json(&root().join("fixtures/qwen3-4b/tokenizer_cases.json"));
    for c in cases["cases"].as_array().unwrap() {
        let text = match c["text_file"].as_str() {
            Some(f) => std::fs::read_to_string(root().join(f)).unwrap(),
            None => c["text"].as_str().unwrap().to_string(),
        };
        let expected: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let t0 = Instant::now();
        let got = tok.encode(&text);
        let dt = t0.elapsed();
        let label: String = text.chars().take(40).collect();
        assert_eq!(got, expected, "caso {label:?}");
        if text.len() > 10_000 {
            eprintln!("{} tokens en {:.1} ms", got.len(), dt.as_secs_f64() * 1e3);
        }
    }
}

#[test]
fn stop_ids_de_qwen3() {
    let tok = tokenizer();
    assert_eq!(tok.stop_ids, vec![151645, 151643]); // <|im_end|>, <|endoftext|>
}
