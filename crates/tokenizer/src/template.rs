//! Chat template Jinja ejecutado con minijinja, con el mismo entorno que usa transformers
//! (`trim_blocks`, `lstrip_blocks`, `tojson` = `json.dumps(ensure_ascii=False)`). ADR 0005.

use minijinja::value::{Value, ValueKind};
use minijinja::{Environment, Error as JinjaError, ErrorKind};

use crate::{Error, Result};

const NAME: &str = "chat";

/// Chat template compilado.
#[derive(Debug)]
pub struct ChatTemplate {
    env: Environment<'static>,
}

/// Parámetros de render, equivalentes a los de `apply_chat_template`.
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    pub add_generation_prompt: bool,
    /// `None` deja la variable sin definir (el template de Qwen3 usa `is defined`).
    pub enable_thinking: Option<bool>,
}

impl ChatTemplate {
    pub fn new(source: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<Value, JinjaError> {
                Err(JinjaError::new(ErrorKind::InvalidOperation, msg))
            },
        );
        env.add_template_owned(NAME, source.to_string())
            .map_err(|e| Error(format!("chat template inválido: {e}")))?;
        Ok(Self { env })
    }

    /// Carga el template desde el contenido de `tokenizer_config.json`.
    pub fn from_tokenizer_config(json: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| Error(format!("tokenizer_config.json: {e}")))?;
        let src = v["chat_template"]
            .as_str()
            .ok_or_else(|| Error("tokenizer_config.json no tiene chat_template".into()))?;
        Self::new(src)
    }

    /// Renderiza `messages` (y `tools`, si hay) con el formato de mensajes de OpenAI que espera el
    /// template: `{"role", "content", "tool_calls"?, "reasoning_content"?}`.
    pub fn render(
        &self,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
        opts: &RenderOptions,
    ) -> Result<String> {
        let tmpl = self.env.get_template(NAME).expect("template registrado");
        let mut ctx = vec![
            ("messages", Value::from_serialize(messages)),
            (
                "add_generation_prompt",
                Value::from(opts.add_generation_prompt),
            ),
        ];
        if let Some(t) = tools {
            ctx.push(("tools", Value::from_serialize(t)));
        }
        if let Some(t) = opts.enable_thinking {
            ctx.push(("enable_thinking", Value::from(t)));
        }
        let ctx: Value = ctx.into_iter().collect();
        tmpl.render(ctx)
            .map_err(|e| Error(format!("error al renderizar el chat template: {e:#}")))
    }
}

/// `tojson` de transformers: `json.dumps(x, ensure_ascii=False)` con separadores `", "`/`": "`.
fn tojson(v: &Value) -> std::result::Result<Value, JinjaError> {
    let mut out = String::new();
    write_py_json(v, &mut out)?;
    // Marcado como seguro: el resultado se inserta tal cual.
    Ok(Value::from_safe_string(out))
}

fn write_py_json(v: &Value, out: &mut String) -> std::result::Result<(), JinjaError> {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            if let Ok(i) = i64::try_from(v.clone()) {
                if v.is_integer() {
                    out.push_str(&i.to_string());
                    return Ok(());
                }
            }
            let f = f64::try_from(v.clone()).map_err(|_| {
                JinjaError::new(ErrorKind::InvalidOperation, "número no serializable")
            })?;
            out.push_str(&py_float(f));
        }
        ValueKind::String => write_py_str(v.as_str().unwrap_or_default(), out),
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (i, item) in v.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(&item, out)?;
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, key) in v.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_str(&key.to_string(), out);
                out.push_str(": ");
                write_py_json(&v.get_item(&key)?, out)?;
            }
            out.push('}');
        }
        _ => {
            return Err(JinjaError::new(
                ErrorKind::InvalidOperation,
                format!("tojson: tipo no soportado {:?}", v.kind()),
            ));
        }
    }
    Ok(())
}

/// Cadena JSON como la escribe Python con `ensure_ascii=False`.
fn write_py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `repr(float)` de Python: notación científica si el exponente es < -4 o >= 16.
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let exp = f.abs().log10().floor() as i32;
    if (-4..16).contains(&exp) {
        let s = format!("{f}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        // Rust: "1e-5" / "1.5e20"; Python: "1e-05" / "1.5e+20".
        let s = format!("{f:e}");
        let (mant, e) = s.split_once('e').unwrap();
        let e: i32 = e.parse().unwrap();
        format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tojson_como_python() {
        let v = Value::from_serialize(serde_json::json!({
            "b": [1, 2.5, true, null], "a": "x\"y\n\tñ", "c": {"d": 1e-5, "e": 1e20}
        }));
        let s = tojson(&v).unwrap().to_string();
        assert_eq!(
            s,
            r#"{"b": [1, 2.5, true, null], "a": "x\"y\n\tñ", "c": {"d": 1e-05, "e": 1e+20}}"#
        );
    }

    #[test]
    fn py_floats() {
        assert_eq!(py_float(1.0), "1.0");
        assert_eq!(py_float(0.1), "0.1");
        assert_eq!(py_float(0.0001), "0.0001");
        assert_eq!(py_float(1.5e-7), "1.5e-07");
        assert_eq!(py_float(-2.0), "-2.0");
    }
}
