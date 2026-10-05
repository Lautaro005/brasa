//! Parser incremental de la salida de Qwen3 (ADR 0008): separa razonamiento (`<think>…</think>`),
//! texto y llamadas a herramientas (`<tool_call>{"name": …, "arguments": {…}}</tool_call>`).
//! Recibe fragmentos de texto en cualquier corte (una etiqueta puede llegar partida).

use brasa_core::chat::{ChatEvent, ToolCall};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Al principio: el modelo puede abrir con `<think>`.
    Start,
    Content,
    Think,
    Call,
}

#[derive(Debug)]
pub struct QwenOutputParser {
    state: State,
    buf: String,
    /// Texto de la llamada en curso.
    call: String,
    calls: usize,
    /// Prefijo para los ids de llamada (único por generación).
    id_prefix: String,
    /// Se descartan los saltos de línea iniciales después de `</think>` y entre llamadas.
    trim_next: bool,
}

/// Mayor sufijo de `s` que es prefijo propio de alguna etiqueta (puede completarse después).
fn pending_tag_len(s: &str, tags: &[&str]) -> usize {
    let mut best = 0;
    for tag in tags {
        for n in (1..tag.len()).rev() {
            if n <= s.len() && s.is_char_boundary(s.len() - n) && tag.starts_with(&s[s.len() - n..])
            {
                best = best.max(n);
                break;
            }
        }
    }
    best
}

impl QwenOutputParser {
    pub fn new(id_prefix: impl Into<String>) -> Self {
        Self {
            state: State::Start,
            buf: String::new(),
            call: String::new(),
            calls: 0,
            id_prefix: id_prefix.into(),
            trim_next: false,
        }
    }

    /// Cantidad de llamadas a herramientas emitidas.
    pub fn tool_calls(&self) -> usize {
        self.calls
    }

    /// Procesa un fragmento y devuelve los eventos que ya se pueden emitir.
    pub fn push(&mut self, text: &str) -> Vec<ChatEvent> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            let progressed = match self.state {
                State::Start => self.step_start(),
                State::Content => self.step_content(&mut out),
                State::Think => self.step_think(&mut out),
                State::Call => self.step_call(&mut out),
            };
            if !progressed {
                break;
            }
        }
        out
    }

    /// Fin de la generación: vacía lo pendiente (una llamada sin cerrar se emite como texto).
    pub fn finish(&mut self) -> Vec<ChatEvent> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        match self.state {
            State::Think => {
                if !rest.is_empty() {
                    out.push(ChatEvent::Reasoning(rest));
                }
            }
            State::Call => {
                let raw = format!("{CALL_OPEN}{}{rest}", self.call);
                out.push(ChatEvent::Text(raw));
            }
            State::Start | State::Content => self.emit_text(rest, &mut out),
        }
        out
    }

    fn emit_text(&mut self, mut s: String, out: &mut Vec<ChatEvent>) {
        if self.trim_next {
            let t = s.trim_start_matches('\n').to_string();
            if t.is_empty() {
                return;
            }
            s = t;
            self.trim_next = false;
        }
        // Después de una llamada, el espacio en blanco entre llamadas no es contenido.
        if self.calls > 0 && s.trim().is_empty() {
            return;
        }
        if !s.is_empty() {
            out.push(ChatEvent::Text(s));
        }
    }

    fn step_start(&mut self) -> bool {
        let t = self.buf.trim_start_matches(['\n', ' ']);
        if t.starts_with(THINK_OPEN) {
            let skip = self.buf.len() - t.len() + THINK_OPEN.len();
            self.buf.drain(..skip);
            self.state = State::Think;
            return true;
        }
        if t.is_empty() || THINK_OPEN.starts_with(t) {
            return false; // todavía puede ser "<think>"
        }
        self.state = State::Content;
        true
    }

    fn step_content(&mut self, out: &mut Vec<ChatEvent>) -> bool {
        if let Some(i) = self.buf.find(CALL_OPEN) {
            let before: String = self.buf.drain(..i).collect();
            self.buf.drain(..CALL_OPEN.len());
            self.emit_text(before, out);
            self.state = State::Call;
            self.call.clear();
            return true;
        }
        // Un `</think>` suelto (razonamiento abierto en el prompt): lo previo era razonamiento.
        if let Some(i) = self.buf.find(THINK_CLOSE) {
            let before: String = self.buf.drain(..i).collect();
            self.buf.drain(..THINK_CLOSE.len());
            if !before.trim().is_empty() {
                out.push(ChatEvent::Reasoning(before));
            }
            self.trim_next = true;
            return true;
        }
        let keep = pending_tag_len(&self.buf, &[CALL_OPEN, THINK_CLOSE]);
        let emit = self.buf.len() - keep;
        if emit > 0 {
            let s: String = self.buf.drain(..emit).collect();
            self.emit_text(s, out);
        }
        false
    }

    fn step_think(&mut self, out: &mut Vec<ChatEvent>) -> bool {
        if let Some(i) = self.buf.find(THINK_CLOSE) {
            let before: String = self.buf.drain(..i).collect();
            self.buf.drain(..THINK_CLOSE.len());
            if !before.is_empty() {
                out.push(ChatEvent::Reasoning(before));
            }
            self.state = State::Content;
            self.trim_next = true;
            return true;
        }
        let keep = pending_tag_len(&self.buf, &[THINK_CLOSE]);
        let emit = self.buf.len() - keep;
        if emit > 0 {
            out.push(ChatEvent::Reasoning(self.buf.drain(..emit).collect()));
        }
        false
    }

    fn step_call(&mut self, out: &mut Vec<ChatEvent>) -> bool {
        let Some(i) = self.buf.find(CALL_CLOSE) else {
            let keep = pending_tag_len(&self.buf, &[CALL_CLOSE]);
            let take = self.buf.len() - keep;
            let s: String = self.buf.drain(..take).collect();
            self.call.push_str(&s);
            return false;
        };
        let rest: String = self.buf.drain(..i).collect();
        self.buf.drain(..CALL_CLOSE.len());
        self.call.push_str(&rest);
        let body = std::mem::take(&mut self.call);
        match parse_call(&body) {
            Some((name, arguments)) => {
                let id = format!("{}_{}", self.id_prefix, self.calls);
                self.calls += 1;
                out.push(ChatEvent::ToolCall(ToolCall {
                    id,
                    name,
                    arguments,
                }));
            }
            None => out.push(ChatEvent::Text(format!("{CALL_OPEN}{body}{CALL_CLOSE}"))),
        }
        self.state = State::Content;
        self.trim_next = true;
        true
    }
}

/// `{"name": "...", "arguments": {...}}` (los argumentos pueden venir como string JSON).
fn parse_call(body: &str) -> Option<(String, serde_json::Value)> {
    let v: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let args = match v.get("arguments") {
        Some(serde_json::Value::String(s)) => serde_json::from_str(s).ok()?,
        Some(a @ serde_json::Value::Object(_)) => a.clone(),
        None => serde_json::json!({}),
        Some(_) => return None,
    };
    Some((name, args))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Pasa `s` cortado en fragmentos de `n` caracteres y junta los eventos.
    fn run(s: &str, n: usize) -> Vec<ChatEvent> {
        let mut p = QwenOutputParser::new("call");
        let chars: Vec<char> = s.chars().collect();
        let mut out = Vec::new();
        for c in chars.chunks(n) {
            out.extend(p.push(&c.iter().collect::<String>()));
        }
        out.extend(p.finish());
        merge(out)
    }

    /// Une fragmentos consecutivos del mismo tipo para comparar fácil.
    fn merge(evs: Vec<ChatEvent>) -> Vec<ChatEvent> {
        let mut out: Vec<ChatEvent> = Vec::new();
        for e in evs {
            match (out.last_mut(), e) {
                (Some(ChatEvent::Text(a)), ChatEvent::Text(b)) => a.push_str(&b),
                (Some(ChatEvent::Reasoning(a)), ChatEvent::Reasoning(b)) => a.push_str(&b),
                (_, e) => out.push(e),
            }
        }
        out
    }

    #[test]
    fn texto_razonamiento_y_llamadas_en_cualquier_corte() {
        let s = "<think>\nLeo el archivo.\n</think>\n\nVoy a mirar.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"run\", \"arguments\": \"{\\\"cmd\\\": \\\"ls\\\"}\"}\n</tool_call>";
        let expected = vec![
            ChatEvent::Reasoning("\nLeo el archivo.\n".into()),
            ChatEvent::Text("Voy a mirar.\n".into()),
            ChatEvent::ToolCall(ToolCall {
                id: "call_0".into(),
                name: "read_file".into(),
                arguments: json!({"path": "Cargo.toml"}),
            }),
            ChatEvent::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "run".into(),
                arguments: json!({"cmd": "ls"}),
            }),
        ];
        for n in [1, 2, 3, 5, 7, 1000] {
            assert_eq!(run(s, n), expected, "fragmentos de {n}");
        }
    }

    #[test]
    fn solo_texto_y_texto_con_menor() {
        assert_eq!(
            run("Hola, 3 < 4 y <b>", 2),
            vec![ChatEvent::Text("Hola, 3 < 4 y <b>".into())]
        );
    }

    #[test]
    fn llamada_invalida_queda_como_texto() {
        let s = "<tool_call>\n{\"name\": \"x\", \"arguments\": {roto}\n</tool_call>";
        assert_eq!(run(s, 3), vec![ChatEvent::Text(s.into())]);
    }

    #[test]
    fn llamada_sin_cerrar_queda_como_texto() {
        let s = "<tool_call>\n{\"name\": \"x\"";
        assert_eq!(run(s, 4), vec![ChatEvent::Text(s.into())]);
    }

    #[test]
    fn think_vacio_de_enable_thinking_false() {
        // Con enable_thinking = false el bloque vacío está en el prompt; la salida es texto.
        assert_eq!(
            run("\n\nRespuesta.", 3),
            vec![ChatEvent::Text("\n\nRespuesta.".into())]
        );
    }
}
