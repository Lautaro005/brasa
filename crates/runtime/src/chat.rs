//! Conversación con tipos internos (`brasa_core::chat`): render con el chat template del modelo,
//! generación y eventos de texto, razonamiento y herramientas (ADR 0008).

use brasa_core::chat::{
    ChatEvent, ChatRequest, ErrorKind, FinishReason, Message, Role, ToolChoice, Usage,
};
use brasa_tokenizer::RenderOptions;
use serde_json::{Value, json};

use crate::qwen_output::QwenOutputParser;
use crate::sampler::{Sampler, SamplingParams};
use crate::session::{Session, StopReason};

fn role(r: Option<Role>) -> &'static str {
    match r {
        Some(Role::System) => "system",
        Some(Role::Assistant) => "assistant",
        Some(Role::Tool) => "tool",
        Some(Role::User) | None => "user",
    }
}

/// Mensajes en el formato que espera el chat template (estilo OpenAI).
pub fn template_messages(messages: &[Message]) -> Value {
    Value::Array(
        messages
            .iter()
            .map(|m| {
                let mut v = json!({"role": role(m.role), "content": m.content});
                if !m.tool_calls.is_empty() {
                    v["tool_calls"] = m
                        .tool_calls
                        .iter()
                        .map(|c| {
                            json!({"type": "function", "id": c.id,
                                   "function": {"name": c.name, "arguments": c.arguments}})
                        })
                        .collect();
                }
                if let Some(r) = &m.reasoning {
                    v["reasoning_content"] = json!(r);
                }
                v
            })
            .collect(),
    )
}

/// Texto del prompt de `req` con el chat template del tokenizer (sin necesitar la sesión, para
/// poder contar tokens fuera del hilo del modelo).
pub fn render_prompt(tok: &brasa_tokenizer::Tokenizer, req: &ChatRequest) -> crate::Result<String> {
    let tools: Option<Value> =
        (req.tool_choice != ToolChoice::None && !req.tools.is_empty()).then(|| {
            req.tools
                .iter()
                .map(|t| {
                    let mut f = json!({"name": t.name, "parameters": t.parameters});
                    if let Some(d) = &t.description {
                        f["description"] = json!(d);
                    }
                    json!({"type": "function", "function": f})
                })
                .collect()
        });
    let opts = RenderOptions {
        add_generation_prompt: true,
        enable_thinking: Some(req.thinking),
    };
    Ok(tok
        .template
        .render(&template_messages(&req.messages), tools.as_ref(), &opts)?)
}

impl Session {
    /// Texto del prompt de `req` con el chat template del modelo.
    pub fn render(&self, req: &ChatRequest) -> crate::Result<String> {
        render_prompt(self.tokenizer(), req)
    }

    /// Cantidad de tokens del prompt de `req`.
    pub fn count_tokens(&self, req: &ChatRequest) -> crate::Result<usize> {
        Ok(self.tokenizer().encode(&self.render(req)?).len())
    }

    /// Genera la respuesta a `req`, llamando a `on_event` con cada evento. Si `on_event`
    /// devuelve `false` (cliente desconectado), se cancela. Siempre termina con `Done` o `Error`.
    pub fn chat(&mut self, req: &ChatRequest, mut on_event: impl FnMut(ChatEvent) -> bool) {
        let ids = match self.render(req) {
            Ok(text) => self.tokenizer().encode(&text),
            Err(e) => {
                on_event(ChatEvent::Error(ErrorKind::InvalidRequest, e.0));
                return;
            }
        };
        let ctx = self.limits().ctx;
        // Como vLLM: si el cliente pide `max_tokens` explícito, prompt + salida tienen que entrar
        // en el contexto; si no, se rechaza (el agente compacta) en vez de recortar la salida.
        let need = ids.len() + req.max_tokens.unwrap_or(0);
        if ids.len() >= ctx || need > ctx {
            on_event(ChatEvent::Error(
                ErrorKind::ContextLength,
                format!(
                    "{need} tokens > {ctx} maximum (prompt {} + max_tokens {}; contexto del perfil de memoria)",
                    ids.len(),
                    req.max_tokens.unwrap_or(0)
                ),
            ));
            return;
        }
        let max_new = req.max_tokens.unwrap_or(ctx).min(ctx - ids.len()).max(1);
        let seed = req.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64)
        });
        let mut params = if req.thinking {
            SamplingParams::qwen3_thinking(seed)
        } else {
            SamplingParams::qwen3_no_thinking(seed)
        };
        if let Some(t) = req.temperature {
            params.temperature = t;
        }
        if let Some(p) = req.top_p {
            params.top_p = p;
        }
        if let Some(k) = req.top_k {
            params.top_k = k;
        }
        let mut sampler = Sampler::new(params, self.vocab());
        let mut parser = QwenOutputParser::new(format!("call_{seed:x}"));
        let mut alive = true;
        let res = self.generate(&ids, max_new, &mut sampler, |text| {
            for e in parser.push(text) {
                if !on_event(e) {
                    alive = false;
                    return false;
                }
            }
            true
        });
        let stats = match res {
            Ok(s) => s,
            Err(e) => {
                on_event(ChatEvent::Error(ErrorKind::Internal, e.0));
                return;
            }
        };
        if !alive {
            return;
        }
        for e in parser.finish() {
            if !on_event(e) {
                return;
            }
        }
        let reason = match stats.stop {
            StopReason::Cancelled => FinishReason::Cancelled,
            StopReason::MaxTokens | StopReason::ContextFull => FinishReason::Length,
            StopReason::Stop if parser.tool_calls() > 0 => FinishReason::ToolCalls,
            StopReason::Stop => FinishReason::Stop,
        };
        on_event(ChatEvent::Done {
            reason,
            usage: Usage {
                input_tokens: stats.prompt_tokens,
                output_tokens: stats.generated,
                cached_tokens: stats.reused_tokens,
            },
        });
    }
}
