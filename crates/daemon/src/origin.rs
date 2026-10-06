//! Protección contra pedidos que llegan desde páginas web abiertas en el navegador del usuario.
//!
//! El daemon no tiene autenticación y escucha en loopback. Eso alcanza contra otras máquinas,
//! pero no contra una página web cualquiera, que puede hacer que el navegador envíe:
//! - **pedidos de otro sitio (CSRF):** un `POST` sin cuerpo o con `text/plain` no necesita
//!   preflight de CORS, así que una página podría llamar a `/api/model/stop` o gastar GPU con
//!   `/v1/*`. Los clientes legítimos (SDKs, agentes, `curl`) no mandan `Origin`; el navegador sí.
//!   Los pedidos que modifican estado se aceptan solo sin `Origin` o con el origen del propio
//!   daemon (la GUI de `/ui`).
//! - **DNS rebinding:** un dominio que resuelve a 127.0.0.1 pasaría como mismo origen. Si el
//!   daemon escucha en loopback, el `Host` tiene que ser `127.0.0.1`, `localhost` o `[::1]`.

use std::net::SocketAddr;

use axum::extract::{Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::Shared;

fn forbidden(msg: &str) -> Response {
    (StatusCode::FORBIDDEN, axum::Json(json!({ "error": msg }))).into_response()
}

/// Nombre de host de un valor `Host` (`host[:puerto]`, `[v6][:puerto]`).
fn host_name(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next().unwrap_or("");
    }
    host.rsplit_once(':').map_or(host, |(h, _)| h)
}

/// Decide si se acepta un pedido. `None` si pasa; si no, el motivo.
pub fn check(
    addr: SocketAddr,
    method: &Method,
    host: Option<&str>,
    origin: Option<&str>,
) -> Option<&'static str> {
    if addr.ip().is_loopback() {
        if let Some(h) = host {
            if !matches!(host_name(h), "127.0.0.1" | "localhost" | "::1") {
                return Some("Host no local: el daemon solo atiende 127.0.0.1, localhost o [::1]");
            }
        }
    }
    let modifies = !matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS);
    if modifies {
        if let Some(o) = origin {
            let same = host.is_some_and(|h| o == format!("http://{h}"));
            if !same {
                return Some("pedido desde otro origen (página web): rechazado");
            }
        }
    }
    None
}

pub async fn local_only(State(s): State<Shared>, req: Request, next: Next) -> Response {
    let h = req.headers();
    let host = h.get(header::HOST).and_then(|v| v.to_str().ok());
    let origin = h.get(header::ORIGIN).and_then(|v| v.to_str().ok());
    if let Some(why) = check(s.addr, req.method(), host, origin) {
        return forbidden(why);
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lo() -> SocketAddr {
        "127.0.0.1:8080".parse().unwrap()
    }

    #[test]
    fn acepta_clientes_sin_origin_y_la_gui() {
        assert_eq!(
            check(lo(), &Method::POST, Some("127.0.0.1:8080"), None),
            None
        );
        assert_eq!(
            check(
                lo(),
                &Method::POST,
                Some("localhost:8080"),
                Some("http://localhost:8080")
            ),
            None
        );
        assert_eq!(
            check(
                lo(),
                &Method::GET,
                Some("[::1]:8080"),
                Some("https://otra.com")
            ),
            None
        );
        assert_eq!(check(lo(), &Method::POST, None, None), None);
    }

    #[test]
    fn rechaza_otro_origen_y_dns_rebinding() {
        assert!(
            check(
                lo(),
                &Method::POST,
                Some("127.0.0.1:8080"),
                Some("https://evil.example")
            )
            .is_some()
        );
        assert!(check(lo(), &Method::POST, Some("127.0.0.1:8080"), Some("null")).is_some());
        assert!(check(lo(), &Method::GET, Some("evil.example:8080"), None).is_some());
        assert!(
            check(
                lo(),
                &Method::POST,
                Some("127.0.0.1:8080"),
                Some("http://127.0.0.1:9999")
            )
            .is_some()
        );
    }

    #[test]
    fn expuesto_en_otra_interfaz_no_mira_el_host() {
        let any: SocketAddr = "0.0.0.0:8080".parse().unwrap();
        assert_eq!(
            check(any, &Method::GET, Some("mi-mac.local:8080"), None),
            None
        );
        assert!(
            check(
                any,
                &Method::POST,
                Some("mi-mac.local:8080"),
                Some("https://evil.example")
            )
            .is_some()
        );
    }
}
