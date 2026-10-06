//! GUI web embebida en el binario (U2): HTML, CSS y JS sin framework ni paso de build, servidos
//! en `/ui`. Todo sale de `include_str!`; la GUI no carga nada de internet (regla 3).

use axum::http::header;
use axum::response::{IntoResponse, Response};

pub const INDEX_HTML: &str = include_str!("../assets/index.html");
pub const APP_CSS: &str = include_str!("../assets/app.css");
pub const APP_JS: &str = include_str!("../assets/app.js");

fn asset(body: &'static str, content_type: &'static str) -> Response {
    ([(header::CONTENT_TYPE, content_type)], body).into_response()
}

pub async fn index() -> Response {
    asset(INDEX_HTML, "text/html; charset=utf-8")
}

pub async fn css() -> Response {
    asset(APP_CSS, "text/css; charset=utf-8")
}

pub async fn js() -> Response {
    asset(APP_JS, "application/javascript; charset=utf-8")
}
