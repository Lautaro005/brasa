//! Cliente HTTP mínimo para localhost (GET JSON). Evita una dependencia de red para `brasa ps`
//! y el resto de comandos que consultan un `serve` local: `TcpStream` + HTTP/1.1.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde_json::Value;

/// Hace `GET path` a `host:port` y devuelve el cuerpo de la respuesta (error si el estado no
/// es 2xx).
pub fn get(host: &str, port: u16, path: &str) -> Result<String, String> {
    let addr = format!("{host}:{port}");
    let mut stream = TcpStream::connect(&addr)
        .map_err(|e| format!("no se pudo conectar a {addr}: {e}. ¿Corriste `brasa serve`?"))?;
    stream.set_read_timeout(Some(Duration::from_secs(15))).ok();
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("error escribiendo el pedido: {e}"))?;
    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .map_err(|e| format!("error leyendo la respuesta: {e}"))?;
    let (status, body) = parse(&buf)?;
    if !(200..300).contains(&status) {
        return Err(format!("{addr}{path} respondió {status}: {}", body.trim()));
    }
    Ok(body)
}

/// Igual que [`get`], pero parsea el cuerpo como JSON.
pub fn get_json(host: &str, port: u16, path: &str) -> Result<Value, String> {
    let body = get(host, port, path)?;
    serde_json::from_str(&body).map_err(|e| format!("respuesta JSON inválida de {path}: {e}"))
}

fn parse(buf: &[u8]) -> Result<(u16, String), String> {
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("respuesta HTTP truncada")?;
    let head = std::str::from_utf8(&buf[..split]).map_err(|e| e.to_string())?;
    let mut lines = head.lines();
    let status_line = lines.next().ok_or("respuesta HTTP vacía")?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("estado HTTP inválido: {status_line}"))?;
    let mut chunked = false;
    for h in lines {
        let l = h.to_ascii_lowercase();
        if let Some(v) = l.strip_prefix("transfer-encoding:") {
            chunked = v.contains("chunked");
        }
    }
    let raw = &buf[split + 4..];
    let body = if chunked {
        dechunk(raw)?
    } else {
        String::from_utf8_lossy(raw).into_owned()
    };
    Ok((status, body))
}

fn dechunk(mut raw: &[u8]) -> Result<String, String> {
    let mut out = Vec::new();
    loop {
        let line_end = raw
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or("chunk truncado")?;
        let size = usize::from_str_radix(
            std::str::from_utf8(&raw[..line_end])
                .map_err(|e| e.to_string())?
                .split(';')
                .next()
                .unwrap_or("")
                .trim(),
            16,
        )
        .map_err(|e| format!("tamaño de chunk inválido: {e}"))?;
        raw = &raw[line_end + 2..];
        if size == 0 {
            break;
        }
        out.extend_from_slice(&raw[..size]);
        raw = &raw[size + 2..];
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsea_respuesta_con_content_length() {
        let r = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(parse(r).unwrap(), (200, "{}".to_string()));
    }

    #[test]
    fn parsea_respuesta_chunked() {
        let r = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n";
        assert_eq!(parse(r).unwrap(), (200, "{}".to_string()));
    }

    #[test]
    fn rechaza_sin_cuerpo() {
        assert!(parse(b"HTTP/1.1 200 OK\r\n").is_err());
    }
}
