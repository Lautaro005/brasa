//! U3: `brasa pull` contra un servidor local chico (sin red): descarga, reanudación y fallo
//! limpio cuando el sha256 no coincide.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use brasa_catalog::manifest::{FileSpec, Manifest};
use sha2::{Digest, Sha256};

struct Server {
    endpoint: String,
    stop: Arc<AtomicBool>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn handle(mut stream: TcpStream, files: Arc<HashMap<String, Vec<u8>>>) {
    // En macOS el socket aceptado hereda O_NONBLOCK del listener; se vuelve a bloqueante.
    stream.set_nonblocking(false).unwrap();
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            return;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    // El servidor de prueba indexa por nombre de archivo (el manifiesto usa rutas del repo).
    let key = text
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .rsplit('/')
        .next()
        .unwrap_or("")
        .to_string();
    let start = text
        .lines()
        .find_map(|l| {
            let l = l.to_ascii_lowercase();
            l.strip_prefix("range:")
                .and_then(|v| v.trim().strip_prefix("bytes="))
                .and_then(|v| v.trim_end_matches('-').parse::<usize>().ok())
        })
        .unwrap_or(0);
    let Some(content) = files.get(&key) else {
        let _ = stream
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    };
    if start > 0 && start < content.len() {
        let body = &content[start..];
        let head = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
            body.len(),
            start,
            content.len() - 1,
            content.len()
        );
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(body);
    } else {
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            content.len()
        );
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(content);
    }
}

fn serve(files: HashMap<String, Vec<u8>>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    std::thread::spawn(move || {
        let files = Arc::new(files);
        while !s.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    let f = files.clone();
                    std::thread::spawn(move || handle(stream, f));
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                Err(_) => break,
            }
        }
    });
    Server { endpoint, stop }
}

fn sha(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn manifest_for(path: &str, content: &[u8]) -> Manifest {
    Manifest {
        name: "test".into(),
        family: "qwen3".into(),
        source_repo: "test/repo".into(),
        source_revision: "rev".into(),
        hf_dir: "out".into(),
        quant: String::new(),
        max_context: 8,
        license: "test".into(),
        files: vec![FileSpec {
            path: path.into(),
            sha256: sha(content),
            size: Some(content.len() as u64),
        }],
    }
}

#[test]
fn descarga_y_verifica_el_sha256() {
    let content = b"contenido de prueba\ny algo mas\n".to_vec();
    let mut files = HashMap::new();
    files.insert("weights.bin".to_string(), content.clone());
    let server = serve(files);
    let dir = tempfile::tempdir().unwrap();
    let m = manifest_for("weights.bin", &content);

    let done =
        brasa_catalog::pull::download(&m, dir.path(), &server.endpoint, |_, _, _| {}).unwrap();
    assert_eq!(done.len(), 1);
    assert_eq!(
        std::fs::read(dir.path().join("weights.bin")).unwrap(),
        content
    );
    assert!(!dir.path().join("weights.bin.part").exists());
}

#[test]
fn reanuda_desde_el_parcial() {
    let content: Vec<u8> = (0..3_000_000).map(|i| (i % 251) as u8).collect();
    let mut files = HashMap::new();
    files.insert("big.bin".to_string(), content.clone());
    let server = serve(files);
    let dir = tempfile::tempdir().unwrap();
    let m = manifest_for("big.bin", &content);

    // Simula una descarga cortada a la mitad.
    std::fs::write(dir.path().join("big.bin.part"), &content[..1_500_000]).unwrap();
    brasa_catalog::pull::download(&m, dir.path(), &server.endpoint, |_, _, _| {}).unwrap();
    assert_eq!(std::fs::read(dir.path().join("big.bin")).unwrap(), content);
    assert!(!dir.path().join("big.bin.part").exists());
}

#[test]
fn falla_limpio_si_el_hash_no_coincide() {
    let mut files = HashMap::new();
    files.insert("weights.bin".to_string(), b"contenido alterado".to_vec());
    let server = serve(files);
    let dir = tempfile::tempdir().unwrap();
    // El manifiesto espera otro contenido.
    let m = manifest_for("weights.bin", b"contenido original");

    let err =
        brasa_catalog::pull::download(&m, dir.path(), &server.endpoint, |_, _, _| {}).unwrap_err();
    assert!(err.0.contains("sha256 no coincide"), "{}", err.0);
    assert!(!dir.path().join("weights.bin").exists());
    assert!(!dir.path().join("weights.bin.part").exists());
}
