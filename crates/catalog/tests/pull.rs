//! U3: `brasa pull` contra un servidor local chico (sin red): descarga, reanudación y fallo
//! limpio cuando el sha256 no coincide. ADR 0031: pesos ya convertidos (`[prebuilt]`) desde una
//! subcarpeta del repo y cancelación que deja el `.part`.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use brasa_catalog::manifest::{FileSpec, Manifest, Prebuilt};
use sha2::{Digest, Sha256};

struct Server {
    endpoint: String,
    stop: Arc<AtomicBool>,
    /// Rutas pedidas, en orden.
    log: Arc<Mutex<Vec<String>>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn handle(
    mut stream: TcpStream,
    files: Arc<HashMap<String, Vec<u8>>>,
    log: Arc<Mutex<Vec<String>>>,
) {
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
    log.lock()
        .unwrap()
        .push(text.split_whitespace().nth(1).unwrap_or("").to_string());
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
    let log = Arc::new(Mutex::new(Vec::new()));
    let s = stop.clone();
    let l = log.clone();
    std::thread::spawn(move || {
        let files = Arc::new(files);
        while !s.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    let f = files.clone();
                    let l = l.clone();
                    std::thread::spawn(move || handle(stream, f, l));
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                Err(_) => break,
            }
        }
    });
    Server {
        endpoint,
        stop,
        log,
    }
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
        convertible: true,
        prebuilt: None,
    }
}

fn spec(path: &str, content: &[u8]) -> FileSpec {
    FileSpec {
        path: path.into(),
        sha256: sha(content),
        size: Some(content.len() as u64),
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

#[test]
fn baja_los_pesos_convertidos_desde_la_subcarpeta() {
    let tok = b"{\"tokenizer\": 1}".to_vec();
    let weights: Vec<u8> = (0..2_000_000).map(|i| (i % 241) as u8).collect();
    let mut files = HashMap::new();
    files.insert("tokenizer.json".to_string(), tok.clone());
    files.insert("model.brasa".to_string(), weights.clone());
    let server = serve(files);
    let dir = tempfile::tempdir().unwrap();
    let mut m = manifest_for("no-se-usa.safetensors", b"x");
    m.prebuilt = Some(Prebuilt {
        repo: "dueno/brasa-base".into(),
        revision: "main".into(),
        subdir: "qwen3-4b-q4".into(),
        files: vec![spec("tokenizer.json", &tok), spec("model.brasa", &weights)],
    });
    let dest = dir.path().join("qwen3-4b-q4");
    let mut seen = Vec::new();
    let done = brasa_catalog::pull::download_prebuilt(
        &m,
        &dest,
        &server.endpoint,
        |p, got, total| seen.push((p.to_string(), got, total)),
        None,
    )
    .unwrap();
    assert_eq!(done.len(), 2);
    assert_eq!(std::fs::read(dest.join("model.brasa")).unwrap(), weights);
    assert_eq!(std::fs::read(dest.join("tokenizer.json")).unwrap(), tok);
    // URL: <endpoint>/<repo>/resolve/<revision>/<subdir>/<archivo>, en el orden del manifiesto.
    let log = server.log.lock().unwrap().clone();
    assert_eq!(
        log,
        [
            "/dueno/brasa-base/resolve/main/qwen3-4b-q4/tokenizer.json",
            "/dueno/brasa-base/resolve/main/qwen3-4b-q4/model.brasa"
        ]
    );
    let last = seen.last().unwrap();
    assert_eq!(
        (last.0.as_str(), last.1, last.2),
        (
            "model.brasa",
            weights.len() as u64,
            Some(weights.len() as u64)
        )
    );
    // Sin [prebuilt], error claro.
    let sin = manifest_for("a", b"a");
    let e =
        brasa_catalog::pull::download_prebuilt(&sin, &dest, &server.endpoint, |_, _, _| {}, None)
            .unwrap_err();
    assert!(e.0.contains("prebuilt"), "{}", e.0);
}

#[test]
fn cancelar_deja_el_parcial_y_despues_reanuda() {
    let content: Vec<u8> = (0..6_000_000).map(|i| (i % 239) as u8).collect();
    let mut files = HashMap::new();
    files.insert("model.brasa".to_string(), content.clone());
    let server = serve(files);
    let dir = tempfile::tempdir().unwrap();
    let mut m = manifest_for("x", b"x");
    m.prebuilt = Some(Prebuilt {
        repo: "u/r".into(),
        revision: "main".into(),
        subdir: String::new(),
        files: vec![spec("model.brasa", &content)],
    });
    let cancel = AtomicBool::new(false);
    // Cancela apenas llega el primer bloque.
    let e = brasa_catalog::pull::download_prebuilt(
        &m,
        dir.path(),
        &server.endpoint,
        |_, got, _| {
            if got > 0 {
                cancel.store(true, Ordering::Relaxed);
            }
        },
        Some(&cancel),
    )
    .unwrap_err();
    assert_eq!(e.0, brasa_catalog::pull::CANCELLED);
    let part = dir.path().join("model.brasa.part");
    let partial = std::fs::metadata(&part).unwrap().len();
    assert!(partial > 0 && partial < content.len() as u64, "{partial}");
    assert!(!dir.path().join("model.brasa").exists());

    // Reanuda: pide con Range desde lo que ya tenía.
    brasa_catalog::pull::download_prebuilt(&m, dir.path(), &server.endpoint, |_, _, _| {}, None)
        .unwrap();
    assert_eq!(
        std::fs::read(dir.path().join("model.brasa")).unwrap(),
        content
    );
    assert!(!part.exists());
}
