use std::process::Command;

#[test]
fn imprime_version() {
    let out = Command::new(env!("CARGO_BIN_EXE_brasa"))
        .arg("--version")
        .output()
        .expect("no se pudo ejecutar brasa");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(
        stdout
            .trim()
            .starts_with(&format!("brasa {} (", env!("CARGO_PKG_VERSION"))),
        "{stdout}"
    );
}
