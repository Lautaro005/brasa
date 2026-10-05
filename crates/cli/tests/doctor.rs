use std::process::Command;

fn brasa(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_brasa"))
        .args(args)
        .output()
        .expect("no se pudo ejecutar brasa");
    assert!(out.status.success(), "{out:?}");
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn doctor_json_tiene_los_campos() {
    let v: serde_json::Value = serde_json::from_str(&brasa(&["doctor", "--json"])).unwrap();
    let hw = &v["hardware"];
    assert!(hw["chip"].as_str().unwrap().starts_with("Apple M"));
    assert!(hw["gpu_cores"].as_u64().unwrap() >= 7);
    assert!(hw["memory_bytes"].as_u64().unwrap() >= 8 << 30);
    assert!(hw["metal"]["apple_family"].is_string());
    assert!(v["memory"]["pressure"].is_string());
}

#[test]
fn doctor_texto() {
    let s = brasa(&["doctor"]);
    assert!(s.contains("chip") && s.contains("Metal") && s.contains("presión"));
}

/// Compara `brasa doctor --json` con fuentes independientes (`system_profiler`, `sw_vers`).
/// Es el criterio de T0.2: corre igual en la M1 Pro 16 GB y en la M2 8 GB.
#[test]
fn doctor_coincide_con_system_profiler() {
    let v: serde_json::Value = serde_json::from_str(&brasa(&["doctor", "--json"])).unwrap();
    let hw = &v["hardware"];

    let out = Command::new("system_profiler")
        .args(["-json", "SPHardwareDataType", "SPDisplaysDataType"])
        .output()
        .expect("system_profiler");
    let sp: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let hwsp = &sp["SPHardwareDataType"][0];
    let gpu = &sp["SPDisplaysDataType"][0];

    assert_eq!(hw["chip"], hwsp["chip_type"]);
    assert_eq!(hw["model"], hwsp["machine_model"]);

    // "16 GB"
    let ram_gb: u64 = hwsp["physical_memory"]
        .as_str()
        .unwrap()
        .trim_end_matches(" GB")
        .parse()
        .unwrap();
    assert_eq!(hw["memory_bytes"].as_u64().unwrap(), ram_gb << 30);

    // "proc 10:8:2" o "proc 10:0:8:2": total, [otros niveles], rendimiento, eficiencia.
    let procs: Vec<u64> = hwsp["number_processors"]
        .as_str()
        .unwrap()
        .trim_start_matches("proc ")
        .split(':')
        .map(|s| s.parse().unwrap())
        .collect();
    let n = procs.len();
    assert_eq!(hw["cpu_performance_cores"].as_u64().unwrap(), procs[n - 2]);
    assert_eq!(hw["cpu_efficiency_cores"].as_u64().unwrap(), procs[n - 1]);

    let gpu_cores: u64 = gpu["sppci_cores"].as_str().unwrap().parse().unwrap();
    assert_eq!(hw["gpu_cores"].as_u64().unwrap(), gpu_cores);
    assert_eq!(hw["metal"]["name"], gpu["sppci_model"]);

    let sw = |flag: &str| {
        let o = Command::new("sw_vers").arg(flag).output().unwrap();
        String::from_utf8(o.stdout).unwrap().trim().to_string()
    };
    assert_eq!(hw["macos_version"].as_str().unwrap(), sw("-productVersion"));
    assert_eq!(hw["macos_build"].as_str().unwrap(), sw("-buildVersion"));
}
