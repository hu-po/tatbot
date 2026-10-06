use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, path::Path};

fn collect(root: &Path, path: &Path, files: &mut BTreeMap<String, String>) {
    println!("cargo:rerun-if-changed={}", path.display());
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("native source directory") {
            collect(root, &entry.expect("native source entry").path(), files);
        }
    } else {
        let name = path
            .strip_prefix(root)
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        files.insert(
            name,
            format!("{:x}", Sha256::digest(fs::read(path).unwrap())),
        );
    }
}

fn main() {
    let crate_dir = std::path::PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = crate_dir.parent().unwrap().parent().unwrap();
    let mut sources = BTreeMap::new();
    // Keep this inventory in sync with arm_calibration.native_source_digests.
    for name in [
        "rust/Cargo.toml",
        "rust/Cargo.lock",
        "rust/tatbot-arm/Cargo.toml",
        "rust/tatbot-arm/build.rs",
        "rust/tatbot-arm/src",
        "rust/trossen-arm-sys/Cargo.toml",
        "rust/trossen-arm-sys/build.rs",
        "rust/trossen-arm-sys/src",
        "rust/trossen-arm-sys/include",
    ] {
        collect(root, &root.join(name), &mut sources);
    }
    println!("cargo:rerun-if-env-changed=TATBOT_SOURCE_COMMIT");
    let sdk = env::var("DEP_TATBOT_TROSSEN_BRIDGE_SDK_VERSION")
        .ok()
        .map(|version| {
            serde_json::json!({
                "version": version,
                "revision": env::var("DEP_TATBOT_TROSSEN_BRIDGE_SDK_REVISION").unwrap(),
                "library_sha256": env::var("DEP_TATBOT_TROSSEN_BRIDGE_SDK_LIBRARY_SHA256").unwrap(),
            })
        });
    let info = serde_json::json!({
        "schema": "tatbot.arm-guide-build/1",
        "source_commit": env::var("TATBOT_SOURCE_COMMIT").unwrap_or_else(|_| "development".into()),
        "sources": sources,
        "target": env::var("TARGET").unwrap(),
        "profile": env::var("PROFILE").unwrap(),
        "native_backend_compiled": env::var_os("CARGO_FEATURE_TROSSEN").is_some(),
        "sdk": sdk,
    });
    let out = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(out.join("native-build.json"), info.to_string()).unwrap();
}
