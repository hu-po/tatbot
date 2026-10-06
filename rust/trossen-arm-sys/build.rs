fn main() {
    println!("cargo:rerun-if-env-changed=TROSSEN_ARM_SDK_ROOT");
    #[cfg(feature = "sdk")]
    build_sdk();
}

#[cfg(feature = "sdk")]
fn build_sdk() {
    use sha2::{Digest, Sha256};
    use std::{env, path::PathBuf, process::Command};
    let root = env::var_os("TROSSEN_ARM_SDK_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("../../cpp/teleop/build/_deps/trossen_arm_sdk-src")
        });
    let root = root
        .canonicalize()
        .expect("pinned SDK absent; configure cpp/teleop or set TROSSEN_ARM_SDK_ROOT");
    let head = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("read SDK revision");
    assert!(
        head.status.success()
            && String::from_utf8_lossy(&head.stdout).trim()
                == "fdfd9f68f57b3bd05c4e85011fa5c11296525b2f",
        "SDK must be pinned v1.8.5"
    );
    let dirty = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .expect("read SDK status");
    assert!(
        dirty.status.success() && dirty.stdout.is_empty(),
        "SDK tracked sources are modified"
    );
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "linux",
        "SDK platform"
    );
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    assert!(
        matches!(arch.as_str(), "x86_64" | "aarch64"),
        "SDK architecture"
    );
    let lib = root.join("lib/linux").join(&arch);
    assert!(
        lib.join("libtrossen_arm.a").is_file(),
        "SDK static library missing"
    );
    println!("cargo:sdk_version=1.8.5");
    println!("cargo:sdk_revision=fdfd9f68f57b3bd05c4e85011fa5c11296525b2f");
    println!(
        "cargo:sdk_library_sha256={:x}",
        Sha256::digest(std::fs::read(lib.join("libtrossen_arm.a")).expect("SDK library"))
    );
    cxx_build::bridge("src/lib.rs")
        .file("src/bridge.cc")
        .include(root.join("include"))
        .std("c++17")
        .warnings(true)
        .compile("tatbot_trossen_bridge");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=trossen_arm");
    println!("cargo:rustc-link-lib=pthread");
    for path in ["src/lib.rs", "src/bridge.cc", "include/bridge.h"] {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rerun-if-changed={}", root.join("include").display());
    println!(
        "cargo:rerun-if-changed={}",
        lib.join("libtrossen_arm.a").display()
    );
}
