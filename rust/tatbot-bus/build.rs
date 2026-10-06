//! Embed the deployment's fleet map. A checkout that describes no fleet (the
//! public export) builds against `config/examples/nodes.json` instead.

use std::path::PathBuf;

fn main() {
    let config = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../config");
    let deployment = config.join("nodes.json");
    let nodes = if deployment.is_file() {
        println!("cargo:rerun-if-changed={}", deployment.display());
        deployment
    } else {
        // Watch the directory so a fleet map added later is picked up.
        println!("cargo:rerun-if-changed={}", config.display());
        config.join("examples/nodes.json")
    };
    println!("cargo:rerun-if-changed={}", nodes.display());
    println!("cargo:rustc-env=TATBOT_NODES_JSON={}", nodes.display());
}
