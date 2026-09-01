use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=TATBOT_SOURCE_COMMIT");
    for git_path in ["HEAD", "refs/heads/main"] {
        if let Some(path) = Command::new("git")
            .args(["rev-parse", "--git-path", git_path])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
        {
            println!("cargo:rerun-if-changed={}", path.trim());
        }
    }
    let commit = env::var("TATBOT_SOURCE_COMMIT").ok().or_else(|| {
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_owned())
    });
    println!(
        "cargo:rustc-env=TATBOT_BUILD_GIT_SHA={}",
        commit
            .as_deref()
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown")
    );
}
