//! Probes the librealsense2 SDK version when the raw device clock feature is on.

/// Record the SDK version the raw device clock probe was built against.
fn main() {
    if std::env::var_os("CARGO_FEATURE_RAW_DEVICE_CLOCK").is_none() {
        return;
    }
    let sdk = pkg_config::Config::new()
        .atleast_version("2.58.4")
        .probe("realsense2")
        .expect("raw device clock requires librealsense2 SDK >= 2.58.4");
    println!("cargo:rustc-env=TATBOT_RAW_CLOCK_SDK_VERSION={}", sdk.version);
}
