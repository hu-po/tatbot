This directory copies the Apache-2.0 `realsense-rust` 1.3.0 crate from
crates.io, archive SHA-256
`7595b5cfd3d9c7f11419904869f85aa1813ac501bdb8a519ed34a8ba2b463609`
(upstream revision `e7aacc112d435e72c23e86de46f83dbcdfd2e14c`). The original
`LICENSE`, `AUTHORS.md`, and `README.md` are retained.

Tatbot's functional changes are limited to the package manifest and build
script, plus a feature-gated `Device::raw_device_time_ms` method and SDK-version
constant, and `Context::with_settings`, which wraps the SDK's
`rs2_create_context_ex` so an owner can enable DDS for one Ethernet camera. `AUTHORS.md` has its trailing blank line removed to pass the
repository whitespace check. The method uses the already-open pipeline device.
It requires a native librealsense2 SDK at least 2.58.4; the ordinary
`realsense` feature does not enable it. The raw counter is diagnostic and has
no established relation to frame metadata on the installed D555.
