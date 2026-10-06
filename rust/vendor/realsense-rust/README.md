# RealSense Bindings for Rust

The project provides high-level bindings (crate `realsense_rust`) to librealsense2 library as well
as low-level FFI (crate `realsense_sys`) interface.

**Default bindings are for librealsense version: 2.56.5**

This project is hosted on both [Github](https://github.com/Tangram-Vision/realsense-rust) and
[Gitlab](https://gitlab.com/tangram-vision/oss/realsense-rust/). While we're happy to receive pull /
merge requests on either platform, we focus most of our work on Gitlab, so please submit an issue
there if you've found something we need to improve or have a question regarding how things work.

## Getting Started

Make sure the current librealsense version above is installed on your system. Visit the
[RealSense official repository](https://github.com/IntelRealSense/librealsense) to download and
install this on the host machine.

Once that's done, add this crate to your project's `Cargo.toml`.

## Examples and Usage

Check out the examples folder for helpful snippets of code, as well as minimal configurations that
fit some of the most popular RealSense devices. For more explanation, see the crate documentation.

### Features

Use these by running `cargo run --features <name of feature>`

-   **buildtime-bindgen**: Generate Rust bindings during build time.
-   **device-test**: Enable tests that requires connections to RealSense devices.

## Regenerating the API Bindings

_Non-Linux users_: The current bindings are formatted for Linux. Users on systems other than Linux
must run with the `buildtime-bindgen` feature to reformat the bindings. See the README in
realsense-sys for more.

_Backwards compatibility_: If you're using an older librealsense version, you may enable the
`buildtime-bindgen` feature to re-generate the bindings. We make no claims of backwards
compatibility; good luck.

## Special Considerations

-   **USB Current Draw**: Many RealSense devices draw more current than a standard USB cable can
    provide. For example, standard USB can run 0.9 amps, while the RealSense 435i draws 2 amps.
    Using a USB cable that doesn't have the right current capability will interfere with the USB
    connection on the host, and the device will seem to disconnect. A device power cycle doesn't
    always remedy this, either. In many cases, the host USB hub itself will need a reset. Make sure
    any USB cables used are able to draw at least 2 amps. Read more on the issue
    [here](https://support.intelrealsense.com/hc/en-us/community/posts/360033595714-D435-USB-connection-issues).

-   **USB Bandwidth**: When a device is connected, librealsense will measure the transmission speed
    of data across its USB connection. USB3 speeds can handle all streams running simultaneously.
    USB2 speeds _cannot_; trying to set a streaming configuration that is too much for USB2 will
    result in a failed streaming config, and will cause the program to fail. Luckily, this
    information can be looked up and compensated for during runtime. See the
    [device-specific demo examples](examples/) for ways to achieve this.

-   **Supported but Ignored Stream Options**: There are a few Sensor options that are registered as
    "supported" by the sensor, but are actually just set to their default values on runtime. These
    options are listed and tested in
    [check_supported_but_ignored_sensor_options](./tests/connectivity_l500.rs) device tests.
    Currently, `Rs2Option::GlobalTimeEnabled` on the L500 is the only setting known to suffer from
    this. However, the test has been written in a way that makes it easy to test more Options for
    this same behavior.

## Realsense-sys: A low-level API

The realsense-sys crate provides C bindings generated from librealsense headers. See the
realsense-sys [README](./realsense-sys/README.md) for more information.

## Design Philosophy

There's a lot of thought that went into making this library Rust-safe. Check out the
[architecture](./src/docs/architecture.rs) doc for our thoughts on Rust safety, error handling, and
more for this API.

## Contributing

First, check out our [contributing guidelines](CONTRIBUTING.md). After that, make sure that you read
through the documentation in [lib.rs](src/lib.rs) as well as any of the modules you might be
interested in contributing to! If you find documentation missing, this is considered a bug, so
please submit a bug report!

## License

Apache 2.0. See [LICENSE](./LICENSE) file.
