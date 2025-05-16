## Intel Realsense Camera Setup

https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

```bash
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
```

Installation on Jetson

https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md

had to put the overhead realsense on the usb port of the other side of the meerkat

testing out rust realsense library on trossen-ai pc with 2 realsenses connected via usb

```bash
cd ~ && git clone https://github.com/Tangram-Vision/realsense-rust
cd realsense-rust
cargo clean && cargo build
cargo run --example enumerate_devices
```

had to install on trossen-ai pc

```bash
sudo apt update
sudo apt install clang cargo build-essential libstdc++-12-dev libfreetype-dev libopencv-dev pkg-config
```

```bash
----
Enumerating all devices compatible with RealSense:
----
>  Intel RealSense D405      | SN: 230422273017    | Curr Fw Ver: 5.12.14.100     | Rec FW Ver: 5.16.0.1       
>  Intel RealSense D405      | SN: 218622278376    | Curr Fw Ver: 5.16.0.1        | Rec FW Ver: 5.16.0.1       
---
```

https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d405.html?srsltid=AfmBOoq2TAPRoeGQJJM2s6P3xPfYx2Bk7334eDPXgjCwgIdHGYjcPQwy

```bash
cargo run --example opencv
```

had to unplug and replug the realsense camera to get it to work

```rust
.enable_stream(Rs2StreamKind::Color, None, 1280, 0, Rs2Format::Bgr8, 30)?
.enable_stream(Rs2StreamKind::Depth, None, 1280, 0, Rs2Format::Z16, 30)
```