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

had to install rust, clang

```bash
sudo apt update
sudo apt install clang cargo build-essential libstdc++-12-dev libfreetype-dev
```