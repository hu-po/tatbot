## Intel Realsense Camera Setup

https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

```bash
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
```

Installation on Jetson

https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md

had to put the overhead realsense on the usb port of the other side of the meerkat