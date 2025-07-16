# Cameras

- `tatbot/tatbot/data/cams.py`
- `tatbot/config/cams/fast.yaml`
- [`cv2`](https://github.com/opencv/opencv-python)

## IP PoE (ip)

Amcrest 5MP Turret POE Camera, UltraHD Outdoor IP Camera POE with Mic/Audio, 5-Megapixel Security Surveillance Cameras, 98ft NightVision, 132° FOV, MicroSD (256GB), (IP5M-T1179EW-AI-V3)

the cameras are currently set at:

- resolution: 1920x1080
- fps: 5
- bitrate CBR max 2048
- frameinterval: 10
- no substream, all watermarks off 

## Realsense (rs)

tatbot uses two [D405 Intel Realsense cameras](https://www.intelrealsense.com/depth-camera-d405/).

- [`pyrealsense2`](https://github.com/IntelRealSense/librealsense)
- both realsense cameras are connected to `trossen-ai` via usb3 port
- Follow the [calibration guide](https://dev.intelrealsense.com/docs/self-calibration-for-depth-cameras).
- Use the `rs-enumerate-devices` command to verify that both realsenses are connected. If this doesn't work, unplug and replug the realsense cameras.
- Should be calibrated out of the box, but can be recalibrated
  - [example1](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_auto_calibration_example.py)
  - [example2](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_ucal_example.py)
- [FOV differs for depth and rgb cameras](https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html)
- TODO: these will somewhat randomly fail, need to create robust exception handling