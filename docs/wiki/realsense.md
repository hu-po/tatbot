# Realsense Cameras

tatbot uses two [D405 Intel Realsense cameras](https://www.intelrealsense.com/depth-camera-d405/).

- both realsense cameras are connected to `trossen-ai` via usb3 port
- both realsense cameras are mounted on adjustable goosenecks, so their extrinsic position changes often
- Follow the [calibration guide](https://dev.intelrealsense.com/docs/self-calibration-for-depth-cameras).
- Use the `rs-enumerate-devices` command to verify that both realsenses are connected. If this doesn't work, unplug and replug the realsense cameras.
- Should be calibrated out of the box, but can be recalibrated: [example1](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_auto_calibration_example.py), [example2](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/depth_ucal_example.py).
- [FOV differs for depth and rgb cameras](https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html)
- TODO: these will somewhat randomly fail, need to create robust exception handling 