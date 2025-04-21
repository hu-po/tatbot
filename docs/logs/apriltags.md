# AprilTags

we use apriltags to track various external objects in the workspace:
- camera 4 (tag #4)
- camera 5 (tag #5)

# What should containers and ros2 nodes look like on the `rpis`?

Given the resource constraints of Raspberry Pi's, it seems likely that using one container per camera with all three nodes (conversion, rectification, detection) is the best approach. This reduces containers to four per Pi, manageable within 4GB RAM, and maintains per-camera isolation, crucial for fault tolerance. Communication within the container, while still involving data copying, may be slightly more efficient than between containers, reducing CPU overhead to an estimated 5-10%

To minimize memory use, ensure nodes process frames sequentially with minimal buffering, use compressed image formats, and leverage ROS 2 quality of service settings. For example, setting history depth to 1 can reduce memory for topic buffers, as per ROS 2 Documentation.

Compute Utilization Strategy
To maximize the Pi 5’s compute:
Offload Decoding: Use HEVC decoder (H.265) or GPU (H.264) via V4L2 in GSCam2.

GPU for Processing: Apply GPU to rectification and April tag detection with OpenCV or Vulkan if decoding leaves GPU capacity (H.265 case).

CPU Fallback: Use the CPU for ROS 2 overhead and non-GPU tasks.

Test and Profile: Use gst-inspect-1.0 for GStreamer elements, htop for CPU/GPU load, and ROS 2’s ros2 topic hz to verify performance.
Recommendation
Start with GPU-accelerated decoding in GSCam2 using V4L2 (e.g., v4l2h264dec), as it’s the most resource-intensive task and has direct hardware support. If cameras use H.265, the HEVC decoder frees the GPU for April tag detection, which is the next easiest to accelerate with OpenCV. Rectification follows similarly. Custom Vulkan drivers are overkill unless you hit specific performance limits with V4L2/OpenCV.

# 3-17 morning

on both rpis:

```bash
cd ~/dev/tatbot-dev && git pull
docker build -f docker/Dockerfile.rpi -t tatbot-rpi --build-arg RPI=1 .
docker run -it --rm --network=host tatbot-rpi
```

on main computer:

```bash
docker build -f docker/Dockerfile.main -t tatbot-main .
xhost +local:docker # this command does not work in Cursor Terminal?
docker run -it --rm --privileged --network=host \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
tatbot-main
```

to just inspect topics:

```bash
docker run -it --rm --privileged --network=host --env ROS_DOMAIN_ID=1 ros:humble-ros-base bash
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic hz /camera_002_image_raw
ros2 topic type /camera_002_image_raw
ros2 topic info -v /apriltag_002/image_rect
ros2 topic info /apriltag_002/image_rect
```

# 3-17 afternoon

modifying apriltag rviz config from oop

```bash
docker ps
docker cp FOO:/root/.rviz2/default.rviz /home/oop/dev/tatbot-dev/cfg/apriltag.rviz
```

rpi one-liner

```bash
cd ~/dev/tatbot-dev && git pull && ./scripts/launch.rpi1.sh
cd ~/dev/tatbot-dev && git pull && ./scripts/launch.rpi2.sh
```

on big computer:

```bash
./scripts/launch.oop-viewer.sh
```