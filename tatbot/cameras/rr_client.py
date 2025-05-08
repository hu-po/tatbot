# $TATBOT_ROOT/tatbot/cameras/rr_client.py

import os
import time
import cv2
import yaml
import rerun as rr
from dataclasses import dataclass
from typing import Dict

# -----------------------------
# Configuration classes
# -----------------------------
@dataclass
class Camera:
    name: str
    ip: str
    username: str
    password: str = None

@dataclass
class PanoramConfig:
    rtsp_url_suffix: str = ":554/cam/realmonitor?channel=1&subtype=0"
    config_path: str = os.environ.get("TATBOT_ROOT", ".") + "/config/cameras.yaml"
    cameras: Dict[str, Camera] = None
    video_fps: float = 15.0  # lower FPS for efficiency

    def __post_init__(self):
        assert os.path.exists(self.config_path), f"Camera config not found: {self.config_path}"
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        cams = {}
        for key, val in data.items():
            cam = Camera(name=key,
                         ip=val["ip"],
                         username=val["username"],
                         password=os.getenv(f"{key.upper()}_PASSWORD"))
            cams[key] = cam
        self.cameras = cams

    def rtsp_url(self, cam: Camera) -> str:
        return f"rtsp://{cam.username}:{cam.password}@{cam.ip}{self.rtsp_url_suffix}"

# -----------------------------
# Rerun client main
# -----------------------------
def main():
    # Load config
    config = PanoramConfig()

    # Initialize Rerun: use same recording_id across all clients for unified view
    RERUN_SERVER = os.environ.get("RERUN_SERVER", "localhost:9876")
    rr.init(
        "rr_panoram_client",
        recording_id=os.environ.get("RERUN_RECORDING_ID", "panoram_run"),
        spawn=False
    )
    rr.connect_grpc(f"rerun+http://{RERUN_SERVER}/proxy")

    # Open VideoCapture objects for each camera
    caps = {}
    for cam in config.cameras.values():
        url = config.rtsp_url(cam)
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            rr.log_text(f"error/{cam.name}", f"Failed to open {url}")
        else:
            caps[cam.name] = cap

    # Stream loop
    try:
        interval = 1.0 / config.video_fps
        while True:
            start = time.time()
            for name, cap in caps.items():
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                rr.log(f"cameras/{name}", rr.Image(frame))
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps.values():
            cap.release()

if __name__ == "__main__":
    main()
