# $TATBOT_ROOT/tatbot/cameras/rr_server.py

import os
import time
from dataclasses import dataclass
import logging
import rerun as rr
import rerun.blueprint as rrb
import yaml

# -----------------------------
# Configuration classes
# -----------------------------
@dataclass
class PanoramConfig:
    config_path: str = os.environ.get("TATBOT_ROOT", ".") + "/config/cameras.yaml"
    cameras: dict = None

    def __post_init__(self):
        assert os.path.exists(self.config_path), f"Camera config not found: {self.config_path}"
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        # Only need names here for blueprint
        self.cameras = list(data.keys())

# -----------------------------
# Rerun server main
# -----------------------------
@dataclass
class RerunServerConfig:
    application_id: str = "rr_panoram_server"
    web_port: int = int(os.environ.get("RERUN_WEB_PORT", 9876))
    open_browser: bool = False
    recording_id: str = os.environ.get("RERUN_RECORDING_ID", "panoram_run")
    host: str = "0.0.0.0"

# Default blueprint: one 2D panel per camera
def default_blueprint(cam_names):
    panels = []
    for name in cam_names:
        panels.append(
            rrb.Spatial2DView(name=name, origin=f"/{name}"))
    layout = rrb.Blueprint(
        rrb.Horizontal(*panels),
        rrb.TimePanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed")
    )
    return layout

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # Load camera names
    config = PanoramConfig()
    server_cfg = RerunServerConfig()
    
    log.info(f"Starting Rerun: app_id={server_cfg.application_id}, recording_id={server_cfg.recording_id}")

    # Initialize Rerun SDK
    rr.init(
        server_cfg.application_id,
        recording_id=server_cfg.recording_id,
        spawn=False,
        strict=True,
        default_blueprint=default_blueprint(config.cameras)
    )

    # Start the web viewer. This will also handle the gRPC server on the same port.
    log.info(f"Starting Rerun web viewer and gRPC server on port {server_cfg.web_port}")
    rr.serve_web_viewer(
        web_port=server_cfg.web_port,
        open_browser=server_cfg.open_browser
    )

    # Use Rerun's script_teardown to keep the server alive and handle shutdown
    updated_return_code = rr.script_teardown()
    if updated_return_code != 0:
        log.warning(f"Rerun script_teardown exited with code: {updated_return_code}")
    else:
        log.info("Rerun server shut down gracefully.")
