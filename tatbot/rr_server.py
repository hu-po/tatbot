# $TATBOT_ROOT/tatbot/rr_server.py
from dataclasses import dataclass
import logging
from uuid import uuid4

import rerun as rr
import rerun.blueprint as rrb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class RerunServerConfig:
    application_id: str = "tabtot"
    web_port: int = 9876 # Default Rerun web port
    ws_port: int = 9877 # Default Rerun WebSocket port (SDK connects here)
    open_browser: bool = True
    recording_id: str = f"tabtot-{str(uuid4())}"

def default_blueprint() -> rrb.Blueprint:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="scene", origin="/scene"),
            rrb.Spatial2DView(name="design", origin="/design"),
        ),
        rrb.BlueprintPanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
    return blueprint

def main():
    config = RerunServerConfig()
    log.info(f"initializing rerun server with config: {config}")
    rr.init(
        config.application_id,
        recording_id=config.recording_id,
        spawn=False,
        strict=True,
        default_blueprint=default_blueprint()
    )
    rr.serve_web_viewer(web_port=config.web_port, open_browser=config.open_browser)

if __name__ == "__main__":
    main()