"""Teleoperation visualization tool for interactive robot control via IK."""

import threading

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.viz import get_server_url, register_server
from tatbot.tools.viz.models import TeleopVizInput, TeleopVizOutput
from tatbot.utils.log import get_logger
from tatbot.viz.teleop import TeleopViz, TeleopVizConfig

log = get_logger("tools.viz.teleop", "🎮")


@tool(
    name="start_teleop_viz",
    nodes=["oop", "ook"],
    description="Start a teleoperation visualization server using viser",
    input_model=TeleopVizInput,
    output_model=TeleopVizOutput,
    requires=["viz"],
)
async def start_teleop_viz(input_data: TeleopVizInput, ctx: ToolContext):
    """
    Start a teleoperation visualization server for interactive robot control via inverse kinematics.
    
    This creates an interactive 3D visualization using viser that allows:
    - Interactive end-effector control via transform controls
    - Real-time inverse kinematics solving
    - Pose saving and loading
    - Direct robot control (if enable_robot=True)
    - Calibrator positioning
    - EE offset calculation and saving
    
    The server runs in a background thread and can be stopped with stop_viz_server.
    
    Parameters:
    - scene: Scene configuration name (default: "default")
    - meta: Optional meta config to apply (default: null)
    - enable_robot: Connect to and control real robot hardware
    - enable_depth: Enable depth camera visualization
    - transform_control_scale: Scale of the interactive transform controls
    - transform_control_opacity: Opacity of the transform controls
    """
    server_name = "teleop_viz"
    
    yield {"progress": 0.1, "message": f"Initializing {server_name} server..."}
    
    yield {"progress": 0.2, "message": "Creating visualization config..."}
    
    try:
        # Convert input to TeleopVizConfig
        config = TeleopVizConfig(
            scene=input_data.scene,
            meta=input_data.meta,
            env_map_hdri=input_data.env_map_hdri,
            view_camera_position=input_data.view_camera_position,
            view_camera_look_at=input_data.view_camera_look_at,
            enable_robot=input_data.enable_robot,
            enable_depth=input_data.enable_depth,
            speed=input_data.speed,
            fps=input_data.fps,
            bind_host=input_data.bind_host,
            transform_control_scale=input_data.transform_control_scale,
            transform_control_opacity=input_data.transform_control_opacity,
            debug=input_data.debug,
        )
        
        yield {"progress": 0.4, "message": "Initializing teleoperation visualization..."}
        
        # Create viz instance
        viz = TeleopViz(config)
        
        yield {"progress": 0.6, "message": "Starting viser server..."}
        
        # Start the visualization loop in a background thread
        def run_viz():
            try:
                viz.run()
            except Exception as e:
                log.error(f"Visualization loop error: {e}")
        
        viz_thread = threading.Thread(target=run_viz, daemon=True)
        viz_thread.start()
        
        yield {"progress": 0.7, "message": "Waiting for server to be ready..."}
        
        # Wait for server to be ready
        if not viz.wait_for_ready(timeout=10.0):
            raise Exception("Server failed to start within timeout")
        
        # Register the server with thread (atomic check-and-register)
        if not register_server(server_name, viz, viz_thread):
            # Race condition - another server started between our check and registration
            viz.stop()
            viz_thread.join(timeout=5.0)
            existing_url = get_server_url(server_name, ctx.node_name) or "unknown"
            yield TeleopVizOutput(
                success=False,
                message=f"Teleop viz server already running at {existing_url}. Stop it first with stop_viz_server.",
                server_name=server_name,
                server_url=existing_url,
                running=True,
                ee_links=[],
            )
            return
        
        # Get server URL using node host
        from tatbot.tools.registry import _load_node_config
        node_config = _load_node_config(ctx.node_name)
        host = node_config.get("host", "localhost")
        server_url = f"http://{host}:{viz.server.get_port()}"
        
        yield {"progress": 0.9, "message": f"Server ready at {server_url}..."}
        
        yield {"progress": 1.0, "message": f"Teleop viz server started at {server_url}"}
        
        yield TeleopVizOutput(
            success=True,
            message=f"✅ Teleop viz server running at {server_url}",
            server_name=server_name,
            server_url=server_url,
            running=True,
            ee_links=viz.scene.urdf.ee_link_names,
        )
        
    except Exception as e:
        import traceback
        error_details = f"{type(e).__name__}: {e}"
        log.error(f"Failed to start teleop viz: {error_details}")
        log.debug(f"Traceback: {traceback.format_exc()}")
        yield TeleopVizOutput(
            success=False,
            message=f"❌ Failed to start teleop viz: {error_details}",
            server_name=server_name,
            running=False,
            ee_links=[],
        )