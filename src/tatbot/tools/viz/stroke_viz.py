"""Stroke visualization tool for visualizing StrokeBatch execution."""

import threading

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.viz import get_server_url, register_server
from tatbot.tools.viz.models import StrokeVizInput, StrokeVizOutput
from tatbot.utils.log import get_logger
from tatbot.viz.stroke import VizStrokes, VizStrokesConfig

log = get_logger("tools.viz.stroke", "🎨")


@tool(
    name="start_stroke_viz",
    nodes=["oop", "ook", "eek"],
    description="Start a stroke visualization server using viser",
    input_model=StrokeVizInput,
    output_model=StrokeVizOutput,
    requires=["viz"],
)
async def start_stroke_viz(input_data: StrokeVizInput, ctx: ToolContext):
    """
    Start a stroke visualization server to visualize StrokeBatch execution.
    
    This creates an interactive 3D visualization using viser that shows:
    - Robot arm movements during stroke execution
    - Point clouds of stroke paths
    - Real-time joint positions
    - Optional depth camera feeds
    
    The server runs in a background thread and can be stopped with stop_viz_server.
    
    Parameters:
    - scene: Scene configuration name (default: "default")
    - align: Visualize alignment strokes instead of design strokes
    - enable_robot: Connect to and control real robot hardware
    - enable_depth: Enable depth camera visualization
    - speed: Playback speed multiplier
    """
    server_name = "stroke_viz"
    
    yield {"progress": 0.1, "message": f"Initializing {server_name} server..."}
    
    yield {"progress": 0.2, "message": "Creating visualization config..."}
    
    try:
        # Convert input to VizStrokesConfig
        config = VizStrokesConfig(
            scene=input_data.scene,
            env_map_hdri=input_data.env_map_hdri,
            view_camera_position=input_data.view_camera_position,
            view_camera_look_at=input_data.view_camera_look_at,
            enable_robot=input_data.enable_robot,
            enable_depth=input_data.enable_depth,
            speed=input_data.speed,
            fps=input_data.fps,
            bind_host=input_data.bind_host,
            align=input_data.align,
            design_pointcloud_point_size=input_data.design_pointcloud_point_size,
            design_pointcloud_point_shape=input_data.design_pointcloud_point_shape,
            path_highlight_radius=input_data.path_highlight_radius,
            pose_highlight_radius=input_data.pose_highlight_radius,
            debug=input_data.debug,
        )
        
        yield {"progress": 0.4, "message": "Initializing stroke visualization..."}
        
        # Create viz instance
        viz = VizStrokes(config)
        
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
            yield StrokeVizOutput(
                success=False,
                message=f"Stroke viz server already running at {existing_url}. Stop it first with stop_viz_server.",
                server_name=server_name,
                server_url=existing_url,
                running=True,
                num_strokes=0,
                stroke_length=0,
            )
            return
        
        # Get server URL using node host
        from tatbot.tools.registry import _load_node_config
        node_config = _load_node_config(ctx.node_name)
        host = node_config.get("host", "localhost")
        server_url = f"http://{host}:{viz.server.get_port()}"
        
        yield {"progress": 0.9, "message": f"Server ready at {server_url}..."}
        
        yield {"progress": 1.0, "message": f"Stroke viz server started at {server_url}"}
        
        yield StrokeVizOutput(
            success=True,
            message=f"✅ Stroke viz server running at {server_url}",
            server_name=server_name,
            server_url=server_url,
            running=True,
            num_strokes=viz.num_strokes,
            stroke_length=viz.scene.stroke_length,
        )
        
    except Exception as e:
        import traceback
        error_details = f"{type(e).__name__}: {e}"
        log.error(f"Failed to start stroke viz: {error_details}")
        log.debug(f"Traceback: {traceback.format_exc()}")
        yield StrokeVizOutput(
            success=False,
            message=f"❌ Failed to start stroke viz: {error_details}",
            server_name=server_name,
            running=False,
            num_strokes=0,
            stroke_length=0,
        )