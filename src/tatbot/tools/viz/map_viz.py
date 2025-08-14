"""Surface mapping visualization tool for debugging 2D-to-3D mapping."""

import threading

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.viz import get_server_url, register_server
from tatbot.tools.viz.models import MapVizInput, MapVizOutput
from tatbot.utils.log import get_logger
from tatbot.viz.map import VizMap, VizMapConfig

log = get_logger("tools.viz.map", "🗺️")


@tool(
    name="start_map_viz",
    nodes=["oop", "ook", "eek"],
    description="Start a surface mapping visualization server for 2D-to-3D debugging",
    input_model=MapVizInput,
    output_model=MapVizOutput,
    requires=["viz"],
)
async def start_map_viz(input_data: MapVizInput, ctx: ToolContext):
    """
    Start a surface mapping visualization server for debugging 2D-to-3D stroke mapping.
    
    This creates an interactive 3D visualization using viser that shows:
    - Original 2D stroke designs in 3D space
    - PLY point cloud data from depth cameras
    - Interactive mesh building from point clouds
    - Stroke mapping to 3D surface mesh
    - Design pose adjustment controls
    
    The server runs in a background thread and can be stopped with stop_viz_server.
    
    Parameters:
    - scene: Scene configuration name (default: "default")
    - meta: Optional meta config to apply (default: null)
    - stroke_point_size: Size of stroke points in visualization
    - skin_ply_point_size: Size of PLY point cloud points
    - transform_control_scale: Scale of the design pose transform control
    - enable_robot: Connect to and control real robot hardware
    - enable_depth: Enable depth camera visualization
    """
    server_name = "map_viz"
    
    yield {"progress": 0.1, "message": f"Initializing {server_name} server..."}
    
    yield {"progress": 0.2, "message": "Creating visualization config..."}
    
    try:
        # Convert input to VizMapConfig
        config = VizMapConfig(
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
            stroke_point_size=input_data.stroke_point_size,
            stroke_point_shape=input_data.stroke_point_shape,
            skin_ply_point_size=input_data.skin_ply_point_size,
            skin_ply_point_shape=input_data.skin_ply_point_shape,
            transform_control_scale=input_data.transform_control_scale,
            transform_control_opacity=input_data.transform_control_opacity,
            debug=input_data.debug,
        )
        
        yield {"progress": 0.4, "message": "Initializing surface mapping visualization..."}
        
        # Create viz instance
        viz = VizMap(config)
        
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
            yield MapVizOutput(
                success=False,
                message=f"Map viz server already running at {existing_url}. Stop it first with stop_viz_server.",
                server_name=server_name,
                server_url=existing_url,
                running=True,
                num_strokes=0,
                ply_files_count=0,
            )
            return
        
        # Get server URL using node host
        from tatbot.tools.registry import _load_node_config
        node_config = _load_node_config(ctx.node_name)
        host = node_config.get("host", "localhost")
        server_url = f"http://{host}:{viz.server.get_port()}"
        
        yield {"progress": 0.9, "message": f"Server ready at {server_url}..."}
        
        yield {"progress": 1.0, "message": f"Map viz server started at {server_url}"}
        
        yield MapVizOutput(
            success=True,
            message=f"✅ Map viz server running at {server_url}",
            server_name=server_name,
            server_url=server_url,
            running=True,
            num_strokes=len(viz.strokes.strokes),
            ply_files_count=len(viz.skin_ply_files),
        )
        
    except Exception as e:
        import traceback
        error_details = f"{type(e).__name__}: {e}"
        log.error(f"Failed to start map viz: {error_details}")
        log.debug(f"Traceback: {traceback.format_exc()}")
        yield MapVizOutput(
            success=False,
            message=f"❌ Failed to start map viz: {error_details}",
            server_name=server_name,
            running=False,
            num_strokes=0,
            ply_files_count=0,
        )