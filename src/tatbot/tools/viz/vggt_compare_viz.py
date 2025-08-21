"""Visualization tool to compare VGGT vs RealSense reconstructions and frustums."""

import threading

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.viz import get_server_url, register_server
from tatbot.tools.viz.models import VGGTCompareVizInput, VGGTCompareVizOutput
from tatbot.utils.log import get_logger
from tatbot.viz.vggt_compare import VGGTCompareConfig, VGGTCompareViz

log = get_logger("tools.viz.vggt_compare", "🔭")


@tool(
    name="start_vggt_compare_viz",
    nodes=["ook", "oop"],
    description="Start a visualization server to compare VGGT and RealSense outputs",
    input_model=VGGTCompareVizInput,
    output_model=VGGTCompareVizOutput,
    requires=["viz"],
)
async def start_vggt_compare_viz(input_data: VGGTCompareVizInput, ctx: ToolContext):
    server_name = "vggt_compare_viz"

    yield {"progress": 0.1, "message": f"Initializing {server_name} server..."}

    try:
        config = VGGTCompareConfig(
            scene=input_data.scene,
            meta=input_data.meta,
            env_map_hdri=input_data.env_map_hdri,
            view_camera_position=input_data.view_camera_position,
            view_camera_look_at=input_data.view_camera_look_at,
            enable_robot=input_data.enable_robot,
            enable_depth=False,
            speed=input_data.speed,
            fps=input_data.fps,
            bind_host=input_data.bind_host,
            dataset_dir=input_data.dataset_dir,
            vggt_point_size=input_data.vggt_pointcloud_point_size,
            rs_point_size=input_data.rs_pointcloud_point_size,
            debug=input_data.debug,
        )

        viz = VGGTCompareViz(config)

        def run_viz():
            try:
                viz.run()
            except Exception as e:
                log.error(f"Visualization loop error: {e}")

        viz_thread = threading.Thread(target=run_viz, daemon=True)
        viz_thread.start()

        if not viz.wait_for_ready(timeout=10.0):
            raise Exception("Server failed to start within timeout")

        if not register_server(server_name, viz, viz_thread):
            viz.stop()
            viz_thread.join(timeout=5.0)
            existing_url = get_server_url(server_name, ctx.node_name) or "unknown"
            yield VGGTCompareVizOutput(
                success=False,
                message=f"Viz server already running at {existing_url}.",
                server_name=server_name,
                server_url=existing_url,
                running=True,
            )
            return

        from tatbot.tools.registry import _load_node_config
        node_config = _load_node_config(ctx.node_name)
        host = node_config.get("host", "localhost")
        server_url = f"http://{host}:{viz.server.get_port()}"

        yield VGGTCompareVizOutput(
            success=True,
            message=f"✅ VGGT compare viz running at {server_url}",
            server_name=server_name,
            server_url=server_url,
            running=True,
        )

    except Exception as e:
        yield VGGTCompareVizOutput(
            success=False,
            message=f"❌ Failed to start VGGT compare viz: {e}",
            server_name=server_name,
            running=False,
        )

