"""Reset robot tool for returning to safe position."""

import logging

from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig

from tatbot.main import compose_and_validate_scene
from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.robot.models import ResetInput, ResetOutput
from tatbot.utils.log import get_logger

log = get_logger("tools.reset", "🔄")


@tool(
    name="reset",
    nodes=["trossen-ai", "ook", "oop"],
    description="Reset robot to safe/ready position",
    input_model=ResetInput,
    output_model=ResetOutput,
)
async def reset_tool(input_data: ResetInput, ctx: ToolContext):
    """
    Reset robot to safe/ready position.
    
    This tool disconnects and reconnects the robot, bringing it to its home (sleep) position.
    It's useful for recovering from error states or preparing for new operations.
    
    Parameters:
    - scene_name (str, optional): Scene configuration to use. Default: "default"
    - debug (bool, optional): Enable debug logging. Default: false
    
    Returns:
    - success (bool): Whether reset completed successfully
    - message (str): Status message
    
    Example usage:
    {"scene_name": "default"}
    {}
    """
    
    if input_data.debug:
        logging.getLogger("lerobot").setLevel(logging.DEBUG)
    
    yield {"progress": 0.1, "message": "Loading scene configuration..."}
    
    try:
        # Load scene configuration
        scene = compose_and_validate_scene(input_data.scene_name)
        
        yield {"progress": 0.3, "message": "🤖 Resetting robot..."}
        
        # Create robot with sleep position as home
        robot: Robot = make_robot_from_config(
            TatbotConfig(
                ip_address_l=scene.arms.ip_address_l,
                ip_address_r=scene.arms.ip_address_r,
                arm_l_config_filepath=scene.arms.arm_l_config_filepath,
                arm_r_config_filepath=scene.arms.arm_r_config_filepath,
                goal_time=scene.arms.goal_time_slow,
                connection_timeout=scene.arms.connection_timeout,
                home_pos_l=scene.sleep_pos_l.joints,
                home_pos_r=scene.sleep_pos_r.joints,
                rs_cameras={},
                ip_cameras={},
            )
        )
        
        yield {"progress": 0.8, "message": "Disconnecting robot..."}
        
        # Reset by disconnecting (robot goes to home position)
        robot.disconnect()
        
        log.info("✅ Robot reset completed successfully")
        
        yield ResetOutput(
            success=True,
            message="✅ Robot reset to safe position completed"
        )
        
    except Exception as e:
        error_msg = f"❌ Robot reset failed: {e}"
        log.error(error_msg)
        yield ResetOutput(
            success=False,
            message=error_msg
        )