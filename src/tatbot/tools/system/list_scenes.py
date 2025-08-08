"""List scenes tool for discovering available scene configurations."""

import os
from pathlib import Path

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.system.models import ListScenesInput, ListScenesOutput
from tatbot.utils.log import get_logger

log = get_logger("tools.list_scenes", "🎬")


@tool(
    name="list_scenes",
    nodes=["rpi2"],
    description="List available scenes from the config directory",
    input_model=ListScenesInput,
    output_model=ListScenesOutput,
)
async def list_scenes(input_data: ListScenesInput, ctx: ToolContext):
    """
    List available scenes from the config directory.
    
    No parameters required. Returns list of available scene names.
    
    Example usage:
    {}
    """
    yield {"progress": 0.1, "message": "Scanning for scene configurations..."}
    
    try:
        scenes_dir = Path("~/tatbot/src/conf/scenes").expanduser().resolve()
        if not scenes_dir.exists():
            log.warning(f"Scenes directory not found: {scenes_dir}")
            yield ListScenesOutput(
                success=True,
                message="No scenes directory found",
                scenes=[],
                count=0
            )
            return
        
        yield {"progress": 0.5, "message": "Reading scene files..."}
        
        scenes = [
            f.replace(".yaml", "") 
            for f in os.listdir(str(scenes_dir)) 
            if f.endswith(".yaml")
        ]
        scenes.sort()
        
        log.info(f"Found {len(scenes)} scenes")
        
        yield ListScenesOutput(
            success=True,
            message=f"Found {len(scenes)} available scenes",
            scenes=scenes,
            count=len(scenes)
        )
        
    except Exception as e:
        log.error(f"Error listing scenes: {e}")
        yield ListScenesOutput(
            success=False,
            message=f"❌ Error listing scenes: {e}",
            scenes=[],
            count=0
        )