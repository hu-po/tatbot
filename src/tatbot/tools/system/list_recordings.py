"""List recordings tool for discovering available recordings."""

from pathlib import Path

from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.system.models import ListRecordingsInput, ListRecordingsOutput
from tatbot.utils.log import get_logger

log = get_logger("tools.list_recordings", "📼")


@tool(
    name="list_recordings",
    nodes=["rpi2"],
    description="List available recordings from the recordings directory",
    input_model=ListRecordingsInput,
    output_model=ListRecordingsOutput,
)
async def list_recordings(input_data: ListRecordingsInput, ctx: ToolContext):
    """
    List available recordings from the recordings directory.
    
    No parameters required. Returns list of available recording directory names.
    
    Example usage:
    {}
    """
    yield {"progress": 0.1, "message": "Scanning for recording directories..."}
    
    try:
        recordings_dir = Path("~/tatbot/nfs/recordings").expanduser().resolve()
        if not recordings_dir.exists():
            log.warning(f"Recordings directory not found: {recordings_dir}")
            yield ListRecordingsOutput(
                success=True,
                message="No recordings directory found",
                recordings=[],
                count=0
            )
            return
        
        yield {"progress": 0.5, "message": "Reading recording directories..."}
        
        recordings = [
            d.name
            for d in recordings_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ]
        recordings.sort()
        
        log.info(f"Found {len(recordings)} recordings")
        
        yield ListRecordingsOutput(
            success=True,
            message=f"Found {len(recordings)} available recordings",
            recordings=recordings,
            count=len(recordings)
        )
        
    except Exception as e:
        log.error(f"Error listing recordings: {e}")
        yield ListRecordingsOutput(
            success=False,
            message=f"❌ Error listing recordings: {e}",
            recordings=[],
            count=0
        )