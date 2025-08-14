"""GPU-accelerated stroke conversion tool."""

import time
from pathlib import Path

import hydra
import yaml

from tatbot.data.stroke import StrokeList
from tatbot.gen.batch import strokebatch_from_strokes
from tatbot.main import compose_and_validate_scene
from tatbot.tools.base import ToolContext
from tatbot.tools.gpu.models import ConvertStrokesInput, ConvertStrokesOutput
from tatbot.tools.registry import tool
from tatbot.utils.exceptions import TatbotError
from tatbot.utils.log import get_logger

log = get_logger("tools.convert_strokes", "🎨⚡")


@tool(
    name="convert_strokelist_to_batch", 
    nodes=["ook", "oop"],
    description="Convert StrokeList to StrokeBatch using GPU-accelerated inverse kinematics",
    input_model=ConvertStrokesInput,
    output_model=ConvertStrokesOutput,
    requires=["gpu"],
)
async def convert_strokes(input_data: ConvertStrokesInput, ctx: ToolContext):
    """
    Convert StrokeList to StrokeBatch using GPU-accelerated inverse kinematics.
    
    This tool performs cross-node GPU processing for stroke trajectory conversion.
    Robot operations on non-GPU nodes automatically route stroke conversion requests
    to GPU-enabled nodes via this MCP tool. The system handles NFS path translation
    between different node mount points for seamless file sharing.
    
    GPU Requirements:
    - Only available on nodes with 'gpu' in their extras configuration
    - Uses JAX with GPU acceleration for inverse kinematics solving
    - Automatically selected by GPUProxy when local GPU unavailable
    
    NFS Integration:
    - Files are shared via NFS mount points specific to each node
    - Path translation handles canonical /nfs/tatbot mount point across all nodes
    - Input and output files remain on shared storage throughout process
    
    Parameters:
    - strokes_file_path (str, required): Path to strokes YAML file on shared NFS
    - strokebatch_file_path (str, required): Output path for strokebatch on shared NFS  
    - scene_name (str, required): Scene configuration name for IK parameters
    - first_last_rest (bool, optional): Apply rest positions at stroke endpoints. Default: true
    - use_ee_offsets (bool, optional): Apply end-effector offset corrections. Default: true
    
    Returns:
    - success (bool): Whether GPU conversion succeeded
    - message (str): Status message with output file path
    - strokebatch_base64 (str): Empty (file saved to NFS, not transferred)
    
    Cross-Node Usage:
    This tool is automatically called by robot operations (align, stroke) when:
    1. Local node lacks GPU support (check_local_gpu() returns False)
    2. GPUProxy routes request to available GPU node (currently 'ook')
    3. Paths are translated to target node's NFS mount point
    4. Remote MCP session established with proper JSON-RPC protocol
    5. Conversion performed on GPU node, file saved to shared NFS
    
    """
    
    # Verify GPU support on this node
    cfg = hydra.compose(config_name="config")
    
    # Load node-specific MCP config
    config_dir = Path(__file__).parent.parent.parent.parent / "conf" / "mcp"
    node_config_file = config_dir / f"{ctx.node_name}.yaml"
    
    if node_config_file.exists():
        with open(node_config_file, 'r') as f:
            node_cfg = yaml.safe_load(f)
    else:
        node_cfg = cfg.mcp
    
    if "gpu" not in node_cfg.get("extras", []):
        yield ConvertStrokesOutput(
            success=False,
            message=f"Node {ctx.node_name} does not have GPU support",
            strokebatch_base64=""
        )
        return
    
    await ctx.info(f"Converting StrokeList to StrokeBatch on GPU node {ctx.node_name}")
    
    try:
        # Wait for NFS file synchronization
        yield {"progress": 0.1, "message": "Waiting for NFS file synchronization..."}
        
        strokes_path = Path(input_data.strokes_file_path)
        max_wait_time = 10
        start_time = time.time()
        
        while not strokes_path.exists():
            if time.time() - start_time > max_wait_time:
                raise FileNotFoundError(f"Strokes file not found after {max_wait_time}s: {strokes_path}")
            time.sleep(0.5)
        
        yield {"progress": 0.15, "message": "Verifying array files are synced..."}
        time.sleep(1.0)
        
        # Load strokes and scene
        yield {"progress": 0.2, "message": "Loading strokes and scene configuration..."}
        
        strokes = StrokeList.from_yaml_with_arrays(input_data.strokes_file_path)
        scene = compose_and_validate_scene(input_data.scene_name)
        
        yield {"progress": 0.3, "message": f"Loaded {len(strokes.strokes)} strokes for scene '{input_data.scene_name}'"}
        
        # Perform GPU-accelerated conversion
        yield {"progress": 0.4, "message": "Starting GPU-accelerated inverse kinematics..."}
        
        strokebatch = strokebatch_from_strokes(
            scene, 
            strokes, 
            first_last_rest=input_data.first_last_rest,
            use_ee_offsets=input_data.use_ee_offsets
        )
        
        yield {"progress": 0.8, "message": "Conversion complete, saving to shared NFS..."}
        
        strokebatch.save(input_data.strokebatch_file_path)
        
        yield {"progress": 1.0, "message": "GPU conversion successful"}
        
        yield ConvertStrokesOutput(
            success=True,
            message=f"Successfully converted StrokeList to StrokeBatch at {input_data.strokebatch_file_path}",
            strokebatch_base64=""
        )
        
    except FileNotFoundError as e:
        log.error(f"File not found during conversion: {e}")
        yield ConvertStrokesOutput(
            success=False,
            message=f"File not found: {e}",
            strokebatch_base64=""
        )
    except PermissionError as e:
        log.error(f"Permission error during conversion: {e}")
        yield ConvertStrokesOutput(
            success=False,
            message=f"Permission error: {e}",
            strokebatch_base64=""
        )
    except TatbotError as e:
        log.error(f"Tatbot error during conversion: {e}")
        yield ConvertStrokesOutput(
            success=False,
            message=f"Tatbot error: {e}",
            strokebatch_base64=""
        )
    except Exception as e:
        log.error(f"Unexpected error converting strokelist to batch: {e}")
        yield ConvertStrokesOutput(
            success=False,
            message=f"Unexpected conversion error: {e}",
            strokebatch_base64=""
        )