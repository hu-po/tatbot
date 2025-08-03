"""Base MCP server abstractions."""

import os
import traceback
from typing import Optional

from pydantic import BaseModel, field_validator
from mcp.server.fastmcp import Context

from tatbot.data.base import BaseCfg
from tatbot.ops import get_op, NODE_AVAILABLE_OPS
from tatbot.utils.log import get_logger

log = get_logger("mcp.base", "🔌")


class MCPConfig(BaseCfg):
    """MCP server configuration with debug and transport settings."""
    debug: bool = False
    transport: str = "streamable-http"


class RunOpInput(BaseModel):
    """Input model for running robot operations."""
    op_name: str
    scene_name: str = "default"
    debug: bool = False

    @field_validator('op_name')
    @classmethod
    def validate_op_name(cls, v: str) -> str:
        """Validate that the operation name exists in available operations."""
        # Get all available operations across all nodes
        all_ops = set()
        for ops in NODE_AVAILABLE_OPS.values():
            all_ops.update(ops)
        
        if v not in all_ops:
            available_ops = sorted(list(all_ops))
            raise ValueError(f"Invalid op_name: {v}. Available: {available_ops}")
        return v

    @field_validator('scene_name')
    @classmethod
    def validate_scene_name(cls, v: str) -> str:
        """Validate that the scene name exists in the scenes directory."""
        # Use the same logic as the list_scenes function in rpi2.py
        scenes_dir = os.path.expanduser("~/tatbot/config/scenes")
        try:
            available_scenes = [f.replace(".yaml", "") for f in os.listdir(scenes_dir) if f.endswith(".yaml")]
            if v not in available_scenes:
                raise ValueError(f"Invalid scene_name: {v}. Available scenes: {available_scenes}")
        except FileNotFoundError:
            # If scenes directory doesn't exist, allow any scene name
            log.warning(f"Scenes directory not found: {scenes_dir}. Allowing any scene name.")
        return v

async def _run_op(input: RunOpInput, ctx: Context, node_name: str) -> str:
    """Runs an operation, yields intermediate results, see available ops in tatbot.ops module."""
    await ctx.info(f"Running robot op: {input.op_name}")
    try:
        op_class, op_config = get_op(input.op_name, node_name)
        config = op_config(scene=input.scene_name, debug=input.debug)
        op = op_class(config)
        await ctx.report_progress(
            progress=0.01, total=1.0, message=f"Created op: {config}"
        )
    except Exception:
        _msg = f"❌ Exception when creating op: {traceback.format_exc()}"
        log.error(_msg)
        return _msg    
    
    try:
        async for result in op.run():
            log.info(f"Intermediate result: {result}")
            await ctx.report_progress(progress=result['progress'], total=1.0, message=result['message'])
        _msg = f"✅ Completed {input.op_name}"
        log.info(_msg)
        return _msg
    except KeyboardInterrupt:
        _msg = "🛑⌨️ Keyboard/E-stop interrupt detected"
        log.error(_msg)
        return _msg
    except Exception:
        _msg = f"❌ Exception when running op: {traceback.format_exc()}"
        log.error(_msg)
        return _msg
    finally:
        op.cleanup()