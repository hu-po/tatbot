NODE_AVAILABLE_OPS: dict[str, list[str]] = {
    "trossen-ai": ["align", "sense", "stroke", "reset"],
    "ook": ["align", "stroke", "reset"],
}

def get_op(op_name: str, node_name: str) -> tuple:
    if node_name not in NODE_AVAILABLE_OPS:
        raise ValueError(f"Node {node_name} does not support any operations")
    if op_name not in NODE_AVAILABLE_OPS[node_name]:
        raise ValueError(f"Operation {op_name} is not supported on {node_name}")
    if op_name == "base":
        from tatbot.ops.base import BaseOp, BaseOpConfig
        return BaseOp, BaseOpConfig
    if op_name == "reset":
        from tatbot.ops.reset import ResetOp, ResetOpConfig
        return ResetOp, ResetOpConfig
    elif op_name == "align":
        from tatbot.ops.record_align import AlignOp, AlignOpConfig
        return AlignOp, AlignOpConfig
    elif op_name == "sense":
        from tatbot.ops.record_sense import SenseOp, SenseOpConfig
        return SenseOp, SenseOpConfig
    elif op_name == "stroke":
        from tatbot.ops.record_stroke import StrokeOp, StrokeOpConfig
        return StrokeOp, StrokeOpConfig
    else:
        raise ValueError(f"Unknown operation: {op_name}")