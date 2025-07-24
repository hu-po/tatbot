from tatbot.ops.base import BaseOp, BaseOpConfig


def get_op(op_name: str, node_name: str) -> tuple[BaseOp, BaseOpConfig]:
    if op_name == "base":
        return BaseOp, BaseOpConfig
    elif op_name == "align":
        from tatbot.ops.align import AlignOp, AlignOpConfig
        return AlignOp, AlignOpConfig
    elif op_name == "sense":
        if node_name not in ["trossen-ai"]:
            raise ValueError(f"Sense op requires realsense cameras and is not supported on {node_name}")
        from tatbot.ops.sense import SenseOp, SenseOpConfig
        return SenseOp, SenseOpConfig
    elif op_name == "stroke":
        from tatbot.ops.stroke import StrokeOp, StrokeOpConfig
        return StrokeOp, StrokeOpConfig
    else:
        raise ValueError(f"Unknown operation: {op_name}")