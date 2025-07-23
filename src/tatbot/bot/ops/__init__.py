from tatbot.bot.ops.base import BaseOp, BaseOpConfig


def get_op(op_name: str) -> tuple[BaseOp, BaseOpConfig]:
    if op_name == "base":
        return BaseOp, BaseOpConfig
    elif op_name == "align":
        from tatbot.bot.ops.align import AlignOp, AlignOpConfig
        return AlignOp, AlignOpConfig
    elif op_name == "sense":
        from tatbot.bot.ops.sense import SenseOp, SenseOpConfig
        return SenseOp, SenseOpConfig
    elif op_name == "stroke":
        from tatbot.bot.ops.stroke import StrokeOp, StrokeOpConfig
        return StrokeOp, StrokeOpConfig
    else:
        raise ValueError(f"Unknown operation: {op_name}")