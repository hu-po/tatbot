from dataclasses import dataclass

from tatbot.bot.ops.record import RecordOp, RecordOpConfig
from tatbot.utils.log import get_logger

log = get_logger("bot.ops.align", "🔍")


@dataclass
class AlignOpConfig(RecordOpConfig):
    pass


class AlignOp(RecordOp):

    op_name: str = "align"

    async def run(self):
        async for progress_update in super().run():
            yield progress_update