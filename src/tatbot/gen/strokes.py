import os

from tatbot.data.scene import Scene
from tatbot.data.stroke import StrokeBatch, StrokeList
from tatbot.gen.align import make_align_strokes
from tatbot.gen.batch import strokebatch_from_strokes
from tatbot.gen.gcode import make_gcode_strokes
from tatbot.utils.log import get_logger

log = get_logger("gen.strokes", "🖌️")


def load_make_strokes(scene: Scene, dirpath: str, resume: bool = False) -> tuple[StrokeList, StrokeBatch]:
    strokes_path = os.path.join(dirpath, "strokes.yaml")
    strokebatch_path = os.path.join(dirpath, "strokebatch.safetensors")
    if resume:
        log.info(f"🔄 Resuming from {dirpath}")
        assert os.path.exists(strokes_path), f"❌ Strokes file {strokes_path} does not exist"
        assert os.path.exists(strokebatch_path), f"❌ Strokebatch file {strokebatch_path} does not exist"
        strokes: StrokeList = StrokeList.from_yaml(strokes_path)
        strokebatch: StrokeBatch = StrokeBatch.load(strokebatch_path)
    else:
        if scene.design_dir is not None:
            log.info("📂 Generating strokes from design")
            strokes: StrokeList = make_gcode_strokes(scene)
        else:
            log.info("📂 Generating generic alignment strokes")
            strokes: StrokeList = make_align_strokes(scene)
        strokes.to_yaml(strokes_path)
        strokebatch: StrokeBatch = strokebatch_from_strokes(scene=scene, strokelist=strokes)
        strokebatch.save(strokebatch_path)
    log.info(f"✅ Loaded {len(strokes.strokes)} strokes")
    return strokes, strokebatch
