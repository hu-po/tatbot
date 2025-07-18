from tatbot.data.cams import Cams
from tatbot.utils.log import get_logger

log = get_logger("cam.intrinsics_ip", "📱")


def get_intrinsics(
    image_paths: list[str],
    cams: Cams,
) -> Cams:
    log.info("Calculating camera intrinsics...")
    log.debug(f"cameras: {cams}")
    log.debug(f"image_paths: {image_paths}")

    # TODO : Calculate intrinsics for the poe ip cameras, perhaps using checkerboard?

    log.info("✅ Done")
    return cams
