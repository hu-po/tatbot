from dataclasses import dataclass
import logging
import os
import time

import cv2
import numpy as np
import PIL
import viser
from viser.extras import ViserUrdf
import yourdfpy

from _log import get_logger, COLORS, print_config, setup_log_with_config
from _path import PathBatch, Stroke
from _plan import Plan
from _viz import BaseViz, BaseVizConfig

log = get_logger('viz_scan')

@dataclass
class VizScanConfig(BaseVizConfig):
    debug: bool = False
    """Enable debug logging."""

    scan_dir: str = os.path.expanduser("~/tatbot/output/scans/bench")
    """Directory containing scan."""

    point_size: float = 0.001
    """Size of points in the point cloud visualization (meters)."""
    point_shape: str = "rounded"
    """Shape of points in the point cloud visualization."""


class VizScan(BaseViz):
    def __init__(self, config: VizScanConfig):
        super().__init__(config)


    def run(self):
        while True:
            pass

if __name__ == "__main__":
    args = setup_log_with_config(VizScanConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    viz = VizScan(args)
    viz.run()