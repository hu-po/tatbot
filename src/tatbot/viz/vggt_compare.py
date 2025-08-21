import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from tatbot.utils.log import get_logger
from tatbot.utils.plymesh import load_ply, ply_files_from_dir
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger("viz.vggt_compare", "🔭")


@dataclass
class VGGTCompareConfig(BaseVizConfig):
    dataset_dir: str = ""
    vggt_point_size: float = 0.001
    rs_point_size: float = 0.001
    show_vggt: bool = True
    show_rs: bool = True
    show_vggt_frustums: bool = True
    show_apriltag_frustums: bool = True


class VGGTCompareViz(BaseViz):
    def __init__(self, config: VGGTCompareConfig):
        super().__init__(config)
        self.config: VGGTCompareConfig
        self.dataset_dir = Path(self.config.dataset_dir).expanduser()

        # Load RS PLYs and VGGT PLY (if present)
        self.point_clouds = {}
        rs_dir = self.dataset_dir / "pointclouds"
        vggt_ply = rs_dir / "vggt_dense.ply"
        # Load all RS ply files
        if rs_dir.exists():
            for ply_file in ply_files_from_dir(str(rs_dir)):
                if Path(ply_file).name.startswith("vggt_"):
                    continue
                pts, cols = load_ply(ply_file)
                self.point_clouds[ply_file] = self.server.scene.add_point_cloud(
                    name=f"/rs/{Path(ply_file).name}",
                    points=pts,
                    colors=cols,
                    point_size=self.config.rs_point_size,
                )
        # VGGT point cloud (single ply)
        if vggt_ply.exists():
            pts, cols = load_ply(str(vggt_ply))
            self.point_clouds[str(vggt_ply)] = self.server.scene.add_point_cloud(
                name="/vggt/dense",
                points=pts,
                colors=cols,
                point_size=self.config.vggt_point_size,
            )

        # Frustums from JSON (VGGT and AprilTag)
        self.vggt_frustums = []
        self.apriltag_frustums = []
        frustum_dir = self.dataset_dir / "metadata"
        vggt_json = frustum_dir / "vggt_frustums.json"
        apriltag_json = frustum_dir / "apriltag_frustums.json"
        if vggt_json.exists():
            try:
                data = json.loads(vggt_json.read_text())
                for cam in data:
                    # Convert camera-from-world (E) to world-from-camera pose for viser
                    import jaxlie
                    E = np.array(cam["extrinsic_3x4"], dtype=float)
                    R = E[:, :3]
                    t = E[:, 3]
                    # Invert pose: T_wc = (R|t)^{-1}
                    Rw = R.T
                    tw = -Rw @ t
                    qw, qx, qy, qz = tuple(jaxlie.SO3.from_matrix(Rw).wxyz)
                    fr = self.server.scene.add_camera_frustum(
                        f"/vggt/frustums/{cam['name']}",
                        fov=1.0,
                        aspect=1.0,
                        scale=self.config.camera_frustrum_scale,
                        color=(0, 255, 0),
                        position=tuple(tw.tolist()),
                        wxyz=(float(qw), float(qx), float(qy), float(qz)),
                    )
                    self.vggt_frustums.append(fr)
            except Exception as e:
                log.warning(f"Failed to load VGGT frustums: {e}")
        if apriltag_json.exists():
            try:
                data = json.loads(apriltag_json.read_text())
                for cam in data:
                    pos = cam["pose"]["position"]
                    wxyz = cam["pose"]["wxyz"]
                    fr = self.server.scene.add_camera_frustum(
                        f"/apriltag/frustums/{cam['name']}",
                        fov=1.0,
                        aspect=1.0,
                        scale=self.config.camera_frustrum_scale,
                        color=(255, 0, 0),
                        position=tuple(pos),
                        wxyz=tuple(wxyz),
                    )
                    self.apriltag_frustums.append(fr)
            except Exception as e:
                log.warning(f"Failed to load AprilTag frustums: {e}")

        with self.server.gui.add_folder("Comparison", expand_by_default=True):
            self.show_vggt_points = self.server.gui.add_checkbox("Show VGGT", initial_value=self.config.show_vggt)
            self.show_rs_points = self.server.gui.add_checkbox("Show RealSense", initial_value=self.config.show_rs)
            self.show_vggt_frusta = self.server.gui.add_checkbox("Show VGGT Frustums", initial_value=self.config.show_vggt_frustums)
            self.show_april_frusta = self.server.gui.add_checkbox("Show AprilTag Frustums", initial_value=self.config.show_apriltag_frustums)
            # Metrics panel
            self.metrics_text = self.server.gui.add_text("Metrics", initial_value="Loading metrics...", disabled=True)

        # Load and display metrics
        self._update_metrics()

        @self.show_vggt_points.on_update
        def _(_):
            for k, pc in self.point_clouds.items():
                if k.endswith("vggt_dense.ply") or k.endswith("/vggt/dense"):
                    pc.visible = self.show_vggt_points.value

        @self.show_rs_points.on_update
        def _(_):
            for k, pc in self.point_clouds.items():
                if not (k.endswith("vggt_dense.ply") or k.endswith("/vggt/dense")):
                    pc.visible = self.show_rs_points.value

        @self.show_vggt_frusta.on_update
        def _(_):
            for fr in self.vggt_frustums:
                fr.visible = self.show_vggt_frusta.value

        @self.show_april_frusta.on_update
        def _(_):
            for fr in self.apriltag_frustums:
                fr.visible = self.show_april_frusta.value

    def step(self):
        pass

    def _update_metrics(self):
        try:
            metrics_path = self.dataset_dir / "metadata" / "metrics.json"
            txt = []
            if metrics_path.exists():
                try:
                    data = json.loads(metrics_path.read_text(encoding="utf-8"))
                    for k, v in data.items():
                        txt.append(f"{k}: {v}")
                except Exception as e:
                    log.warning(f"Failed to read metrics.json: {e}")
            # If both frusta exist, compute mean translation deviation on overlapping names
            vggt_json = self.dataset_dir / "metadata" / "vggt_frustums.json"
            april_json = self.dataset_dir / "metadata" / "apriltag_frustums.json"
            if vggt_json.exists() and april_json.exists():
                try:
                    v = {c["name"]: c for c in json.loads(vggt_json.read_text(encoding="utf-8"))}
                    a = {c["name"]: c for c in json.loads(april_json.read_text(encoding="utf-8"))}
                    common = sorted(set(v.keys()) & set(a.keys()))
                    if common:
                        errs = []
                        for n in common:
                            # compare camera centers
                            E = np.array(v[n]["extrinsic_3x4"], dtype=float)
                            R = E[:, :3]; t = E[:, 3]
                            C_v = -R.T @ t
                            C_a = np.array(a[n]["pose"]["position"], dtype=float)
                            errs.append(np.linalg.norm(C_v - C_a))
                        if errs:
                            txt.append(f"mean_cam_center_err_m: {float(np.mean(errs)):.4f}")
                            txt.append(f"median_cam_center_err_m: {float(np.median(errs)):.4f}")
                            txt.append(f"num_overlap_cams: {len(common)}")
                except Exception as e:
                    log.warning(f"Failed to load frustum JSON: {e}")
            self.metrics_text.value = "\n".join(txt) if txt else "No metrics available"
        except Exception as e:
            log.warning(f"Failed to compute metrics: {e}")
