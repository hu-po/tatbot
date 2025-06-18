from dataclasses import dataclass, field
import os
import math

import numpy as np
import yaml
from PIL import Image
import jax.numpy as jnp

from _ik import batch_ik
from _ink import InkCap, InkPalette
from _log import get_logger
from _path import Path, PathBatch, PixelPath

log = get_logger('_plan')

# plan objects stored inside folder, these are the filenames
METADATA_FILENAME: str = "meta.yaml"
IMAGE_FILENAME: str = "image.png"
PATHS_FILENAME: str = "paths.safetensors"
PIXELPATHS_FILENAME: str = "pixelpaths.yaml"
PATHSTATS_FILENAME: str = "pathstats.yaml"
INKPALETTE_FILENAME: str = "inkpalette.yaml"

@dataclass
class Plan:
    name: str = "plan"
    """Name of the plan."""

    dirpath: str = ""
    """Path to the directory containing the plan files."""

    path_descriptions: dict[str, str] = field(default_factory=dict)
    """Descriptions for each path in the plan."""

    image_width_m: float = 0.04
    """Width of the image in meters."""
    image_height_m: float = 0.04
    """Height of the image in meters."""
    image_width_px: int | None = None
    """Width of the image in pixels."""
    image_height_px: int | None = None
    """Height of the image in pixels."""

    ik_batch_size: int = 256
    """Batch size for IK computation."""
    path_pad_len: int = 128
    """Length to pad paths to."""
    path_dt_fast: float = 0.1
    """Time between poses in seconds for fast movement."""
    path_dt_slow: float = 2.0
    """Time between poses in seconds for slow movement."""

    ee_design_pos: tuple[float, float, float] = (0.08, 0.0, 0.04)
    """position of the design ee transform."""
    ee_design_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the design ee transform."""

    hover_offset: tuple[float, float, float] = (0.0, 0.0, 0.006)
    """position offset when hovering over point, relative to current ee frame."""
    needle_offset: tuple[float, float, float] = (0.0, 0.0, -0.0065)
    """position offset to ensure needle touches skin, relative to current ee frame."""

    view_offset: tuple[float, float, float] = (0.0, -0.16, 0.16)
    """position offset when viewing design with right arm (relative to design ee frame)."""
    ee_view_wxyz: tuple[float, float, float, float] = (0.67360666, -0.25201478, 0.24747439, 0.64922119)
    """orientation quaternion (wxyz) of the view ee transform."""

    inkpalette: InkPalette = field(default_factory=InkPalette)
    """Ink palette to use for the plan."""
    ee_inkpalette_pos: tuple[float, float, float] = (0.16, 0.0, 0.04)
    """position of the inkpalette ee transform."""
    ee_inkpalette_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the inkpalette ee transform."""
    inkdip_hover_offset: tuple[float, float, float] = (0.0, 0.0, 0.03)
    """position offset when hovering over inkcap, relative to current ee frame."""
    pathlen_per_inkdip: int = 64
    """Number of poses (path length) per inkdip."""

    @classmethod
    def from_yaml(cls, dirpath: str) -> "Plan":
        log.info(f"⚙️ Loading plan from {dirpath}...")
        filepath = os.path.join(dirpath, METADATA_FILENAME)
        with open(filepath, "r") as f:
            return cls(**yaml.safe_load(f))

    def load_image_np(self) -> np.ndarray:
        filepath = os.path.join(self.dirpath, IMAGE_FILENAME)
        return np.array(Image.open(filepath).convert("RGB"))
    
    def load_pathbatch(self) -> 'PathBatch':
        filepath = os.path.join(self.dirpath, PATHS_FILENAME)
        return PathBatch.load(filepath)

    def load_pixelpaths(self) -> list[PixelPath]:
        filepath = os.path.join(self.dirpath, PIXELPATHS_FILENAME)
        with open(filepath, "r") as f:
            return [PixelPath.from_dict(p) for p in yaml.safe_load(f)]

    def load_pathstats(self) -> dict:
        filepath = os.path.join(self.dirpath, PATHSTATS_FILENAME)
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    
    def save(self, image: np.ndarray = None):
        log.info(f"⚙️💾 Saving plan to {self.dirpath}")
        os.makedirs(self.dirpath, exist_ok=True)

        meta_path = os.path.join(self.dirpath, METADATA_FILENAME)
        log.info(f"⚙️💾 Saving metadata to {meta_path}")
        # Convert inkpalette to dict for YAML serialization
        meta_dict = self.__dict__.copy()
        if isinstance(meta_dict.get('inkpalette'), InkPalette):
            meta_dict['inkpalette'] = {'inkcaps': {k: v.to_dict() for k, v in self.inkpalette.inkcaps.items()}}
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta_dict, f)

        if image is not None:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_path = os.path.join(self.dirpath, IMAGE_FILENAME)
            log.info(f"⚙️💾 Saving image to {image_path}")
            image.save(image_path)

    def add_pixelpaths(self, pixelpaths: list[PixelPath], image: Image):
        num_paths = len(pixelpaths)
        log.info(f"⚙️ Adding {num_paths} pixel paths...")

        log.debug(f"⚙️ Image shape: {image.size}")
        self.image_width_px = image.size[0]
        self.image_height_px = image.size[1]
        scale_x = self.image_width_m / self.image_width_px
        scale_y = self.image_height_m / self.image_height_px

        pixelpaths_path = os.path.join(self.dirpath, PIXELPATHS_FILENAME)
        log.debug(f"⚙️💾 Saving pixelpaths to {pixelpaths_path}...")
        with open(pixelpaths_path, "w") as f:
            yaml.safe_dump([p.to_dict() for p in pixelpaths], f)

        paths = []
        for path_idx, pixelpath in enumerate(pixelpaths):
            log.debug(f"⚙️ Adding pixelpath {path_idx} of {num_paths}...")
            path = Path.padded(self.path_pad_len)
            self.path_descriptions[f'path_{path_idx:03d}'] = pixelpath.description

            pixelpath_length = len(pixelpath.pixels)
            if pixelpath_length + 2 > self.path_pad_len:
                # TODO: resample to fit within pad_len
                log.warning(f"⚙️⚠️ pixelpath {path_idx} has more than {self.path_pad_len} poses, truncating...")
                pixelpath.pixels = pixelpath.pixels[:self.path_pad_len - 2] # -2 for hover positions
                pixelpath_length = len(pixelpath.pixels)

            for i, (pw, ph) in enumerate(pixelpath.pixels):
                # pixel coordinates first need to be converted to meters
                x_m, y_m = pw * scale_x, ph * scale_y
                # center in design frame, add needle offset
                path.ee_pos_l[i + 1, :] = [
                    self.ee_design_pos[0] + x_m - self.image_width_m / 2,
                    self.ee_design_pos[1] + y_m - self.image_height_m / 2,
                    self.ee_design_pos[2] + self.needle_offset[2],
                ]
                path.ee_wxyz_l[i + 1, :] = self.ee_design_wxyz
                # right hand just stares at center of design frame
                path.ee_pos_r[i + 1, :] = [
                    self.ee_design_pos[0] + self.view_offset[0],
                    self.ee_design_pos[1] + self.view_offset[1],
                    self.ee_design_pos[2] + self.view_offset[2],
                ]
                path.ee_wxyz_r[i + 1, :] = self.ee_view_wxyz

            # add hover positions to the beginning and end of the path
            path.ee_pos_l[0, :] = [
                path.ee_pos_l[1, 0] + self.hover_offset[0],
                path.ee_pos_l[1, 1] + self.hover_offset[1],
                path.ee_pos_l[1, 2] + self.hover_offset[2],
            ]
            path.ee_wxyz_l[0, :] = self.ee_design_wxyz
            path.ee_pos_l[pixelpath_length + 1, :] = [
                path.ee_pos_l[pixelpath_length, 0] + self.hover_offset[0],
                path.ee_pos_l[pixelpath_length, 1] + self.hover_offset[1],
                path.ee_pos_l[pixelpath_length, 2] + self.hover_offset[2],
            ]
            path.ee_wxyz_l[pixelpath_length + 1, :] = self.ee_design_wxyz
            # right hand has no hover offset, just use the first and last valid poses
            path.ee_pos_r[0, :] = path.ee_pos_r[1, :]
            path.ee_wxyz_r[0, :] = path.ee_wxyz_r[1, :]
            path.ee_pos_r[pixelpath_length + 1, :] = path.ee_pos_r[pixelpath_length, :]
            path.ee_wxyz_r[pixelpath_length + 1, :] = path.ee_wxyz_r[pixelpath_length, :]
            # slow movement at the hover positions
            path.dt[0] = self.path_dt_slow
            path.dt[1:pixelpath_length + 1] = self.path_dt_fast
            path.dt[pixelpath_length + 1] = self.path_dt_slow
            # set mask: 1 for all valid points (hover + path)
            path.mask[:pixelpath_length + 2] = 1

            paths.append(path)

        # ---- Batch IK ----
        flat_target_pos   : list[list[np.ndarray]] = []
        flat_target_wxyz  : list[list[np.ndarray]] = []
        index_map: list[tuple[int, int]] = [] # (path_idx, pose_idx)
        for p_idx, path in enumerate(paths):
            for pose_idx in range(path.ee_pos_l.shape[0]):
                # Skip padded entries (both arms at zero => unused slot)
                if (np.allclose(path.ee_pos_l[pose_idx], 0.0) and
                    np.allclose(path.ee_pos_r[pose_idx], 0.0)):
                    continue
                index_map.append((p_idx, pose_idx))
                flat_target_pos.append(
                    [path.ee_pos_l[pose_idx], path.ee_pos_r[pose_idx]]
                )
                flat_target_wxyz.append(
                    [path.ee_wxyz_l[pose_idx], path.ee_wxyz_r[pose_idx]]
                )
        target_pos   = jnp.array(flat_target_pos)    # (B, 2, 3)
        target_wxyz  = jnp.array(flat_target_wxyz)   # (B, 2, 4)
        for start in range(0, target_pos.shape[0], self.ik_batch_size):
            end = start + self.ik_batch_size
            batch_pos   = target_pos[start:end]       # (b, 2, 3)
            batch_wxyz  = target_wxyz[start:end]      # (b, 2, 4)
            batch_joints = batch_ik(
                target_wxyz=batch_wxyz,
                target_pos=batch_pos,
            )                                         # (b, 16)
            # write results back into the corresponding path / pose slots
            for local_idx, joints in enumerate(batch_joints):
                p_idx, pose_idx = index_map[start + local_idx]
                paths[p_idx].joints[pose_idx] = np.asarray(joints, dtype=np.float32)

        # overwrites image and metadata
        self.save(image)
        pathbatch = PathBatch.from_paths(paths)
        pathbatch.save(os.path.join(self.dirpath, PATHS_FILENAME))

        # compute path stats
        path_lengths_px = [
            sum(np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(path.pixels[:-1], path.pixels[1:]))
            if len(path.pixels) > 1 else 0.0
            for path in pixelpaths
        ]
        # metric lengths
        path_lengths_m = [
            float(np.sum(np.linalg.norm(np.diff(pathbatch.ee_pos_l[i][pathbatch.mask[i] == 1], axis=0), axis=1)))
            if np.sum(pathbatch.mask[i]) > 1 else 0.0
            for i in range(pathbatch.ee_pos_l.shape[0])
        ]
        stats = {
            "count": len(path_lengths_px),
            "min_px": float(np.min(path_lengths_px)) if path_lengths_px else 0.0,
            "max_px": float(np.max(path_lengths_px)) if path_lengths_px else 0.0,
            "mean_px": float(np.mean(path_lengths_px)) if path_lengths_px else 0.0,
            "sum_px": float(np.sum(path_lengths_px)) if path_lengths_px else 0.0,
            "min_m": float(np.min(path_lengths_m)) if path_lengths_m else 0.0,
            "max_m": float(np.max(path_lengths_m)) if path_lengths_m else 0.0,
            "mean_m": float(np.mean(path_lengths_m)) if path_lengths_m else 0.0,
            "sum_m": float(np.sum(path_lengths_m)) if path_lengths_m else 0.0,
        }
        pathstats_path = os.path.join(self.dirpath, PATHSTATS_FILENAME)
        log.debug(f"⚙️💾 Saving pathstats to {pathstats_path}...")
        with open(pathstats_path, "w") as f:
            yaml.safe_dump(stats, f)

        # add inkdips
        self.add_inkdips()

    def add_inkdips(self) -> None:
        """
        Append short inkdip paths plus dummy PixelPath entries so that:
        - every inkdip is executed by the robot (in `paths.safetensors`);
        - indexing in run_viz stays 1:1 (`pixelpaths.yaml` length matches `PathBatch` length);
        - leave `pathstats.yaml` unchanged.

        A dip path is 5 poses, all executed with the slow dt:

              0 ─ hover  (palette_frame + inkdip_hover_offset)
              1 ─ rim    (top of inkcap)
              2 ─ half   (half-depth of inkcap)
              3 ─ rim    (top of inkcap)
              4 ─ hover  (palette_frame + inkdip_hover_offset)

        Right arm keeps the "view" pose for the whole dip.
        A dip is inserted after drawing `pathlen_per_inkdip` poses have been accumulated over drawing paths.
        """
        # --- 1. Load existing artefacts -------------------------------------------------
        pixelpaths: list[PixelPath] = self.load_pixelpaths()
        pathbatch: PathBatch = self.load_pathbatch()
        pad_len = self.path_pad_len

        # Convert PathBatch arrays to NumPy for easy concatenation
        epl, epr = np.asarray(pathbatch.ee_pos_l),  np.asarray(pathbatch.ee_pos_r)
        ewl, ewr = np.asarray(pathbatch.ee_wxyz_l), np.asarray(pathbatch.ee_wxyz_r)
        joints   = np.asarray(pathbatch.joints)
        dt_arr   = np.asarray(pathbatch.dt)
        mask_arr = np.asarray(pathbatch.mask)

        # --- 2. Helper to build one dip path -------------------------------------------
        right_fixed_pos = np.array(
            [self.ee_design_pos[0] + self.view_offset[0],
             self.ee_design_pos[1] + self.view_offset[1],
             self.ee_design_pos[2] + self.view_offset[2]],
            dtype=np.float32,
        )
        right_fixed_wxyz = np.array(self.ee_view_wxyz, dtype=np.float32)

        def _make_dip_path(cap_name: str, cap: InkCap) -> dict[str, np.ndarray]:
            """Return dict containing all Path fields (pad_len × … ndarray)."""
            base = np.array(self.ee_inkpalette_pos) + np.array(cap.palette_pos)
            hover = base + np.array(self.inkdip_hover_offset)
            rim   = base
            half  = base + np.array([0.0, 0.0, -cap.depth_m * 0.5])

            poses_l = np.stack([hover, rim, half, rim, hover], axis=0)
            poses_r = np.tile(right_fixed_pos, (5, 1))

            epl_dip = np.zeros((pad_len, 3), dtype=np.float32)
            epr_dip = np.zeros_like(epl_dip)
            ewl_dip = np.tile(np.array(self.ee_inkpalette_wxyz, dtype=np.float32), (pad_len, 1))
            ewr_dip = np.tile(right_fixed_wxyz, (pad_len, 1))
            dt_dip  = np.zeros((pad_len,), dtype=np.float32)
            msk_dip = np.zeros((pad_len,), dtype=np.int32)
            jnt_dip = np.zeros((pad_len, 16), dtype=np.float32)

            # write first 5 poses
            epl_dip[:5] = poses_l
            epr_dip[:5] = poses_r
            dt_dip[:5]  = self.path_dt_slow
            msk_dip[:5] = 1
            # orientations already correct in ewl_dip / ewr_dip

            return dict(
                ee_pos_l=epl_dip,
                ee_pos_r=epr_dip,
                ee_wxyz_l=ewl_dip,
                ee_wxyz_r=ewr_dip,
                joints=jnt_dip,
                dt=dt_dip,
                mask=msk_dip,
                description=f"inkdip_{cap_name} ({cap.color})",
                color=cap.color,
            )

        # --- 3. Walk paths, insert dips -------------------------------------------------
        new_pixelpaths: list[PixelPath] = []
        dip_paths_raw: list[dict[str, np.ndarray]] = []

        pose_counter = 0
        for p_idx, p in enumerate(pixelpaths):
            new_pixelpaths.append(p)
            pose_counter += len(p)            # count drawing poses

            if pose_counter >= self.pathlen_per_inkdip:
                # choose an ink‑cap that matches current path colour (fallback to large_0)
                chosen_name = None
                for name, cap in self.inkpalette.inkcaps.items():
                    if cap.color.lower() == p.color.lower():
                        chosen_name = name
                        break
                if chosen_name is None:
                    chosen_name = "large_0"
                cap = self.inkpalette.inkcaps[chosen_name]

                dip = _make_dip_path(chosen_name, cap)
                dip_paths_raw.append(dip)

                # dummy PixelPath keeps indexing intact
                new_pixelpaths.append(
                    PixelPath(
                        pixels=[],               # nothing to draw on the 2‑D image
                        color=cap.color,
                        description=dip["description"],
                    )
                )
                pose_counter = 0  # reset after dipping

        # No dips? nothing to do
        if not dip_paths_raw:
            return

        # --- 4. Batch‑IK for the dip paths ---------------------------------------------
        flat_pos, flat_wxyz, idx_map = [], [], []
        for dp_idx, dip in enumerate(dip_paths_raw):
            for pose_idx in range(pad_len):
                if dip["mask"][pose_idx] == 0:
                    continue
                flat_pos.append(
                    [dip["ee_pos_l"][pose_idx], dip["ee_pos_r"][pose_idx]]
                )
                flat_wxyz.append(
                    [dip["ee_wxyz_l"][pose_idx], dip["ee_wxyz_r"][pose_idx]]
                )
                idx_map.append((dp_idx, pose_idx))

        if flat_pos:  # should always be true
            import jax.numpy as jnp
            sols = batch_ik(
                target_wxyz=jnp.array(flat_wxyz),
                target_pos=jnp.array(flat_pos),
            )

            for s_idx, joints_sol in enumerate(np.asarray(sols)):
                dp, ps = idx_map[s_idx]
                dip_paths_raw[dp]["joints"][ps] = joints_sol.astype(np.float32)

        # --- 5. Stitch arrays and overwrite files --------------------------------------
        # concatenate old arrays with new dip arrays
        for field in ("ee_pos_l", "ee_pos_r", "ee_wxyz_l", "ee_wxyz_r",
                      "joints", "dt", "mask"):
            dip_stack = np.stack([d[field] for d in dip_paths_raw], axis=0)

        epl_new = np.concatenate([epl, dip_stack := np.stack([d["ee_pos_l"] for d in dip_paths_raw])], axis=0)
        epr_new = np.concatenate([epr, np.stack([d["ee_pos_r"] for d in dip_paths_raw])], axis=0)
        ewl_new = np.concatenate([ewl, np.stack([d["ee_wxyz_l"] for d in dip_paths_raw])], axis=0)
        ewr_new = np.concatenate([ewr, np.stack([d["ee_wxyz_r"] for d in dip_paths_raw])], axis=0)
        jnt_new = np.concatenate([joints, np.stack([d["joints"] for d in dip_paths_raw])], axis=0)
        dt_new  = np.concatenate([dt_arr, np.stack([d["dt"] for d in dip_paths_raw])], axis=0)
        msk_new = np.concatenate([mask_arr, np.stack([d["mask"] for d in dip_paths_raw])], axis=0)

        # rebuild PathBatch and save
        new_batch = PathBatch(
            ee_pos_l=jnp.array(epl_new),
            ee_pos_r=jnp.array(epr_new),
            ee_wxyz_l=jnp.array(ewl_new),
            ee_wxyz_r=jnp.array(ewr_new),
            joints=jnp.array(jnt_new),
            dt=jnp.array(dt_new),
            mask=jnp.array(msk_new),
        )
        pathbatch_path = os.path.join(self.dirpath, PATHS_FILENAME)
        log.debug(f"⚙️💾 Saving pathbatch to {pathbatch_path}...")
        new_batch.save(pathbatch_path)

        # overwrite pixelpaths.yaml
        pixelpaths_path = os.path.join(self.dirpath, PIXELPATHS_FILENAME)
        log.debug(f"⚙️💾 Saving pixelpaths to {pixelpaths_path}...")
        with open(pixelpaths_path, "w") as f:
            yaml.safe_dump([pp.to_dict() for pp in new_pixelpaths], f)

        # overwrite inkpalette.yaml
        inkpalette_path = os.path.join(self.dirpath, INKPALETTE_FILENAME)
        log.debug(f"⚙️💾 Saving inkpalette to {inkpalette_path}...")
        self.inkpalette.save_yaml(inkpalette_path)

        # update descriptions
        log.debug(f"⚙️💾 Updating descriptions...")
        start_idx = len(self.path_descriptions)
        for k, dip in enumerate(dip_paths_raw, start=start_idx):
            self.path_descriptions[f"path_{k:03d}"] = dip["description"]
        # updates meta.yaml (image unchanged)
        self.save()          
