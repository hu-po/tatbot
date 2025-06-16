from dataclasses import dataclass, field, replace

import cv2
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float, Int
import json

from _log import COLORS, get_logger

log = get_logger('_path')

@jdc.pytree_dataclass
class Pose:
    pos: Float[Array, "3"] = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    """Position in meters (x, y, z)."""
    wxyz: Float[Array, "4"] = field(default_factory=lambda: jnp.array([1.0, 0.0, 0.0, 0.0]))
    """Orientation as quaternion (w, x, y, z)."""
    pixel_coords: Int[Array, "2"] = field(default_factory=lambda: jnp.array([0, 0]))
    """Pixel coordinates of the pose in image space (width, height), origin is top left."""
    metric_coords: Float[Array, "2"] = field(default_factory=lambda: jnp.array([0.0, 0.0]))
    """Metric (meters) coordinates of the pose in foo space, origin is center of foo (x, y)."""

@jdc.pytree_dataclass
class Path:
    positions: Float[Array, "l 3"] = field(default_factory=lambda: Pose().pos)
    orientations: Float[Array, "l 4"] = field(default_factory=lambda: Pose().wxyz)
    pixel_coords: Int[Array, "l 2"] = field(default_factory=lambda: Pose().pixel_coords)
    metric_coords: Float[Array, "l 2"] = field(default_factory=lambda: Pose().metric_coords)

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx) -> Pose:
        return Pose(
            pos=self.positions[idx],
            wxyz=self.orientations[idx],
            pixel_coords=self.pixel_coords[idx],
            metric_coords=self.metric_coords[idx],
        )

@dataclass
class Pattern:
    paths: list[Path] = field(default_factory=list)
    """Ordered list of paths defining a pattern."""
    name: str = "pattern"
    """Name of the pattern."""
    width_m: float = 0.04
    """Width of the pattern in meters."""
    height_m: float = 0.04
    """Height of the pattern in meters."""
    width_px: int = 256
    """Width of the pattern in pixels."""
    height_px: int = 256
    """Height of the pattern in pixels."""
    image_np: np.ndarray | None = field(default=None, repr=False, compare=False)
    """Optional image for visualization."""

    @classmethod
    def from_json(cls, data: dict) -> "Pattern":
        paths = []
        for path_data in data.get("paths", []):
            poses_data = path_data.get("poses", [])
            if not poses_data:
                continue
            paths.append(
                Path(
                    positions=jnp.array([p.get("pos", [0, 0, 0]) for p in poses_data]),
                    orientations=jnp.array([p.get("wxyz", [1, 0, 0, 0]) for p in poses_data]),
                    pixel_coords=jnp.array([p.get("pixel_coords", [0, 0]) for p in poses_data]),
                    metric_coords=jnp.array([p.get("metric_coords", [0.0, 0.0]) for p in poses_data]),
                )
            )
        return cls(
            paths=paths,
            name=data.get("name", cls.name),
            width_m=data.get("width_m", cls.width_m),
            height_m=data.get("height_m", cls.height_m),
            width_px=data.get("width_px", cls.width_px),
            height_px=data.get("height_px", cls.height_px),
        )

    def to_json(self) -> dict:
        """Serializes the Pattern to a JSON-serializable dict."""
        json_data = {
            "name": self.name,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "paths": [],
        }
        for path in self.paths:
            positions = np.asarray(path.positions)
            orientations = np.asarray(path.orientations)
            pixel_coords = np.asarray(path.pixel_coords)
            metric_coords = np.asarray(path.metric_coords)
            poses = [
                {
                    "pos": positions[i].tolist(),
                    "wxyz": orientations[i].tolist(),
                    "pixel_coords": pixel_coords[i].tolist(),
                    "metric_coords": metric_coords[i].tolist(),
                }
                for i in range(len(path))
            ]
            json_data["paths"].append({"poses": poses})
        return json_data

# TODO: make this fast with JAX
def offset_path(path: Path, offset: Float[Array, "3"]) -> Path:
    """Offsets all poses in a path by a given vector."""
    return replace(path, positions=path.positions + offset)

# TODO: make this fast with JAX
def resample_path(path: Path, num_points: int):
    # resample path to num_points length
    pass

def resample_pattern(pattern: Pattern, num_points: int):
    # resample each path in pattern to num_points length
    pass

def add_entry_exit_hover(path: Path, offset: Float[Array, "3"]) -> Path:
    """
    Returns a new Path with an entry pose prepended and an exit pose appended.
    The entry pose is the first pose offset by `offset`, and the exit pose is the last pose offset by `offset`.
    All fields (positions, orientations, pixel_coords, metric_coords) are handled.
    """
    # Entry pose: first pose with position offset
    entry_pos = path.positions[0] + offset
    exit_pos = path.positions[-1] + offset
    # For orientations, just copy first/last
    entry_ori = path.orientations[0]
    exit_ori = path.orientations[-1]
    # For pixel_coords and metric_coords, just copy first/last
    entry_pix = path.pixel_coords[0]
    exit_pix = path.pixel_coords[-1]
    entry_metric = path.metric_coords[0]
    exit_metric = path.metric_coords[-1]

    new_positions = jnp.concatenate([
        entry_pos[None],
        path.positions,
        exit_pos[None],
    ], axis=0)
    new_orientations = jnp.concatenate([
        entry_ori[None],
        path.orientations,
        exit_ori[None],
    ], axis=0)
    new_pixel_coords = jnp.concatenate([
        entry_pix[None],
        path.pixel_coords,
        exit_pix[None],
    ], axis=0)
    new_metric_coords = jnp.concatenate([
        entry_metric[None],
        path.metric_coords,
        exit_metric[None],
    ], axis=0)

    return replace(
        path,
        positions=new_positions,
        orientations=new_orientations,
        pixel_coords=new_pixel_coords,
        metric_coords=new_metric_coords,
    )

def make_pathviz_image(pattern: Pattern) -> np.ndarray:
    """Creates an image with overlayed paths from a pattern.

    The path is visualized with a color gradient indicating the order of poses (time).

    Args:
        pattern: The pattern containing the paths and optional image to visualize.

    Returns:
        The visualization image as a numpy array (BGR).
    """
    if pattern.image_np is None:
        path_viz_np = np.full((pattern.height_px, pattern.width_px, 3), 255, dtype=np.uint8)
    else:
        path_viz_np = pattern.image_np.copy()

    for path in pattern.paths:
        # Convert JAX arrays to list of numpy arrays for processing
        if path.pixel_coords is None:
            continue
        pixel_coords = [np.array(pc) for pc in path.pixel_coords]

        if len(pixel_coords) < 2:
            continue

        path_indices = np.linspace(0, 255, len(pixel_coords), dtype=np.uint8)
        colormap = cv2.applyColorMap(path_indices.reshape(-1, 1), cv2.COLORMAP_JET)

        for path_idx in range(len(pixel_coords) - 1):
            p1 = tuple(pixel_coords[path_idx].astype(int))
            p2 = tuple(pixel_coords[path_idx + 1].astype(int))
            color = colormap[path_idx][0].tolist()
            cv2.line(path_viz_np, p1, p2, color, 2)

    return path_viz_np

def get_path_length_stats(pattern: Pattern) -> dict:
    """Compute path length statistics for a pattern in both pixels and meters."""
    path_lengths_px = [
        sum(np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(np.asarray(path.pixel_coords)[:-1], np.asarray(path.pixel_coords)[1:]))
        if len(path.pixel_coords) > 1 else 0.0
        for path in pattern.paths
    ]
    path_lengths_m = [
        float(jnp.sum(jnp.linalg.norm(jnp.diff(path.positions, axis=0), axis=1)))
        if len(path.positions) > 1 else 0.0
        for path in pattern.paths
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
    return stats

def make_pathlen_image(pattern: Pattern, n_bins: int = 24) -> np.ndarray:
    """Creates a three-part image: pattern name, stats, histogram, and pathviz colored by path length, with a white border and white background for the design unless an image is provided.
    Args:
        pattern: The pattern to visualize.
        n_bins: Number of bins for the histogram
    Returns:
        Combined image as a numpy array.
    """
    border_px = 16  # set border back to 16px
    # --- Compute stats ---
    stats = get_path_length_stats(pattern)
    stats_lines = [
        f"Total paths in pattern: {stats['count']}",
        f"Min path length: {stats['min_px']:.2f} px ({stats['min_m']:.4f} m)",
        f"Max path length: {stats['max_px']:.2f} px ({stats['max_m']:.4f} m)",
        f"Average path length: {stats['mean_px']:.2f} px ({stats['mean_m']:.4f} m)",
        f"Total path length: {stats['sum_px']:.2f} px ({stats['sum_m']:.4f} m)",
    ]

    # --- Render pattern name and stats as image ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    name_font_scale = 0.6
    name_font_thickness = 2
    stats_font_scale = 0.32  # smaller font
    stats_font_thickness = 1
    # Calculate heights
    (name_width, name_height), _ = cv2.getTextSize(pattern.name, font, name_font_scale, name_font_thickness)
    stats_height = 15 * len(stats_lines) + 8  # smaller font, less vertical space
    name_stats_gap = 20  # more space between name and stats
    total_stats_height = name_height + name_stats_gap + stats_height
    stats_img = np.full((total_stats_height, pattern.width_px, 3), 255, dtype=np.uint8)
    # Draw pattern name centered at the top
    name_x = (pattern.width_px - name_width) // 2
    name_y = name_height + 2
    cv2.putText(stats_img, pattern.name, (name_x, name_y), font, name_font_scale, (0,0,0), name_font_thickness, lineType=cv2.LINE_AA)
    # Draw stats below
    y = name_y + name_stats_gap
    for line in stats_lines:
        cv2.putText(stats_img, line, (10, y), font, stats_font_scale, (0,0,0), stats_font_thickness, lineType=cv2.LINE_AA)
        y += 15

    # --- Calculate path lengths in pixels for histogram and coloring ---
    path_lengths = []
    for path in pattern.paths:
        pixel_coords = np.asarray(path.pixel_coords)
        if len(pixel_coords) < 2:
            path_lengths.append(0.0)
            continue
        diffs = np.diff(pixel_coords, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        path_lengths.append(np.sum(seg_lengths))
    path_lengths = np.array(path_lengths)

    # Normalize lengths for color mapping
    if len(path_lengths) > 0 and np.max(path_lengths) > 0:
        norm_lengths = (path_lengths - np.min(path_lengths)) / (np.ptp(path_lengths) if np.ptp(path_lengths) > 0 else 1)
    else:
        norm_lengths = np.zeros_like(path_lengths)

    # --- Top: Histogram ---
    hist_height = 64
    colorbar_height = 12  # thicker colorbar
    label_height = 8  # reduce label height
    indicator_height = 32  # reduce indicator height
    colorbar_margin = 12  # vertical space between bars and colorbar, and between colorbar and numbers
    hist_width = pattern.width_px
    # Add extra space for indicators
    total_hist_height = hist_height + colorbar_margin + colorbar_height + colorbar_margin + indicator_height + label_height
    hist_img = np.full((total_hist_height, hist_width, 3), 255, dtype=np.uint8)
    if len(path_lengths) > 0:
        # Use n_bins even if there are fewer paths
        hist, bin_edges = np.histogram(path_lengths, bins=n_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        max_hist = np.max(hist) if np.max(hist) > 0 else 1
        bar_width = max(1, hist_width // n_bins)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw histogram bars, bottom-aligned to the top of the colorbar
        colorbar_top = hist_height + colorbar_margin
        for i in range(n_bins):
            color_val = int(255 * (bin_centers[i] - np.min(path_lengths)) / (np.ptp(path_lengths) if np.ptp(path_lengths) > 0 else 1))
            color = tuple(int(c) for c in cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0])
            x0 = i * bar_width
            x1 = (i + 1) * bar_width
            y1 = colorbar_top - 1  # bottom of the bar is just above the colorbar
            bar_height = int(hist[i] / max_hist * (hist_height - 10))
            y0 = y1 - bar_height if bar_height > 0 else y1
            cv2.rectangle(hist_img, (x0, y0), (x1 - 2, y1), color, -1)
        # Draw thicker colorbar below the bars
        colorbar_bottom = colorbar_top + colorbar_height
        for x in range(hist_width):
            color_val = int(255 * x / (hist_width - 1))
            color = tuple(int(c) for c in cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0])
            hist_img[colorbar_top:colorbar_bottom, x] = color
        # Draw 4 evenly spaced indicators under the colorbar, not at the edges
        min_val = np.min(path_lengths)
        max_val = np.max(path_lengths)
        vals = [min_val, min_val + (max_val - min_val) / 3, min_val + 2 * (max_val - min_val) / 3, max_val]
        margin = 24
        positions = [margin, hist_width // 3, 2 * hist_width // 3, hist_width - 1 - margin]
        indicator_y = colorbar_bottom + 2  # just below colorbar, no extra whitespace
        indicator_font_scale = 0.32
        indicator_font_thickness = 1
        for v, x in zip(vals, positions):
            text = f"{int(round(v))}"
            (text_width, text_height), _ = cv2.getTextSize(text, font, indicator_font_scale, indicator_font_thickness)
            text_x = x - text_width // 2
            text_y = indicator_y + text_height
            cv2.putText(hist_img, text, (text_x, text_y), font, indicator_font_scale, (0,0,0), indicator_font_thickness)
        # Draw axis label directly below the numbers, add whitespace
        axis_label = "path length"
        font_scale_label = 0.38
        (label_width, label_height_text), _ = cv2.getTextSize(axis_label, font, font_scale_label, stats_font_thickness)
        label_x = (hist_width - label_width) // 2
        whitespace_between_numbers_and_label = 6  # pixels
        label_y = indicator_y + text_height + whitespace_between_numbers_and_label + label_height_text - 6  # add whitespace
        cv2.putText(hist_img, axis_label, (label_x, label_y), font, font_scale_label, (0,0,0), stats_font_thickness)

    # --- Bottom: Pathviz colored by length, inside a white square ---
    # Create a white square for the design
    path_viz_np = np.full((pattern.height_px, pattern.width_px, 3), 255, dtype=np.uint8)
    if pattern.image_np is not None:
        # If a background image is provided, draw it inside the white square
        img = pattern.image_np.copy()
        if img.shape[:2] != (pattern.height_px, pattern.width_px):
            img = cv2.resize(img, (pattern.width_px, pattern.height_px))
        # Blend or overlay as needed (here, just copy)
        path_viz_np = img
    # Draw the colored paths
    for idx, path in enumerate(pattern.paths):
        pixel_coords = np.asarray(path.pixel_coords)
        if len(pixel_coords) < 2:
            continue
        color_val = int(255 * norm_lengths[idx])
        color = tuple(int(c) for c in cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0])
        for path_idx in range(len(pixel_coords) - 1):
            p1 = tuple(pixel_coords[path_idx].astype(int))
            p2 = tuple(pixel_coords[path_idx + 1].astype(int))
            cv2.line(path_viz_np, p1, p2, color, 2)

    # --- Combine images vertically with separation lines and whitespace ---
    sep_thick = 2
    sep_color = (0,0,0)
    white_space = 10
    # Add white space and black line between stats and histogram
    stats_pad = np.full((white_space, pattern.width_px, 3), 255, dtype=np.uint8)
    stats_sep = np.full((sep_thick, pattern.width_px, 3), 0, dtype=np.uint8)
    # Add white space and black line between histogram and image
    hist_pad = np.full((white_space, pattern.width_px, 3), 255, dtype=np.uint8)
    hist_sep = np.full((sep_thick, pattern.width_px, 3), 0, dtype=np.uint8)
    content_img = np.vstack([
        stats_img,
        stats_pad,
        stats_sep,
        hist_img,
        # Remove hist_pad here to reduce whitespace
        hist_sep,
        path_viz_np
    ])
    # Add a border around the whole image (using numpy slicing, not cv2.copyMakeBorder)
    border_color = (255, 255, 255)
    bordered_img = np.full(
        (content_img.shape[0] + 2 * border_px, content_img.shape[1] + 2 * border_px, 3),
        border_color,
        dtype=np.uint8
    )
    bordered_img[border_px:border_px + content_img.shape[0], border_px:border_px + content_img.shape[1]] = content_img
    return bordered_img
