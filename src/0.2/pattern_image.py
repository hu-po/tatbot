from dataclasses import asdict, dataclass
import io
import json
import logging
import os
import shutil

import cv2
import jax.numpy as jnp
import networkx as nx
import numpy as np
import PIL.Image
import replicate
from skimage.morphology import skeletonize
import tyro

from pattern import Path, Pattern, Pose, make_pathviz_image

log = logging.getLogger('tatbot')

@dataclass
class PatternFromImageConfig:
    # image_path: str | None = None
    image_path: str | None = os.path.expanduser("~/tatbot/assets/designs/infinity.png")
    """ (Optional) Local path to the tattoo design image."""
    prompt: str = "infinity"
    """ Prompt for the design image generation."""
    output_dir: str = os.path.expanduser("~/tatbot/output/patterns")
    """ Directory to save the design image and patches."""
    image_width_px: int = 256
    """ Width of the design image (pixels)."""
    image_height_px: int = 256
    """ Height of the design image (pixels)."""
    image_width_m: float = 0.060
    """ Width of the design image (meters)."""
    image_height_m: float = 0.060
    """ Height of the design image (meters)."""
    num_patches_width: int = 16
    """ Number of patches along the x-axis."""
    num_patches_height: int = 16
    """ Number of patches along the y-axis."""
    patch_empty_threshold: float = 254.0
    """(0-255) Pixel intensity mean threshold to consider a patch empty. Higher is more aggressive."""
    binary_threshold: int = 127
    """(0-255) Pixel intensity threshold for binary conversion of patch. Lower is more aggressive."""
    max_components_per_patch: int = 10
    """Maximum number of components to visualize per patch."""


VIZ_COLORS = [
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]


def make_pattern_from_image(config: PatternFromImageConfig):
    log.info(f"🔍 Using output directory: {config.output_dir}")
    os.makedirs(config.output_dir, exist_ok=True)

    if config.image_path:
        design_name = os.path.splitext(os.path.basename(config.image_path))[0]
    else:
        design_name = config.prompt.replace(" ", "_")

    design_output_dir = os.path.join(config.output_dir, design_name)
    log.info(f"🎨 All design outputs will be saved in: {design_output_dir}")
    os.makedirs(design_output_dir, exist_ok=True)

    if config.image_path is None:
        raw_image_path = os.path.join(design_output_dir, "raw.png")
        image_path = os.path.join(design_output_dir, "image.png")
        log.info(" Generating design...")
        # https://replicate.com/black-forest-labs/flux-1.1-pro-ultra/api/schema
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro-ultra",
            input={
                "prompt": f"black tattoo design of {config.prompt}, linework, svg, black on white",
                "aspect_ratio": "1:1",
                "output_format": "png",
                "safety_tolerance": 6,
            }
        )
        with open(raw_image_path, "wb") as file:
            file.write(output.read())
        log.info(f"Saved raw image to {raw_image_path}")

        img_raw = PIL.Image.open(raw_image_path)
        log.info(f"Resizing image to {config.image_width_px}x{config.image_height_px}px...")
        img_resized = img_raw.resize((config.image_width_px, config.image_height_px))
        img_resized.save(image_path)
        log.info(f"Saved resized image to {image_path}")

    else:
        source_image_path = config.image_path
        if not os.path.isabs(source_image_path) and not os.path.exists(source_image_path):
            source_image_path = os.path.join(config.output_dir, config.image_path)
        
        if not os.path.exists(source_image_path):
            log.error(f"Image file not found: {source_image_path}")
            return

        raw_image_path = os.path.join(design_output_dir, f"raw.png")
        image_path = os.path.join(design_output_dir, f"image.png")

        if os.path.abspath(source_image_path) != os.path.abspath(raw_image_path):
            log.info(f"Copying {source_image_path} to {raw_image_path}")
            shutil.copy(source_image_path, raw_image_path)
        
        img_raw = PIL.Image.open(raw_image_path)
        log.info(f"Resizing image to {config.image_width_px}x{config.image_height_px}px...")
        img_resized = img_raw.resize((config.image_width_px, config.image_height_px))
        img_resized.save(image_path)
        log.info(f"Saved resized image to {image_path}")

    img_pil = PIL.Image.open(image_path)
    original_width, original_height = img_pil.size
    log.info(f"🖼️ Design image size: {original_width}x{original_height} pixels.")

    img_viz = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    comp_viz = img_viz.copy()
    path_viz = img_viz.copy()

    log.info("Creating patches...")
    patches_dir = os.path.join(design_output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    log.info(f"Saving patches to {patches_dir}")

    full_patches = 0
    empty_patches = 0
    all_paths = []
    x_coords = np.linspace(0, original_width, config.num_patches_width + 1, dtype=int)
    y_coords = np.linspace(0, original_height, config.num_patches_height + 1, dtype=int)

    for i in range(config.num_patches_height):
        for j in range(config.num_patches_width):
            box = (x_coords[j], y_coords[i], x_coords[j + 1], y_coords[i + 1])
            patch = img_pil.crop(box)

            is_empty = np.array(patch.convert("L")).mean() > config.patch_empty_threshold

            # Draw rectangle and text for visualization
            color = (0, 0, 255) if is_empty else (0, 255, 0)  # Red for empty, Green for non-empty
            cv2.rectangle(img_viz, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.rectangle(comp_viz, (box[0], box[1]), (box[2], box[3]), color, 2)
            center_x = (x_coords[j] + x_coords[j + 1]) // 2
            center_y = (y_coords[i] + y_coords[i + 1]) // 2
            text = f"{i},{j}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            cv2.putText(img_viz, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
            cv2.putText(comp_viz, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

            if is_empty:
                empty_patches += 1
                continue

            patch_path = os.path.join(patches_dir, f"patch_{i:02d}_{j:02d}.png")
            patch.save(patch_path)
            full_patches += 1

            patch_gray = np.array(patch.convert("L"))
            _, binary_patch = cv2.threshold(patch_gray, config.binary_threshold, 255, cv2.THRESH_BINARY_INV)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_patch, connectivity=8)

            if num_labels <= 1:
                continue

            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_component_indices = np.argsort(areas)[::-1] + 1

            for k, label_idx in enumerate(sorted_component_indices[:config.max_components_per_patch]):
                component_mask = labels == label_idx

                skeleton = skeletonize(component_mask)
                if not np.any(skeleton):
                    continue

                G = nx.Graph()
                pixel_coords = np.argwhere(skeleton)
                pixel_coords_set = {tuple(p) for p in pixel_coords}
                for r_c_tuple in pixel_coords_set:
                    G.add_node(r_c_tuple)
                    r, c = r_c_tuple
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in pixel_coords_set:
                                G.add_edge((r, c), (nr, nc))
                
                if G.number_of_nodes() == 0:
                    continue

                subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

                for subgraph in subgraphs:
                    if subgraph.number_of_nodes() == 0:
                        continue
                    
                    if subgraph.number_of_nodes() == 1:
                        path_nodes_local = list(subgraph.nodes())
                    else:
                        endpoints = [node for node, degree in subgraph.degree() if degree == 1]
                        start_node = endpoints[0] if endpoints else list(subgraph.nodes())[0]

                        path_nodes_local = []
                        stack = [start_node]
                        visited_nodes = {start_node}
                        path_nodes_local.append(start_node)
                        while stack:
                            curr_node = stack[-1]
                            unvisited_neighbors = [n for n in subgraph.neighbors(curr_node) if n not in visited_nodes]
                            if unvisited_neighbors:
                                next_node = unvisited_neighbors[0]
                                visited_nodes.add(next_node)
                                stack.append(next_node)
                                path_nodes_local.append(next_node)
                            else:
                                stack.pop()
                                if stack:
                                    path_nodes_local.append(stack[-1])
                        
                        if len(path_nodes_local) > 1 and path_nodes_local[-1] == path_nodes_local[-2]:
                            path_nodes_local.pop()

                    patch_start_y, patch_start_x = box[1], box[0]
                    global_path = [(int(c + patch_start_x), int(r + patch_start_y)) for r, c in path_nodes_local]
                    
                    if not global_path:
                        continue

                    all_paths.append(global_path)

                color_vis = VIZ_COLORS[k % len(VIZ_COLORS)]
                patch_h, patch_w = labels.shape
                comp_viz[patch_start_y : patch_start_y + patch_h, patch_start_x : patch_start_x + patch_w][
                    component_mask
                ] = color_vis

    log.info(f"Saved {full_patches} full patches.")
    log.info(f"Found {empty_patches} empty patches.")

    if all_paths:
        scale_x = config.image_width_m / original_width
        scale_y = config.image_height_m / original_height

        paths = []
        for path_px in all_paths:
            poses = [
                Pose(
                    pixel_coords=jnp.array(p_px, dtype=jnp.int32),
                    pos=jnp.array([p_px[0] * scale_x, p_px[1] * scale_y, 0.0]),
                )
                for p_px in path_px
            ]
            paths.append(Path(poses=poses))

        pattern = Pattern(
            name=design_name,
            paths=paths,
            width_m=config.image_width_m,
            height_m=config.image_height_m,
            width_px=config.image_width_px,
            height_px=config.image_height_px,
            image_np=img_viz,
        )

        path_viz = make_pathviz_image(pattern)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.ndarray, jnp.ndarray)):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        paths_path = os.path.join(design_output_dir, "pattern.json")
        with open(paths_path, "w") as f:
            json_data = [[asdict(pose) for pose in path.poses] for path in pattern.paths]
            json.dump(json_data, f, indent=4, cls=NumpyEncoder)
        log.info(f"💾 Saved {len(pattern.paths)} tool paths to {paths_path}")

        path_lengths_px = [
            sum(np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(path[:-1], path[1:]))
            for path in all_paths
        ]

        all_paths_m = [[pose.pos for pose in path.poses] for path in pattern.paths]
        path_lengths_m = [
            sum(np.linalg.norm(p1 - p2) for p1, p2 in zip(path[:-1], path[1:])) for path in all_paths_m
        ]

        log.info("--- Tool Path Statistics ---")
        log.info(f"Total tool paths generated: {len(pattern.paths)}")
        if path_lengths_px:
            log.info(f"Min path length: {np.min(path_lengths_px):.2f} pixels ({np.min(path_lengths_m):.4f} m)")
            log.info(f"Max path length: {np.max(path_lengths_px):.2f} pixels ({np.max(path_lengths_m):.4f} m)")
            log.info(f"Average path length: {np.mean(path_lengths_px):.2f} pixels ({np.mean(path_lengths_m):.4f} m)")
            log.info(f"Total path length: {np.sum(path_lengths_px):.2f} pixels ({np.sum(path_lengths_m):.4f} m)")
    else:
        log.info("No tool paths were generated.")

    viz_path = os.path.join(design_output_dir, f"patchviz.png")
    cv2.imwrite(viz_path, img_viz)
    log.info(f"🖼️ Saved patch visualization to {viz_path}")

    comp_viz_path = os.path.join(design_output_dir, f"compviz.png")
    cv2.imwrite(comp_viz_path, comp_viz)
    log.info(f"🖼️ Saved component visualization to {comp_viz_path}")

    path_viz_path = os.path.join(design_output_dir, f"pathviz.png")
    cv2.imwrite(path_viz_path, path_viz)
    log.info(f"🖼️ Saved tool path visualization to {path_viz_path}")


if __name__ == "__main__":
    args = tyro.cli(PatternFromImageConfig)
    make_pattern_from_image(args)