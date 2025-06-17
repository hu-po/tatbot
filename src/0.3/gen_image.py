from dataclasses import dataclass
import os
import shutil

import cv2
import networkx as nx
import numpy as np
import PIL.Image
import replicate
from skimage.morphology import skeletonize

from _log import get_logger, setup_log_with_config, COLORS
from _plan import Plan

log = get_logger('gen_image')

@dataclass
class ImagePlanConfig:
    debug: bool = False
    """Enable debug logging."""

    output_dir: str = os.path.expanduser("~/tatbot/output/plans/")
    """Directory to save the plan directory into."""

    pad_len: int = 128
    """Number of points to pad the paths to."""


    image_width_px: int = 640
    """ Width of the design image (pixels)."""
    image_height_px: int = 640
    """ Height of the design image (pixels)."""
    image_width_m: float = 0.06
    """ Width of the design image (meters)."""
    image_height_m: float = 0.06
    """ Height of the design image (meters)."""

    # image_path: str | None = None
    image_path: str | None = os.path.expanduser("~/tatbot/assets/designs/cat.png")
    """ (Optional) Local path to the tattoo design image."""
    prompt: str = "cat"
    """ Prompt for the design image generation."""

    num_patches_width: int = 24
    """ Number of patches along the x-axis."""
    num_patches_height: int = 24
    """ Number of patches along the y-axis."""
    patch_empty_threshold: float = 254.0
    """(0-255) Pixel intensity mean threshold to consider a patch empty. Higher is more aggressive."""
    binary_threshold: int = 127
    """(0-255) Pixel intensity threshold for binary conversion of patch. Lower is more aggressive."""
    max_components_per_patch: int = 10
    """Maximum number of components to visualize per patch."""


def plan_from_image(config: ImagePlanConfig):
    if config.image_path:
        name = os.path.splitext(os.path.basename(config.image_path))[0]
    else:
        name = config.prompt.replace(" ", "_")
    
    plan = Plan(
        name=name,
        dirpath=os.path.join(config.output_dir, name),
        path_pad_len=config.pad_len,
        image_width_px=config.image_width_px,
        image_height_px=config.image_height_px,
        image_width_m=config.image_width_m,
        image_height_m=config.image_height_m,
    )

    if config.image_path is None:
        log.info("🖼️ generating design using replicate...")
        raw_image_path = os.path.join(plan_dir, "raw.png")
        image_path = os.path.join(plan_dir, "image.png")
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
        log.info(f"🖼️💾 Saved raw image to {raw_image_path}")

        img_raw = PIL.Image.open(raw_image_path)
        log.info(f"🖼️ Resizing image to {config.image_width_px}x{config.image_height_px}px...")
        img_resized = img_raw.resize((config.image_width_px, config.image_height_px))
        img_resized.save(image_path)
        log.info(f"🖼️💾 Saved resized image to {image_path}")

    else:
        source_image_path = config.image_path
        if not os.path.isabs(source_image_path) and not os.path.exists(source_image_path):
            source_image_path = os.path.join(config.output_dir, config.image_path)
        
        if not os.path.exists(source_image_path):
            log.error(f"Image file not found: {source_image_path}")
            return

        raw_image_path = os.path.join(plan.dirpath, "raw.png")
        image_path = os.path.join(plan.dirpath, "image.png")

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
    img_np = np.array(img_pil.convert("RGB"))
    plan.save(img_np)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_viz = img_bgr.copy()
    comp_viz = img_bgr.copy()

    log.info("Creating patches...")
    patches_dir = os.path.join(plan.dirpath, "patches")
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

                color_vis = list(COLORS.values())[k % len(COLORS)]
                patch_h, patch_w = labels.shape
                comp_viz[patch_start_y : patch_start_y + patch_h, patch_start_x : patch_start_x + patch_w][
                    component_mask
                ] = color_vis

    log.info(f"Saved {full_patches} full patches.")
    log.info(f"Found {empty_patches} empty patches.")

    plan.add_pixel_paths(all_paths)

    viz_path = os.path.join(plan.dirpath, f"patches.png")
    cv2.imwrite(viz_path, img_viz)
    log.info(f"🖼️ Saved patch visualization to {viz_path}")

    comp_viz_path = os.path.join(plan.dirpath, f"components.png")
    cv2.imwrite(comp_viz_path, comp_viz)
    log.info(f"🖼️ Saved component visualization to {comp_viz_path}")

if __name__ == "__main__":
    args = setup_log_with_config(ImagePlanConfig)
    plan_from_image(args)