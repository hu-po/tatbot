import io
import os
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import PIL.Image
import replicate
import tyro

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

@dataclass
class DesignConfig:
    # image_path: str | None = None
    image_filename: str | None = "cat.png"
    """ (Optional) Local path to the tattoo design image."""
    prompt: str = "cat"
    """ Prompt for the design image generation."""
    output_dir: str = os.path.expanduser("~/tatbot/output/design")
    """ Directory to save the design image and patches."""
    image_width_px: int = 2048
    """ Width of the design image (pixels)."""
    image_height_px: int = 2048
    """ Height of the design image (pixels)."""
    image_width_m: float = 0.06
    """ Width of the design image (meters)."""
    image_height_m: float = 0.06
    """ Height of the design image (meters)."""
    num_patches_width: int = 32
    """ Number of patches along the x-axis."""
    num_patches_height: int = 32
    """ Number of patches along the y-axis."""
    patch_empty_threshold: float = 250
    """(0-255) Pixel intensity mean threshold to consider a patch empty. Higher is more aggressive."""
    binary_threshold: int = 127
    """(0-255) Pixel intensity threshold for binary conversion of patch. Lower is more aggressive."""
    max_components_per_patch: int = 5
    """Maximum number of components to visualize per patch."""


def main(config: DesignConfig):
    log.info(f"🔍 Using output directory: {config.output_dir}")
    os.makedirs(config.output_dir, exist_ok=True)

    if config.image_filename is None:
        image_path: str = f"{config.output_dir}/{config.prompt.replace(' ', '_')}.png"
        log.info(" Generating design...")
        # https://replicate.com/black-forest-labs/flux-1.1-pro-ultra/api/schema
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro-ultra",
            input={
                "prompt": f"black tattoo design of {config.prompt}, linework, svg, black on white",
                "aspect_ratio": "1:1",
                "output_format": "png",
                "width": config.image_width_px,
                "height": config.image_height_px,
                "safety_tolerance": 6,
            }
        )
        with open(image_path, "wb") as file:
            file.write(output.read())
    else:
        image_path = f"{config.output_dir}/{config.image_filename}"

    img_pil = PIL.Image.open(image_path)
    original_width, original_height = img_pil.size
    log.info(f"🖼️ Design image size: {original_width}x{original_height} pixels.")

    img_viz = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    comp_viz = img_viz.copy()

    log.info("Creating patches...")
    patches_dir = f"{config.output_dir}/{os.path.splitext(config.image_filename)[0]}_patches"
    os.makedirs(patches_dir, exist_ok=True)
    log.info(f"Saving patches to {patches_dir}")

    full_patches = 0
    empty_patches = 0
    x_coords = np.linspace(0, original_width, config.num_patches_width + 1, dtype=int)
    y_coords = np.linspace(0, original_height, config.num_patches_height + 1, dtype=int)

    component_colors_bgr = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]

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
                color_vis = component_colors_bgr[k % len(component_colors_bgr)]
                patch_h, patch_w = labels.shape
                patch_start_y, patch_start_x = box[1], box[0]
                comp_viz[patch_start_y : patch_start_y + patch_h, patch_start_x : patch_start_x + patch_w][
                    component_mask
                ] = color_vis

    log.info(f"Saved {full_patches} full patches.")
    log.info(f"Found {empty_patches} empty patches.")

    base, ext = os.path.splitext(image_path)
    viz_path = f"{base}_patchviz{ext}"
    cv2.imwrite(viz_path, img_viz)
    log.info(f"🖼️ Saved patch visualization to {viz_path}")

    comp_viz_path = f"{base}_compviz{ext}"
    cv2.imwrite(comp_viz_path, comp_viz)
    log.info(f"🖼️ Saved component visualization to {comp_viz_path}")


if __name__ == "__main__":
    args = tyro.cli(DesignConfig)
    main(args)

# log.info("🔍 Segmenting design...")
# # https://replicate.com/meta/sam-2/api/schema
# output = replicate.run(
#     "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
#     input={
#         "image": open(config.image_path, "rb"),
#         "points_per_side": 32,
#         "mask_threshold": 0.88,
#     }
# )

# mask_files = output['individual_masks']
# log.info(f"🔍 Found {len(mask_files)} masks.")

# individual_masks_np = [np.array(PIL.Image.open(io.BytesIO(f.read()))) for f in mask_files]

# log.info(f"Converted masks to {len(individual_masks_np)} NumPy arrays.")
# if individual_masks_np:
#     log.info(f"First mask shape: {individual_masks_np[0].shape}")

# log.info("🖼️ Loading design...")
# img_pil = PIL.Image.open(config.image_path)
# original_width, original_height = img_pil.size
# if original_width > config.image_width_px or original_height > config.image_height_px:
#     log.info(f"🖼️ Resizing from {original_width}x{original_height} to {config.image_width_px}x{config.image_height_px}...")
#     img_pil = img_pil.resize((config.image_width_px, config.image_height_px), PIL.Image.LANCZOS)
# img_pil = img_pil.convert("L")
# img_np = np.array(img_pil)
# img_width_px, img_height_px = img_pil.size
# thresholded_pixels = img_np <= config.image_threshold
# pixel_to_meter_x = config.image_width_m / img_width_px
# pixel_to_meter_y = config.image_height_m / img_height_px

# log.info("🔢 Creating pixel batches...")
# num_targets: int = img_height_px * img_width_px
# design_pointcloud_positions: np.ndarray = np.zeros((num_targets, 3), dtype=np.float32)
# design_pointcloud_colors: np.ndarray = np.zeros((num_targets, 3), dtype=np.uint8)
# batch_radius_px_x = int(config.batch_radius_m / pixel_to_meter_x)
# batch_radius_px_y = int(config.batch_radius_m / pixel_to_meter_y)
# # TODO: some way of ensuring a consistent batch size
# batches: List[PixelBatch] = []
# for center_y in range(batch_radius_px_y, img_height_px, batch_radius_px_y * 2):
#     for center_x in range(batch_radius_px_x, img_width_px, batch_radius_px_x * 2):
#         batch_pixels: List[PixelTarget] = []
#         for y in range(max(0, center_y - batch_radius_px_y), min(img_height_px, center_y + batch_radius_px_y)):
#             for x in range(max(0, center_x - batch_radius_px_x), min(img_width_px, center_x + batch_radius_px_x)):
#                 if thresholded_pixels[y, x]:
#                     meter_x = (x - img_width_px/2) * pixel_to_meter_x
#                     meter_y = (y - img_height_px/2) * pixel_to_meter_y
#                     point_index = y * img_width_px + x
#                     pixel_target = PixelTarget(
#                         pose=Pose( # in design frame
#                             pos=jnp.array([meter_x, meter_y, 0.0]),
#                             # TODO: use normal vector of skin mesh?
#                             wxyz=config.design_pose.wxyz
#                         ),
#                         pixel_index=(x, y),
#                         point_index=point_index
#                     )
#                     batch_pixels.append(pixel_target)
#                     design_pointcloud_positions[point_index] = jnp.array([meter_x, meter_y, 0.0])
#                     design_pointcloud_colors[point_index] = np.array(config.point_color, dtype=np.uint8)
#         # Only create batch if it contains targets
#         if batch_pixels:
#             batch = PixelBatch(
#                 center_pose=Pose(
#                     pos=jnp.array([(center_x - img_width_px/2) * pixel_to_meter_x, (center_y - img_height_px/2) * pixel_to_meter_y, 0.0]),
#                     # TODO: use normal vector of skin mesh?
#                     wxyz=config.design_pose.wxyz
#                 ),
#                 radius_m=config.batch_radius_m,
#                 targets=batch_pixels
#             )
#             batches.append(batch)

# num_batches: int = len(batches)
# log.info(f"🔢 Design has {num_targets} targets in {num_batches} batches.")