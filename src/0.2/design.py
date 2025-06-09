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
    num_patches_width: int = 16
    """ Number of patches along the x-axis."""
    num_patches_height: int = 16
    """ Number of patches along the y-axis."""


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

    log.info("Creating patches...")
    patches_dir = f"{config.output_dir}/{config.image_filename}_patches"
    os.makedirs(patches_dir, exist_ok=True)
    log.info(f"Saving patches to {patches_dir}")

    patch_num = 0
    x_coords = np.linspace(0, original_width, config.num_patches_width + 1, dtype=int)
    y_coords = np.linspace(0, original_height, config.num_patches_height + 1, dtype=int)

    for i in range(config.num_patches_height):
        for j in range(config.num_patches_width):
            box = (x_coords[j], y_coords[i], x_coords[j + 1], y_coords[i + 1])
            patch = img_pil.crop(box)
            patch_path = os.path.join(patches_dir, f"patch_{i:02d}_{j:02d}.png")
            patch.save(patch_path)
            patch_num += 1
    log.info(f"Saved {patch_num} patches.")

    # # TODO: feed each patch to sam-2

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


if __name__ == "__main__":
    args = tyro.cli(DesignConfig)
    main(args)

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