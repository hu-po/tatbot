import os
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

import PIL.Image
import replicate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

@dataclass
class DesignConfig:
    # image_path: str | None = None
    image_path: str | None = os.path.expanduser("~/tatbot/output/design/cat.png")
    """ (Optional) Local path to the tattoo design image."""
    prompt: str = "cat"
    """ Prompt for the design image generation."""
    image_width_px: int = 2048
    """ Width of the design image (pixels)."""
    image_height_px: int = 2048
    """ Height of the design image (pixels)."""
    image_width_m: float = 0.04
    """ Width of the design image (meters)."""
    image_height_m: float = 0.04
    """ Height of the design image (meters)."""

def main(config: DesignConfig):

    if config.image_path is None:
        _prompt = config.prompt.replace(" ", "_")
        config.image_path = os.path.expanduser(f"~/tatbot/output/design/{_prompt}.png")
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
        with open(config.image_path, "wb") as file:
            file.write(output.read())

    img_pil = PIL.Image.open(config.image_path)
    original_width, original_height = img_pil.size
    log.info(f"🖼️ Design image size: {original_width}x{original_height} pixels.")

    log.info("🔍 Segmenting design...")
    # https://replicate.com/meta/sam-2/api/schema
    output = replicate.run(
        "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
        input={
            "image": open(config.image_path, "rb"),
            "points_per_side": 32,
            "mask_threshold": 0.88,
        }
    )
    import io

    mask_files = output['individual_masks']
    log.info(f"🔍 Found {len(mask_files)} masks.")

    individual_masks_np = [np.array(PIL.Image.open(io.BytesIO(f.read()))) for f in mask_files]

    log.info(f"Converted masks to {len(individual_masks_np)} NumPy arrays.")
    if individual_masks_np:
        log.info(f"First mask shape: {individual_masks_np[0].shape}")


if __name__ == "__main__":
    main(DesignConfig())

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