---
summary: VGGT (Visual Geometry Grounded Tracking) integration and usage
tags: [vision, vggt]
updated: 2025-08-21
audience: [dev]
---

# 🔍 VGGT

VGGT (Visual Geometry Grounded Tracking) is integrated into Tatbot to provide dense 3D reconstruction and automatic camera pose estimation from multi-view RGB images. This complements the existing AprilTag-based calibration and RealSense depth sensing.

## Overview

VGGT enhances the vision system by:
- **Dense 3D Reconstruction**: Generates high-quality point clouds from RGB-only images
- **Markerless Pose Estimation**: Estimates camera poses without requiring AprilTags
- **Cross-Node GPU Processing**: Runs on dedicated GPU nodes (ook) while cameras are on sensor nodes (hog)
- **Scale Alignment**: Automatically aligns VGGT outputs with metric measurements using AprilTag references
- **COLMAP Integration**: Exports standard COLMAP format for compatibility with external tools

## Architecture

### Components

```text
src/tatbot/
├── cam/vggt_runner.py           # Core VGGT processing utilities
├── tools/gpu/vggt_recon.py      # Remote GPU reconstruction tool
├── tools/robot/sense.py         # Enhanced with VGGT integration
└── viz/vggt_compare.py          # Comparison visualization tool
```

### Data Flow

1. **Image Capture** (hog): Multi-camera RGB images saved to NFS
2. **GPU Processing** (ook): VGGT model processes images → poses + dense points
3. **Scale Alignment** (ook): Align VGGT scale with AprilTag reference measurements
4. **File Storage** (NFS): Results saved in COLMAP + PLY formats
5. **Visualization** (any): Compare VGGT vs RealSense reconstructions

## Usage

### Enable VGGT in Sense Tool

```json
{
  "tool": "sense",
  "input": {
    "scene": "default",
    "enable_vggt": true,
    "vggt_conf_threshold": 5.0,
    "vggt_use_ba": false,
    "calibrate_extrinsics": true
  }
}
```

**Parameters:**
- `enable_vggt`: Enable VGGT reconstruction (default: false)
- `vggt_conf_threshold`: Confidence threshold for point filtering (default: 5.0)
- `vggt_use_ba`: Enable bundle adjustment refinement (default: false)
- `vggt_image_count`: Number of images per camera (default: 1)

### Direct Reconstruction

You can also run VGGT reconstruction directly on GPU nodes:

```json
{
  "tool": "vggt_reconstruct",
  "input": {
    "image_dir": "/nfs/tatbot/recordings/sense-default-20241215_143022/images",
    "output_pointcloud_path": "/nfs/tatbot/recordings/sense-default-20241215_143022/pointclouds/vggt_dense.ply",
    "output_frustums_path": "/nfs/tatbot/recordings/sense-default-20241215_143022/metadata/vggt_frustums.json",
    "output_colmap_dir": "/nfs/tatbot/recordings/sense-default-20241215_143022/colmap",
    "scene": "default",
    "vggt_conf_threshold": 5.0,
    "shared_camera": false
  }
}
```

### Compare Viz

View VGGT reconstruction alongside RealSense data:

```bash
uv run python -m tatbot.viz.vggt_compare --dataset_dir=/nfs/tatbot/recordings/sense-default-20241215_143022
```

Or via MCP tool:

```json
{
  "tool": "vggt_compare_viz",
  "input": {
    "dataset_dir": "/nfs/tatbot/recordings/sense-default-20241215_143022",
    "show_vggt": true,
    "show_rs": true,
    "show_vggt_frustums": true,
    "show_apriltag_frustums": true
  }
}
```

## Output

VGGT integration generates the following files in sense datasets:

```text
/nfs/tatbot/recordings/sense-{scene}-{timestamp}/
├── images/                      # RGB images for VGGT processing
│   ├── realsense1.png
│   ├── realsense2.png
│   └── ipcamera1.png
├── pointclouds/
│   ├── realsense1_000000.ply    # RealSense depth point clouds
│   ├── realsense2_000000.ply
│   └── vggt_dense.ply           # VGGT dense reconstruction
├── colmap/                      # COLMAP format camera data
│   ├── cameras.txt              # Camera intrinsics
│   ├── images.txt               # Camera poses
│   └── points3D.txt             # 3D points (optional)
└── metadata/
    ├── apriltag_frustums.json   # AprilTag camera poses
    ├── vggt_frustums.json       # VGGT camera poses
    ├── vggt_confidence.npz      # Depth confidence maps
    └── metrics.json             # Reconstruction metrics
```

## Implementation

### Model Management

VGGT uses the HuggingFace model cache for efficient storage:
- **Model**: `facebook/VGGT-1B` (1 billion parameters)
- **Storage**: `~/.cache/huggingface/` on each GPU node
- **Loading**: Automatic download on first use, cached thereafter
- **Memory**: ~8GB VRAM required for inference

### Scale Alignment

VGGT outputs are scale-ambiguous and require metric alignment:

1. **AprilTag Method** (Preferred): Compares camera baseline distances between VGGT and AprilTag poses
2. **Median Baseline**: Robust scale factor = `median(AprilTag_baselines) / median(VGGT_baselines)`
3. **Validation**: Scale factors outside 1e-3 to 1e3 range are reset to 1.0
4. **Application**: Scale applied only to camera poses (not double-scaled with world points)

### Coordinate Systems

- **VGGT Output**: Camera-from-world extrinsics (OpenCV convention)
- **Tatbot Internal**: World-from-camera poses (for consistency)
- **Conversion**: Handled automatically by `vggt_runner.py` utilities
- **COLMAP Export**: Standard COLMAP format with quaternion poses

### Cross-Node Communication

- **Protocol**: MCP (Model Context Protocol) over HTTP
- **Payload**: File paths only (~1KB), not raw data
- **Timeout**: 15 minutes for GPU processing
- **Retry**: 2 attempts with exponential backoff
- **Fallback**: Graceful degradation if VGGT fails

## Configuration

### Nodes

- **GPU Node** (`ook`): CUDA-capable GPU with ≥8GB VRAM
- **Sensor Node** (`hog`): Camera connections (RealSense + IP cameras)
- **Storage**: NFS mount for cross-node file access

### Cameras

VGGT works with any RGB cameras defined in the scene:
- **IP Cameras**: Full resolution RGB frames
- **RealSense**: Color stream (depth stream used separately)
- **Format**: PNG images saved to `images/` subdirectory
- **Naming**: `{camera_name}.png` format for consistent processing

### GPU

Enable VGGT capability in `src/conf/mcp/ook.yaml`:

```yaml
extras: ["gpu"]  # Enables GPU-required tools
tools:
  - vggt_reconstruct
  - convert_strokelist_to_batch
```

### Settings

Optional VGGT-specific settings in `src/conf/cam/vggt.yaml`:

```yaml
preferred_gpu_node: ook
model_path: null  # Use HuggingFace cache
resolution: 518   # VGGT internal resolution
conf_threshold: 5.0
use_ba: false     # Bundle adjustment
return_point_map: true
```

## Performance

### Timing
- **Image Loading**: ~2-5 seconds for typical multi-camera setup
- **VGGT Inference**: ~30-60 seconds on RTX-class GPU
- **Scale Alignment**: ~1 second
- **File I/O**: ~5-10 seconds for PLY/COLMAP export
- **Total**: ~1-2 minutes for complete reconstruction

### Resources
- **GPU Memory**: ~8GB VRAM during inference
- **Storage**: ~50-100MB per reconstruction (dense point cloud)
- **Network**: Minimal (<1KB MCP payloads)

### Quality
- **Point Density**: 10x+ more points than RealSense depth
- **Pose Accuracy**: Typically within 2-5mm of AprilTag calibration
- **Coverage**: Full 360° reconstruction from multi-view images

## Integration

### Existing Vision Pipeline
- **Complementary**: Works alongside AprilTag calibration and RealSense depth
- **Validation**: VGGT poses compared against AprilTag ground truth
- **Enhancement**: Dense reconstruction supplements sparse RealSense data

### Surface Mapping
- **Mesh Input**: VGGT point clouds can be used for skin mesh reconstruction
- **Stroke Mapping**: Enhanced 3D surfaces improve 2D→3D stroke mapping accuracy
- **Geodesic Tracing**: Denser meshes enable more accurate surface path planning

### Visualization
- **Comparison Tool**: Side-by-side VGGT vs RealSense point cloud analysis
- **Camera Poses**: Visual comparison of AprilTag vs VGGT camera frustums
- **Quality Assessment**: Real-time pose accuracy and point density metrics

## Troubleshooting

### Issues

**VGGT Processing Fails:**
- Check GPU memory availability (`nvidia-smi`)
- Verify HuggingFace model cache download completed
- Ensure sufficient disk space for model (~4GB)

**Scale Alignment Problems:**
- Verify AprilTag detection succeeded in images
- Check that AprilTag and VGGT detect same cameras
- Examine scale factor bounds in metrics.json

**Cross-Node Communication:**
- Verify MCP server running on GPU node
- Check network connectivity between hog and ook
- Ensure NFS mount accessible from both nodes

**Poor Reconstruction Quality:**
- Increase number of viewpoints (more camera positions)
- Improve lighting conditions for better image quality
- Adjust confidence threshold for point filtering
- Consider bundle adjustment for pose refinement

### Debugging

Monitor VGGT processing via logs:
```bash
# On GPU node
tail -f /nfs/tatbot/mcp-logs/ook.log

# Metrics file
cat /nfs/tatbot/recordings/{dataset}/metadata/metrics.json
```

Key metrics to check:
- `vggt_scale_factor`: Should be reasonable (0.1 - 10.0)
- `vggt_point_count`: Higher is generally better
- `mean_cam_center_err_m`: Pose accuracy vs AprilTag reference

## Roadmap

- **Bundle Adjustment**: Integration with pycolmap for pose refinement
- **Multi-Scale Processing**: Support for different VGGT resolution settings
- **Temporal Consistency**: Video-based VGGT for improved accuracy
- **Direct Mesh Output**: Skip PLY intermediate and generate meshes directly
- **Real-Time Processing**: Optimize for faster reconstruction cycles
