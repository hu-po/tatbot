VGGT Integration Plan for Tatbot

Summary
- Goal: Evaluate and plan how to leverage VGGT for camera pose/intrinsic estimation and depth/point cloud generation to replace or augment parts of `tatbot`’s sensing and mapping stack.
- Candidates: `src/tatbot/tools/robot/sense.py`, `src/tatbot/cam/extrinsics.py`, `src/tatbot/cam/depth.py`, `src/tatbot/viz/map.py`, plus related validation and data plumbing.
- Outcome: A staged plan to prototype, validate against URDF and current sensors, and, if satisfactory, integrate as an optional or alternative path.

Refinement From Requirements (Second Prompt)
- Keep `sense.py` behavior, add VGGT as an additional path:
  - Capture as today; additionally run VGGT to estimate extrinsics/intrinsics and dense depth; save alongside LeRobot dataset artifacts.
- Persist images and point clouds on NFS in LeRobot dataset layout:
  - Continue using `LeRobotDataset` under `/nfs/tatbot/recordings/<dataset>`; store VGGT outputs in a `vggt/` subdir and register paths in dataset metadata if practical.
- Remote GPU execution:
  - Cameras are attached to `hog`, VGGT runs only on `ook`. Introduce a GPU MCP tool (mirroring `tools/gpu/convert_strokes.py`) that runs on `ook`, reads images from NFS, and writes VGGT outputs back to NFS.
- New visualization tool:
  - Implement a viz to compare VGGT dense reconstruction vs RealSense PLYs and to overlay camera frustums for both AprilTag- and VGGT-derived poses. Use `BaseVizConfig` like existing viz tools.
- Save camera models in COLMAP format under `config/`:
  - After scans, persist extrinsics/intrinsics as COLMAP models in a new `config/colmap/<scene>/` folder. Update these during sense runs similar to URDF updates.

Additional Refinement (Third Prompt)
- Cross-checking a competing plan highlighted useful details to incorporate:
  - Define an explicit NFS dataset layout (images, pointclouds, colmap, metadata) for predictable downstream tooling.
  - Add concrete success metrics (time, pose accuracy, density) and a phased timeline to drive execution.
  - Strengthen remote GPU operations with queueing, retries, preflight model checks, and robust fallbacks.
  - Ensure the GPU tool runs on `ook` by default, and that COLMAP text files mirror standard names (`cameras.txt`, `images.txt`, `points3D.txt`).
  - Provide viz affordances for confidence-based filtering of VGGT points and side-by-side source toggles.

Current Functionality (Tatbot)
- `tools/robot/sense.py`:
  - Captures RGB frames from IP + RealSense cameras via `lerobot`.
  - Calibrates extrinsics using AprilTags (`cam/extrinsics.get_extrinsics` + `cam/tracker.TagTracker`).
  - Compares calibrated extrinsics with URDF (`cam/validation.compare_extrinsics_with_urdf`).
  - Generates 3D point clouds from RealSense depth via `cam/depth.DepthCamera` and saves `.ply`.
- `cam/extrinsics.py`:
  - Estimates multi-camera extrinsics by detecting AprilTags and averaging T_world_cam via iterative alignment; anchors to a reference camera.
- `cam/depth.py`:
  - Streams RealSense depth + color, applies decimation and clipping, unprojects to world using known camera pose, saves PLY.
- `viz/map.py`:
  - Builds a skin mesh from captured PLY point clouds; maps planned strokes onto the mesh for visualization and later execution.

VGGT Capabilities (from demos)
- Single forward pass over an image set produces:
  - Camera extrinsics `(S, 3, 4)` and intrinsics `(S, 3, 3)` (`pose_encoding_to_extri_intri`).
  - Depth map `(S, H, W, 1)` and confidence `(S, H, W)`; can unproject to world points (`unproject_depth_map_to_point_map`).
  - Optional track prediction + COLMAP export path with BA to refine camera/structure.
- Visualization: point cloud + camera frusta in `viser`.
- Notes: The provided demos expect an image folder sequence, run on GPU, and download model weights at runtime.

Where VGGT Could Replace or Augment Tatbot
- Extrinsics calibration (`cam/extrinsics.py`):
  - Replace AprilTag-based multi-camera registration with VGGT camera pose estimation from multi-view RGB sequences.
  - Pros: No markers; could work with any RGB cameras. Cons: Potential scale ambiguity and accuracy drift without BA or metric cues.
- Depth/point cloud (`cam/depth.py`):
  - Replace RealSense depth capture with depth inferred by VGGT and unprojected to world points.
  - Pros: No depth hardware required; can build environment structure from existing IP cams. Cons: Typically offline, static-scene assumption, lower metric reliability vs active depth.
- Sense workflow (`tools/robot/sense.py`):
  - Add a VGGT “mode” to capture a burst of RGB images, estimate intrinsics/extrinsics, produce depth/point clouds, and save PLY; optionally skip RealSense.
  - Keep AprilTag flow as baseline fallback for scale anchoring and validation.
- Skin mesh and mapping (`viz/map.py`):
  - Allow building the skin mesh from VGGT-generated point clouds (in addition to RealSense PLYs). Stroke mapping logic remains unchanged once a mesh exists.

Key Integration Design
- New module: `tatbot/cam/vggt_runner.py`
  - API: `run_vggt(images: list[str] | np.ndarray, resolution=518, use_ba: bool=False, return_point_map: bool=False) -> VGGTResult` where `VGGTResult` contains:
    - `images` (S, 3, H, W), `extrinsic` (S, 3, 4), `intrinsic` (S, 3, 3), `depth` (S, H, W, 1), `depth_conf` (S, H, W), optionally `world_points` (S, H, W, 3).
  - Utilities to:
    - Convert extrinsics/intrinsics to `tatbot.data.cams.Cams` instances.
    - Write `.ply` from world points with RGB colors (per `viz_demo` flow).
    - Optionally run COLMAP export and BA (adapting `vggt_demo_colmap.py`).
    - Device selection, dtype autocast, weight loading from a configured path on NFS (no runtime network).
    - Emit camera frustum payloads (poses + intrinsics) to JSON for viz consumption.
- `tools/robot/sense.py` additions:
  - New optional parameters `use_vggt: bool`, `vggt_image_count: int`, `vggt_use_ba: bool`, `vggt_conf_thresh: float`.
  - Flow (VGGT mode):
    1) Move robot to ready pose; capture synchronized RGB frames from all configured cameras (IP + RealSense color) to a folder.
    2) Invoke a remote GPU tool on `ook` (see below) that runs `vggt_runner.run_vggt` against the captured frames on NFS.
    3) Convert poses to `Cams`; compare with URDF. Optionally compute scale alignment (see “Scale” below).
    4) Generate `.ply` point clouds from depth+conf with thresholding. Save outputs into the dataset folder (parallel to current RealSense PLYs) under `vggt/`.
    5) Save camera frustums for both AprilTag and VGGT solutions to JSON/NPZ for later viz; update COLMAP configs under `config/colmap/<scene>/`.
    6) Report metrics (pose deviation, depth coverage) in the tool output.
  - Preserve existing AprilTag calibration and RealSense capture as defaults; VGGT is opt-in.
- Remote GPU tool: `tatbot/tools/gpu/vggt_recon.py`
  - Decorate with `@tool(..., nodes=["ook"], requires=["gpu"])` and verify GPU in node config as in `convert_strokes.py`.
  - Inputs: NFS image folder path, output folder paths in current dataset, scene/meta, flags for BA/point-map/depth conf threshold.
  - Outputs: success + message; PLYs saved to NFS; JSON/NPZ with camera frustums; optional COLMAP model.
  - The robot-side `sense` tool running on `hog` calls this GPU tool via MCP; both read/write via `/nfs/tatbot`.
- `viz/map.py` enhancements:
  - Add a GUI toggle to load PLYs from a “VGGT” subfolder.
  - Optional: add a small camera-frusta viewer for VGGT poses in map viz (leveraging existing `viser` server).
- New viz: `tatbot/viz/vggt_compare.py` + `tools/viz` entry
  - Base on `BaseViz`/`BaseVizConfig`.
  - Load RealSense PLYs and VGGT PLYs from the same dataset directory; expose toggles to show/hide each source.
  - Load and render camera frustums for AprilTag extrinsics (saved during sense) and VGGT extrinsics; color-code sources.
  - Provide summary stats (pose deviations, point counts) in GUI.
- Configuration & dependencies:
  - Hydra: new config group `cam/vggt.yaml` to declare model path, device, use_ba, and thresholds.
  - Extras: ensure `torch`, `vggt` (or repo path), `pycolmap` (optional), and `onnxruntime` (optional sky masking) in `.[img,viz,gpu]` as needed.
  - Model weights path should be configurable and pre-cached on `/nfs/tatbot/models/vggt/model.pt` to avoid runtime downloads on `ook`.
  - Candidate versions: `torch>=2.0`, `torchvision>=0.15`, `pycolmap>=0.4`, `trimesh>=3.15`. Pin within project constraints.

NFS Dataset Layout (LeRobot-compatible)
- All artifacts live under `/nfs/tatbot/recordings/sense-{scene}-{timestamp}/` owned by the `LeRobotDataset` root:
  - `images/`: captured PNGs (`<camera>.png`) plus any annotated images.
  - `pointclouds/`: RealSense PLYs (`<cam>_######.ply`) and VGGT PLYs (`vggt_dense_*.ply`).
  - `colmap/`: `cameras.txt`, `images.txt`, `points3D.txt` (VGGT outputs; may be absent if BA disabled).
  - `metadata/`: `scene.yaml`, `apriltag_poses.yaml`, `vggt_frustums.json`, `vggt_confidence.npz` (depth_conf, thresholds), and `metrics.json`.

Coordinate Frames and Scale Alignment
- Frame conventions:
  - VGGT extrinsics returned follow OpenCV convention (camera-from-world). Convert to Tatbot’s world-from-camera as needed, consistent with `jaxlie.SE3` usage.
  - Recentering: demos subtract scene mean before visualization; do not apply for production integration—retain absolute frames.
- Scale:
  - Many learned multi-view depth/pose pipelines are up-to-scale without metric anchors. Plan to anchor scale by one of:
    - AprilTag board with known tag size (already in scene config) to estimate a global scale factor vs URDF distances.
    - A single RealSense capture per session to compute a scale factor between VGGT and metric PLYs.
    - Known baseline measurement in the scene (checkerboard or calibrated object).
  - Apply the scale factor to VGGT translations and/or depths prior to saving PLYs and writing `Cams`.

Risks, Gaps, and Mitigations
- Metric accuracy: Depth/pose may be up-to-scale or biased; mitigate with scale anchoring + BA on tracks.
- Runtime/perf: GPU required for timely inference; large sequences may exceed memory. Mitigate via reduced image counts, resolution 518, or batching.
- Static-scene assumption: Motion corrupts reconstruction. Mitigate by pausing robot and environment during capture.
- Dependency friction: `pycolmap`, CUDA, and weight downloads are non-trivial. Mitigate by pinning versions, pre-caching on NFS, and providing CPU fallback for non-BA mode.
- Domain shift: Skin/robot scenes may have low texture. Mitigate with more viewpoints, varied angles, and optional sky/background masking where applicable.
- Licensing/redistribution: Confirm VGGT license compatibility and weight hosting policy for internal mirrors.

Validation and Test Plan
- Data collection:
  - Capture synchronized multi-camera sequences from typical Tatbot scenes (left/right arms, workspace) at ready pose.
  - Record ground-truth proxies: AprilTag detections, URDF link poses, RealSense PLYs.
- Metrics:
  - Pose deviation vs URDF: per-camera translation/rotation error; target <1–2 cm and <1.5° (minimum accept <3 cm/<3°).
  - Relative pose consistency across cameras: average pairwise baseline error vs URDF.
  - Depth quality: density after confidence thresholding; % points within workspace zone; overlap vs RealSense PLYs (Chamfer distance). Confidence threshold sweep.
  - Mapping success: ability to build a mesh and map strokes without major artifacts.
  - Runtime: end-to-end sense + VGGT < 5 minutes on `ook` (non-BA path); memory usage within GPU limits.
- Tests:
  - Unit: SE3 conversions, intrinsics/extrinsics mapping into `Cams`, scale factor application.
  - Integration: Golden sample run that produces deterministic metadata (pose deltas, point counts). Skip heavy GPU paths in CI via marks.
  - Remote tool: Round-trip test that the GPU tool reads from NFS, writes to NFS, and `sense` reports expected artifacts and updates COLMAP config.

Remote GPU Architecture
- Flow: `hog/sense` captures → saves to NFS → MCP call to `ook` → VGGT runs on GPU → saves to NFS → `hog` updates configs/metrics.
- Hardening:
  - Queue/serialize VGGT jobs on `ook` if GPU contention; expose back-pressure signal in tool output.
  - Preflight on startup: verify model weights available and loadable; report VRAM.
  - Robust NFS sync checks with timeouts and retries for image and output file presence.
  - Fallback: If VGGT fails, continue with AprilTag + RealSense; record failure in `metrics.json`.

Milestones
- M0 Spike: Standalone `vggt_runner` that loads images and emits poses/depth/PLY locally; manual compare vs URDF.
- M1 Sense Integration (opt-in): Add VGGT mode to `sense` tool producing calibrated `Cams` + PLYs; report validation metrics; write frustum JSON/NPZ into dataset.
- M2 Viz Integration: Load VGGT PLYs in `viz/map.py`; optional frusta overlay for camera poses.
- M3 Remote GPU: Implement `tools/gpu/vggt_recon.py`; route from `hog` to `ook` with MCP; pre-cache weights on NFS.
- M4 Scale + BA: Introduce scale anchoring and optional BA path with tracks/COLMAP; document trade-offs and runtime.
- M5 COLMAP Config: Save extrinsics/intrinsics to `config/colmap/<scene>/`; update on sense runs.
- M4 Benchmarks: Collect metrics across 3–5 scenes; decide on replacing vs augmenting RealSense/AprilTags per use case.
- M6 Hardening: Configs, docs, error handling, and QA.

Execution Timeline (Guidance)
- Phase 1 (Week 1–2): M0–M1 deliverables + non-BA path; verify storage and config updates.
- Phase 2 (Week 3): M2 compare viz including confidence slider and frustum overlays.
- Phase 3 (Week 4): M3–M5 with BA optional; performance/robustness improvements and documentation.

Open Questions
- Real-time needs: Do we require live depth/pose during teleop, or is offline sensing sufficient?
- Camera set: How many cameras per scene and which are eligible for VGGT (IP cams only, or also RealSense color)?
- Capture protocol: How many frames, which angles, and spacing for robust recon in the workspace?
- Scale source: Which scale anchor do we standardize on (tag board vs RealSense reference vs URDF distances)?
- BA path: Do we invest in `pycolmap` dependency and tracking, or prefer the feedforward only path initially?
- COLMAP config: Exact layout under `config/colmap/` (per-scene vs per-camera) and whether to commit binary or text format.

Next Steps
- Confirm constraints (GPU availability, weight caching location, tolerance thresholds) and desired operating mode (offline vs online).
- Implement `tatbot/cam/vggt_runner.py` with a minimal feedforward path; add a small script or tool hook to run on existing recordings.
- Run M0/M1 on a calibration scene; review metrics and decide whether to proceed with M2+.
