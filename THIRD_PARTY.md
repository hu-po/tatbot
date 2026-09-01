# Third-party assets

Everything in this repository is MIT licensed (see `LICENSE`) except the
files listed here, which come from elsewhere and keep their upstream terms.

## Robot meshes — `urdf/meshes/wxai/`, `urdf/meshes/cameras/d405.stl`

The WidowX AI arm meshes (`base_link.stl`, `link_1..6.stl`,
`carriage_left/right.stl`, `gripper_left/right.stl`, `handle_left.stl`,
`leader_finger_*.stl`, `camera_mount_d405.stl`) and the RealSense D405 body
are taken unmodified from
[TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description)
(`meshes/wxai/`, `meshes/peripherals/`), **BSD-3-Clause**. Verified
byte-identical to upstream.

## Trossen arm SDK

`cpp/teleop` links `libtrossen_arm` from
[TrossenRobotics/trossen_arm](https://github.com/TrossenRobotics/trossen_arm)
(pinned v1.8.5, fetched at configure time), **BSD-3-Clause**. Not vendored
here.

## AprilTag imagery — `urdf/meshes/tags/`

Tag faces render the AprilTag `16h5` family from the
[AprilTag project](https://github.com/AprilRobotics/apriltag), **BSD-2-Clause**.

## Web dependencies

`web/inkmap/src/vendor/vectortracer/` is a prebuilt WASM module from its
upstream project; see that directory for its own license and build metadata.
Runtime JS/TS dependencies are declared in `web/inkmap/package.json` and are
not vendored here.

## Our own work

Everything else under `urdf/meshes/` is modelled for this project and is MIT
like the rest of the tree: `ee/` (the printed end-effector mount, fiducial
cube and pen parts), `frame/palette.stl`, `frame/t2020_*.stl` (aluminium
extrusion) and `cameras/amcrest.stl` (camera body).
