# Todo

Roughly sorted by priority

- resample paths based on distance between poses, don't enforce a pathlen, enforce a posedist
- bimanual using ik collision (faster tattoo, plays to strengths of tatbot)
- two color pattern (black w/ white higlights) seperated by arms
- three color pattern (black with 3d effect using blue andred) with one needle switch
- pattern wrapping over skin mesh using jax
- mcp to expose all robot nodes as tool for agent, mcp replace ros as middleware
- isaacsim (https://www.youtube.com/live/z7KdHGkUTNE)
- compare fake skin and real skin calibration pattern
- squeeze bottle reveal ASMR
- right hand orbiting movement for optimal skin reconstruction
- random start spawn for design and inkcap for domain randomization in episode recording
- train splats on lerobot style data
- crop wrist pointcloud using cone from left hand, combine with cropped head pointcloud for skin mesh
- swiftsketch pattern
- etymology section in paper
- tatbot website
- host policy on replicate
- camera pose as joint in urdf frame for extrinsic calibration
- policy switching and composition into longer behaviors
- apriltag on righthand wrist frame for extrinsic camera calibration
- use ik solver for extrinsic camera calibration using apriltags
- rust apriltag rtsp synced image and video mcp server
- rcam runs on raspberry pis, provides synced camera images and apriltag poses. meerkat runs lerobot controlling arms and realsenses. ojo runs policy server, skin reconstruction.
- VR teleop using pyroki ik
- inkdip as policy
