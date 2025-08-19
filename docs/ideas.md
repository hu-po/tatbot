# Ideas

#### Agents

#### Models
- try out https://github.com/allenai/MolmoAct
- lerobot async policy server/client pattern
- containers on ojo via jetson-containers https://github.com/dusty-nv/jetson-containers/tree/master/packages/ml/jax
- ink dipping as policy, random position start for domain randomization in episode recording
- host fintuned tatbot policy on replicate, use cloud compute for inference

#### Dataset
- data augmentation via nvidia cosmos
- maniskill synthetic data
- ink information as text conditioning
- camera poses and pointclouds in dataset
- isaacsim (https://www.youtube.com/live/z7KdHGkUTNE)
- gsplat from multiview conditioning images

#### Quality, Debugging
- visualization of cpu, memory, disk usage on nodes to check for bottlenecks
- visualization of network traffic between nodes to see if there is a bottleneck with mcp or nfs
- do strokes need to be the same size because the batch size is already arbitrary and coming from concatenation
- teleop calibration, visualization server as optional add ons to any stroke tool call
- compare JAX ik with Nvidia warp ik (https://github.com/NVIDIA/warp/blob/main/warp/examples/sim/example_jacobian_ik.py)
- compare current camera extrinsic calibration with colmap/Xm2
- tune camera resolution for multiview reconstruction

#### Features
- strokes are randomly sampled from strokelist, so order can be random and you can do repeated passes
- queues as core abstractions, arms operate async in their own behavior tree, coordinate centrally.
- queueu UI showing each arm's queue, ability to edit, remove, add to queue. pause arm. see time remaining.
- VLM via API tool to ask if scene is correctly set up (ee alignment, arms in sleep pose, etc)
- batch ik occurs in batches throughout stroke execution rather than all at once in beginning
- STT for agent interaction in ook, eek, rpi1
- pause to request operator needle switch
- sensor fusion for realsense pointcloud and ip camera gsplat
- sync camera images via rust on rpi1/rpi2

#### Social media, SEO, Marketing
- improve tatbot website currently built off of docs
- create PR to add to https://github.com/mjyc/awesome-robotics-projects
- submit to https://devpost.com/submit-to/25802-openai-open-model-hackathon/manage/submissions
- Tattoo outdoors to show off edge compute capability
- Paper flyers to put up in random places around ATX

#### Artwork
- potential improvement in gcode/stroke generation with https://github.com/abey79/vpype
- Medieval engraving tattoo https://youtube.com/shorts/lAyhxgaxQdc
- replicate playground on touchscreen

#### Hardware
- gather feedback on red sharpie cross with laser line leveler workflow
- end effector redesign: suspension, scissor lift, needle guard, sanitation barrier
- better joystick: smoother feedback, seperate joystick for each arm
- apriltags on robot ee to auto determine offsets
