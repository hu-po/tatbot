what is the right runtime abstraction?
- mesh
- point cloud
- hashgrid
- volume

https://nvidia.github.io/warp/modules/runtime.html#meshes

https://developer.nvidia.com/warp-python

https://github.com/NVIDIA/warp/blob/main/warp%2Fexamples%2Fsim%2Fexample_rigid_soft_contact.py

https://github.com/facebookresearch/fast3r

https://fast3r-3d.github.io/

8 views	takes 0.122 seconds and	6.33 GB of VRAM

NeuS2: surface reconstruction

https://github.com/19reborn/NeuS2

OpenDroneMap: create .ply .obj from a folder of jpgs, entirely through docker
seems popular 5k stars, recent commits

https://github.com/dusty-nv/jetson-containers/tree/master/packages/robots/opendronemap

# Asking AIs

grok3 https://x.com/i/grok/share/rPdUxOtIfxWLJLtEAxurW5Ivv

meshroom is too old, opendrone map and fast3r do not support rgbd

nerfstudio requires transforms.json, can export mesh
https://docs.nerf.studio/nerfology/methods/splat.html

gpt deep research https://chatgpt.com/share/67e1ff3c-cb38-8009-b06a-9b4c773bb946

raw point cloud or high resolution mesh

recommends meshroom due to the automation and support, but it doesn't work on arm, so it would have to run on the trossen-ai pc

fast3r could be the fastest (or splatfacto in nerfstudio), but the accuracy might be lower given no transforms

multistep (1) camera calibration process  (2) image matching (3) training (4) mesh reconstruction (5) mesh refinement

https://embodied-gaussians.github.io/

gemini deep research:

high resolution images, optimize camera settings ISO, shutter speed, and aperture

significant amount of overlap, ideally between 60% and 80%

meshes over splats, meshroom and nerfstudio over fast3r

perform an initial high-fidelity scan (using photogrammetry) and then overlay live sensor data (e.g., from RGB-D cameras) to adjust for slight movements.

Publish the mesh or point cloud as a sensor_msgs/PointCloud2 or geometry_msgs/Mesh topic

##  VGGT

https://arxiv.org/pdf/2503.11651
https://github.com/facebookresearch/vggt
https://huggingface.co/spaces/facebook/vggt

```bash
python scripts/oop/camera-snapshot.py --mode image
```

kinda works, but the error is still ~1cm between the different images, which is similar to dust3r and mast3r. Might not actually be usable.