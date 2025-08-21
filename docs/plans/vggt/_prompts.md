# 🤖 Models
- gpt5 via codex
- opus/sonnet 4 via claude code


# Initial Prompt

I am considering using VGGT for tatbot. I copy pasted two demo scripts from the VGGT repo at src/tatbot/cam/vggt_demo_viser.py and src/tatbot/cam/vggt_demo_colmap.py. Evaluate and create a planning document for me that determines whether VGGT could be used to replace some of the functionality in src/tatbot/tools/robot/sense.py, src/tatbot/viz/map.py, src/tatbot/cam/extrinsics.py, src/tatbot/cam/depth.py, and other potentially relevant files. Create your planning document in @docs/plans/vggt/MODEL_plan.md

# Second Prompt

Refine your plan based on these additional details:
- I want to keep the sense.py functionality, but add to it the VGGT extrinsic/intrinsic estimation and dense reconstruction.
- Pointclouds and images should be stored in the nfs in lerobot format, as is already done in sense.py
- Since VGGT requires a GPU, it will have to run on ook. Cameras are only connected to hog, so we will have to implement a remote gpu tool that is similar to the convert_strokes.py pattern used for the batch ik.
- I want you to create a new viz tool that allows me to view the vggt dense reconstruction and the pointclouds coming from the realsenses so I can visually compare the two. Also add camera frustrums to compare the apriltag and vggt solutions. You will have to save them to file during the scan. Use the BaseVizConfig to match existing viz tools.
- Save the extrinsics/intrinsics for cameras in COLMAP format in a new folder in config. The sense behavior should update these config files similar to updating the urdf.

# Third Prompt

You are being evaluated against a competing model. They also created a planning document in the same folder. Make sure to read their plan and understand their ideas. Think about how you could improve your own ideas based on their ideas. Refine your plan based on your new understanding.

# Fourth Prompt

Refine your plan based on these additional details:
- dont send the full information between nodes. For example, VGGTResult has all the images and pointclouds. Instead, save those to file and instead communicate the file paths instead.
- VGGT model weights should be stored in the huggingface cache of the node, not the nfs.

# Fifth Prompt

Compare the current diff to our plan. Create a new feedback document that reviews the diff in @docs/plans/vggt/diff_feedback.md make sure to point out missing functionality, incorrect assumptions, sloppy code, etc.

# Sixth Prompt

Perform another round of feedback on the current diff. This time focus on making sure that the code is correct and that it contains no errors. Put your feedback in @docs/plans/vggt/diff_feedback_2.md
---
orphan: true
---
