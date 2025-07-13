# PoE IP Cameras

tatbot uses 5 poe ip cameras to create a 3d skin reconstruction: see `src/data/cam.py` and `config/cam/default.yaml`.
the cameras are currently set at:
- resolution: 1920x1080
- fps: 5
- bitrate CBR max 2048
- frameinterval: 10
- no substream, all watermarks off 