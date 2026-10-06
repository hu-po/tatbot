"""Bounded camera-local evidence in the existing shared Calibration layout.

One camera per recording: no world extrinsics, camera fusion or new blueprint.
Geometry is diagnostic even when the upstream surface fit is supported.
"""

import json
import re

import cv2
import numpy as np
import tatbot_rerun as tr
from stencil_overlay import annotated
from stencil_scene import objects
from stencil_surface import rgbd_frame

ROOT = "calibration/stencil_camera"
MAX_FPS = 5
MAX_DEPTH_POINTS = 20000


def camera_entity(sensor):
    """Match visiond's camera_entity mapping for the shared camera panes."""
    if re.fullmatch(r"camera\d+", sensor):
        return f"cameras/{int(sensor[6:]):02}_{sensor}/image"
    match = re.fullmatch(r"realsense(\d+)_color", sensor)
    if match:
        return f"cameras/{5+int(match[1]):02}_realsense{match[1]}/color"
    if sensor == "overhead_depth_color":
        return "cameras/08_overhead_depth/color"
    return "cameras/99_stencil_input/image"


def depth_points(frame):
    rgbd, warnings = rgbd_frame(frame)
    depth = rgbd['depth_m']
    step = max(1, int(np.ceil(np.sqrt(depth.size/MAX_DEPTH_POINTS))))
    sampled = depth[::step, ::step]
    valid = np.isfinite(sampled) & (sampled > .03) & (sampled < 2.)
    points = (rgbd['rays'][::step, ::step]*sampled[..., None])[valid]
    colors = cv2.cvtColor(frame['image'], cv2.COLOR_BGR2RGB)[::step, ::step][valid]
    return points, colors, warnings


class StencilRerun:
    def __init__(self, args, output, run_id, references=None):
        self.rr = tr.start('stencil_observer', connect=args.connect,
                           output=output/'replay.rrd' if args.rerun else None,
                           recording_id=args.recording_id or run_id, view_coordinates=False)
        self.rr.log(ROOT, self.rr.ViewCoordinates.RDF, static=True)
        self.next_capture = 0
        self.limiter = tr.RateLimiter(MAX_FPS)
        self.live = args.mode == 'observe'
        self.references = references or {}

    def log(self, frame, observation, meshes):
        stamp = frame['timestamp_ns']
        if stamp < self.next_capture or (self.live and not self.limiter.ready()):
            return
        self.next_capture = stamp+1_000_000_000//MAX_FPS
        tr.set_capture_time(stamp)
        rr = self.rr
        sensor = frame.get('color_metadata', {}).get('sensor_name', 'stencil_input')
        shown = annotated(frame['image'], objects(observation), getattr(self, 'references', {}), camera=sensor)
        rr.log('surface/stencil/image', rr.Image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=85))
        rr.log(camera_entity(sensor), rr.Image(cv2.cvtColor(frame['image'], cv2.COLOR_BGR2RGB)).compress(jpeg_quality=85))
        # Clear each display frame so loss cannot leave a previous mesh or pose visible.
        rr.log(ROOT+'/stencils', rr.Clear(recursive=True))
        rr.log(ROOT+'/raw_depth', rr.Clear(recursive=True))
        rr.log(camera_entity(sensor).rsplit('/', 1)[0]+'/depth', rr.Clear(recursive=True))
        warnings = []
        try:
            points, colors, warnings = depth_points(frame)
            rr.log(ROOT+'/raw_depth', rr.Points3D(points, colors=colors, radii=.0007))
            rr.log(camera_entity(sensor).rsplit('/', 1)[0]+'/depth',
                   rr.DepthImage(frame['depth_m'].astype(np.float32), meter=1.))
            depth_status = f'{len(points)} measured depth points (unregistered scene)'
        except (ValueError, KeyError, TypeError, ImportError) as error:
            depth_status = 'depth unavailable: '+str(error)
        states = []
        for row in objects(observation):
            self._stencil(row, meshes.get(row['pattern_id']))
            surface = row.get('surface', {})
            states.append(f"{row.get('seed')}: {row['status']}; {surface.get('reason', 'surface disabled')}")
        message = '\n'.join([f'Camera-local stencil evidence: {sensor}', depth_status, *states,
                             *warnings, 'Diagnostic only; geometry_valid=false; no world alignment or fusion.',
                             'Recorded capture_time is preserved; this display does not establish freshness.'])
        rr.log(ROOT+'/status', rr.TextLog(message))
        rr.log('session/presentation/info', rr.TextDocument(message))

    def _stencil(self, row, mesh):
        rr = self.rr
        entity = ROOT+'/stencils/'+str(row['pattern_id'])
        surface = row.get('surface', {})
        if not surface.get('candidate_valid'):
            return
        if mesh is not None and len(mesh['triangles']):
            rr.log(entity+'/measured_patch', rr.Mesh3D(vertex_positions=mesh['vertices_camera_m'],
                   triangle_indices=mesh['triangles'], albedo_factor=[60, 220, 160]))
        anchors = surface.get('anchors', [])
        if anchors:
            rr.log(entity+'/anchors', rr.Points3D([a['point_camera_m'] for a in anchors], colors=[255, 170, 20], radii=.0015))
        pose = surface.get('camera_from_stencil_center')
        if pose is not None:
            pose = np.asarray(pose)
            rr.log(entity+'/center', rr.Points3D([pose[:3, 3]], labels=[str(row.get('seed'))+' candidate'], radii=.002))
            rr.log(entity+'/axes', rr.Arrows3D(origins=np.tile(pose[:3, 3], (3, 1)),
                   vectors=pose[:3, :3].T*.025, colors=[[240, 50, 50], [50, 220, 50], [50, 100, 240]]))
        rr.log(entity+'/quality', rr.TextLog(json.dumps({key: value for key, value in surface.items() if key != 'anchors'})))

    def close(self):
        self.rr.get_global_data_recording().flush()
