"""One stencil observer: every fixed view, the overhead RGB-D and each arm's
wrist RGB-D each turn, every visible print's pose with the support it rests on.

The fleet service (`stencild`) runs this on the camera node with the
installed references and an identity world. A wrist view is posed through
the arm's own registration and the URDF at the joints the service paired
with the frame, with the frame's own active intrinsics; the tracked-wrist
topic is never an input. This process opens no device and holds no motion
authority: a print's pose is a number for the daemon to compare, never a
permission.
"""

import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts/lib'), str(REPO/'scripts')]
import stencil_reference  # noqa: E402
from kinematic_calibration import KEY as JOINT_OFFSETS_KEY  # noqa: E402
from kinematic_calibration import validate_offsets  # noqa: E402
from live_inputs import capture, load_frames, rgbd_pair  # noqa: E402
from robot_world import registration, rigid_transform, root_from_world  # noqa: E402
from stencil_coded_live import REFUSALS, CodedBank, drawing_pads, pad_area  # noqa: E402
from stencil_features import Settings  # noqa: E402
from stencil_outline import report_outline  # noqa: E402
from stencil_plane_match import ROTATION_SIGMA_RAD, TRANSLATION_SIGMA_M, find_page, page_corners  # noqa: E402
from stencil_scan import ScanBinding  # noqa: E402
from stencil_surface import rgbd_frame  # noqa: E402
from stencil_surface_fit import bootstrap_pose, center_pose, choose_model, design  # noqa: E402
from stencils import bundle  # noqa: E402
from surface_rgb import MAX_POINTS, ScanAppearance  # noqa: E402
from tatbot_cli.arms import load as configured_arms  # noqa: E402
from urdf_kinematics import UrdfChain, driver_joint_names  # noqa: E402
from view_assets import read  # noqa: E402
from wrist_cameras import optical_frames, registry  # noqa: E402

SCHEMA = 'tatbot.stencil-observer-turn/1'
SUPPORT_SCHEMA = 'tatbot.target-support/1'
MAX_AGE_NS = 3_000_000_000
# One expensive fixed-camera reference search every three turns leaves fresh
# publications between reacquisition attempts while an arm obscures a page.
FIXED_SEARCH_EVERY = 3
MIN_ANCHORS = 12
FIT_ANCHORS = 256          # the fit and its bootstrap keep this many measured anchors
PAGE_CORNERS_UV = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
WRIST_ROLE = 'wrist-'
# A coded print is decoded, 5-120 s per view on the camera node, so it is searched only on each
# arm's drawing pad (a square this far either side of its pad pivot, through the arm's
# registration; a close wrist view searches the part it sees) and around prints a view already
# tracks; with no registered pad a small view is searched whole. The decode runs in one
# background process, one view at a time, and never holds up a turn; its flow track and bits
# carry the print between decodes.
PAD_SEARCH_HALF_M = .25
CODED_WHOLE_IMAGE_PX = 1_400_000
# Seconds the background decode may search one view for components; past it what was found is
# still decoded. A pad square on a cluttered real frame takes 1-2 min on the camera node.
CODED_DECODE_S = 90.


class WristPoser:
    """world <- wrist colour camera for one arm: the arm's own
    `tatbot.arm-registration/1` (accepted on the observer's bundle or carried
    onto it, `robot_world.registration`) times the URDF chain at
    the measured joints. The registration files are read once; the service
    restarts the observer when they change. Offline readers may pass a retained
    configuration root; deployment callers keep the checkout default."""

    def __init__(self, binding, calibration, *, repo=REPO):
        wrist = binding.get('wrist') or {}
        self.repo = Path(repo)
        self.calibration = calibration
        self.registrations = {arm: Path(path) for arm, path in (wrist.get('registrations') or {}).items()}
        golden = wrist.get('robot_world_golden')
        self.golden = json.loads(read(golden, 1024*1024)) if golden else None
        self.urdf = Path(wrist.get('urdf') or self.repo/'urdf/tatbot.urdf')
        self.vision_config = Path(wrist['vision_config']) if wrist.get('vision_config') else None
        self.arms = configured_arms(self.repo)
        from tool_spec import read_workspace
        workspace = read_workspace(self.repo)
        self.joint_offsets = {arm: validate_offsets(workspace.get(arm, {}).get(JOINT_OFFSETS_KEY))
                              for arm in self.arms}
        self.cameras = registry(self.repo, self.vision_config)
        self.by_camera = {camera['name']: camera for camera in self.cameras if camera.get('arm')}
        self.counts = {arm: sum(camera.get('arm') == arm for camera in self.cameras)
                       for arm in self.arms}
        self._chain = None
        self._frames = {}
        self._world_from_root = {}

    @property
    def chain(self):
        if self._chain is None:
            self._chain = UrdfChain(self.urdf)
        return self._chain

    def camera_binding(self, arm, camera):
        """A camera has one physical arm and one configured optical frame."""
        if arm not in self.arms:
            raise ValueError(f'wrist capture names unconfigured arm {arm!r}')
        configured = self.by_camera.get(camera)
        if configured is None or configured['arm'] != arm:
            raise ValueError(f'{camera}: no wrist camera is configured on the {arm} arm')
        role = (f'wrist-{arm}' if self.counts[arm] == 1 else f'wrist-{arm}-{camera}')
        if arm not in self._frames:
            self._frames[arm] = optical_frames(
                self.repo, arm=arm, stream='color', config=self.vision_config, urdf=self.urdf)
        return role, self._frames[arm][configured['role']]

    def world_from_root(self, arm):
        """The arm's registration as world <- URDF root, carry included, with
        its provenance; refused by name when the arm has no registration or
        it was solved against another world."""
        if arm not in self._world_from_root:
            path = self.registrations.get(arm)
            if path is None:
                raise ValueError(f'the {arm} arm has no registration installed beside the observer')
            record = json.loads(read(path, 4*1024*1024))
            world_from_arm_base, provenance = registration(arm, self.calibration, record, self.golden)
            root_from_arm_base = (np.linalg.inv(rigid_transform(record['world_from_root']))
                                  @ rigid_transform(record['world_from_arm_base']))
            provenance['registration_sha256'] = hashlib.sha256(read(path, 4*1024*1024)).hexdigest()
            self._world_from_root[arm] = (world_from_arm_base @ np.linalg.inv(root_from_arm_base), provenance,
                                          world_from_arm_base)
        return self._world_from_root[arm][:2]

    def world_from_arm_base(self, arm):
        """The arm's registration as world <- arm base, carry included."""
        self.world_from_root(arm)
        return self._world_from_root[arm][2]

    def pose(self, arm, camera, joints, carriage_m):
        joints = np.asarray(joints, float)
        if joints.shape != (6,) or not np.isfinite(joints).all() or not np.isfinite(carriage_m):
            raise ValueError('a wrist view needs six finite measured joints and a finite carriage')
        world_from_root, provenance = self.world_from_root(arm)
        _, frame = self.camera_binding(arm, camera)
        prefix = self.arms[arm].urdf_prefix
        corrected = np.r_[joints, carriage_m] + self.joint_offsets[arm]
        values = dict(zip(driver_joint_names(prefix, 7), corrected.tolist(), strict=True))
        return world_from_root @ self.chain.link_pose(frame, values), {
            **provenance, JOINT_OFFSETS_KEY: self.joint_offsets[arm]}


def reference_images(paths):
    """{pattern_id: the reference artwork, grey} for each installed manifest whose image reads."""
    images = {}
    for path in paths:
        try:
            manifest = json.loads(read(path, 64_000))
            image = cv2.imread(str(Path(path).parent / manifest['image']['file']), cv2.IMREAD_GRAYSCALE)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if image is not None:
            images[manifest['pattern_id']] = image
    return images


def usable_references(paths):
    """The installed references this observer can track, and the others with the reason: a
    manifest that fails to load (a coded one without its bound coded.json) or a coded print the
    decoder cannot read. One bad reference never stops the others' tracking; it is published
    lost with its reason (`refused_target`). None usable refuses the start, naming why."""
    usable, refused = [], {}
    for path in paths:
        try:
            manifest, _ = stencil_reference.load(path)
            if stencil_reference.is_coded(manifest):
                CodedBank([path])
            usable.append(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            refused[str(path)] = str(error) or type(error).__name__
    if refused and not usable:
        raise ValueError('no usable stencil reference: ' + '; '.join(f'{k}: {v}' for k, v in refused.items()))
    return usable, refused


def refused_target(path, reason, bundle_id):
    """A `lost` target for a refused reference whose manifest still names its pattern and
    reference id, so a consumer waiting for it reads why; None otherwise."""
    try:
        manifest = json.loads(read(path, 64_000))
    except (OSError, ValueError):
        return None
    pattern, reference_id = manifest.get('pattern_id'), manifest.get('reference_id')
    if not (isinstance(pattern, str) and re.fullmatch(r'stencil-[0-9a-f]{64}', pattern)
            and isinstance(reference_id, str) and re.fullmatch(r'[0-9a-f]{64}', reference_id)):
        return None
    reason = f'reference refused: {reason}'
    return {'pattern_id': pattern, 'reference_id': reference_id, 'reference_physical_instance_id': None,
            'physical_instance_id': None, 'physical_instance_identity_verified': False,
            'physical_instance_capture_ns': None, 'seed': str(manifest.get('seed')), 'target_frame': 'world',
            'source': 'lost', 'world_from_target': None, 'translation_sigma_m': None, 'rotation_sigma_rad': None,
            'capture_ns': int(time.time_ns()),
            'support': {'schema': SUPPORT_SCHEMA, 'anchor_camera': None, 'anchors': 0, 'cameras': {},
                        'anchor_points_m': [], 'reason': reason, 'motion_authority': False,
                        'physical_instance_identity_verified': False, 'bundle_id': bundle_id}}


def pad_search_areas(poser, robot):
    """Root-frame squares on every registered arm's drawing pad, where a coded print is
    searched for (None when no pad has a registration: a small view is then searched whole
    and a large one not at all), and why an arm with a pad has none."""
    areas, refused = [], {}
    for arm, pad in drawing_pads(REPO).items():
        try:
            area = pad_area(poser.world_from_arm_base(arm), pad, PAD_SEARCH_HALF_M)
        except (OSError, ValueError, KeyError, TypeError) as error:
            refused[arm] = str(error)
            continue
        root = root_from_world(robot)
        areas.append(area@root[:3, :3].T+root[:3, 3])
    return areas or None, refused


def wrist_view(path, calibration, robot, poser, source=None):
    """One arm's wrist RGB-D view as the fusion reads it: `(name, (frame,
    root_from_camera), provenance)` from the service's `wrist-<arm>` capture —
    the `<camera>_color`/`<camera>_depth` pair of the arm's own D405 with the
    active intrinsics the owner published, posed through the registration
    and the URDF at the joints the service paired with the exposure."""
    manifest, frames = load_frames(Path(path))
    wrist = manifest.get('wrist')
    if not isinstance(wrist, dict) or wrist.get('arm') not in poser.arms:
        raise ValueError('wrist capture names no arm')
    arm, camera = wrist['arm'], wrist['camera']
    role, _ = poser.camera_binding(arm, camera)
    if source is not None and source != role:
        raise ValueError(f'{source}: capture belongs to {role}')
    if manifest.get('geometry_calibration_id') != calibration['bundle_id']:
        raise ValueError('wrist capture was paired under another camera bundle')
    try:
        (color, _), (depth, _) = frames[f'{camera}_color'], frames[f'{camera}_depth']
    except KeyError as error:
        raise ValueError(f'wrist capture carries no {error.args[0]} frame') from None
    cm, dm = color['metadata'], depth['metadata']
    for metadata in (cm, dm):
        if metadata.get('attributes', {}).get('physical_arm') != arm:
            raise ValueError(f"{metadata['sensor_name']}: capture is of the "
                             f"{metadata.get('attributes', {}).get('physical_arm')!r} arm, not {arm!r}")
    intrinsics = json.loads(cm['attributes']['intrinsics'])
    if (intrinsics.get('width'), intrinsics.get('height')) != (cm['profile']['width'], cm['profile']['height']):
        raise ValueError(f'{camera}: active intrinsics disagree with the image profile')
    stamp = cm['timestamps']['normalized_unix_ns']
    window = manifest['wrist_capture_window']
    if not window['after_ns'] <= stamp <= window['before_ns']:
        raise ValueError('wrist exposure outside capture window')
    frame = rgbd_pair(frames, f'{camera}_color')
    world_from_camera, registration = poser.pose(arm, camera, wrist['joints'], wrist['carriage_m'])
    provenance = {'arm': arm, 'camera': camera, 'capture_ns': stamp, 'measured_wall_ns': wrist['measured_wall_ns'],
                  'joints_skew_ms': wrist.get('joints_skew_ms'), 'joints_calibration_id': wrist.get('joints_calibration_id'),
                  'registration': registration, 'intrinsics': 'owner active', 'motion_authority': False}
    return role.replace('-', '_'), (frame, root_from_world(robot) @ world_from_camera), provenance


class LiveSurface:
    def __init__(self, binding, directory):
        self.directory = Path(directory)
        self.refused_references = {}
        stencils = binding.get('stencils')
        if stencils is None:
            paths, self.refused_references = usable_references(binding['references'])
            stencils = bundle(paths)
        # The service samples at 1 Hz. Its first full-camera census can take
        # much longer; LK must be allowed to bridge that one cold-start gap.
        # Each flow result still needs a valid homography and current depth.
        self.appearance = ScanAppearance(stencils, self.directory,
                                         tracking_settings=Settings(max_gap_ms=20_000,
                                                                    verify_interval_ms=180_000),
                                         coded_options={'background': True,
                                                        'whole_image_px': CODED_WHOLE_IMAGE_PX,
                                                        'decode_s': CODED_DECODE_S})
        self.calibration, self.robot = binding['calibration'], binding['robot_world']
        self.references = {row['reference']['pattern_id']: row['reference'] for row in stencils['references']}
        self.inventory_sha256 = hashlib.sha256(
            json.dumps(sorted(row['reference_id'] for row in self.references.values())).encode()).hexdigest()
        self.excluded_anchors = frozenset(binding.get('excluded_anchors') or ())
        self.capture_reader = capture
        self.scan = ScanBinding(self.directory.parent)
        self.sequence, self.latest = 0, None
        self.anchor_by_pattern = {}
        self.depth_roi = None
        self._candidate_cache = {}
        self.publish_targets_only = bool(binding.get('publish_targets_only', False))
        self.provenance = {key: binding.get(key) for key in ('observer_epoch', 'evidence_kind')}
        self.provenance['calibration_id'] = self.calibration.get('bundle_id')
        self.poser = WristPoser(binding, self.calibration)
        self.appearance.search_areas, self.pad_search_refused = pad_search_areas(self.poser, self.robot)
        self.wrist_views = {}
        self.ambiguous_patterns = set()
        # Where no fit measures a print, the fixed RGB-D view may place it by its artwork on the table plane
        # (stencil_plane_match): opt-in, for a fleet whose only fixed camera cannot resolve the print.
        self.artwork_match = bool(binding.get('overhead_artwork_match', False))
        self.reference_images = reference_images(binding.get('references') or ()) if self.artwork_match else {}
        self.overhead_views, self.artwork_x, self.artwork_reasons = {}, {}, {}

    def observe(self, views, errors, now):
        started = time.perf_counter()
        timings = {}
        mark = started
        # The source set belongs to one request. No previous image is eligible
        # for cross-camera support, including a camera absent from this reply.
        appearance = self.appearance
        appearance.begin_turn()
        appearance.reports.clear()
        for source in list(appearance.views):
            appearance.unavailable(source, 'not in current exposure window')
        for source, reason in errors.items():
            appearance.unavailable(source, reason)
        fixed = sorted(name for name, (frame, _) in views.items() if 'depth_m' not in frame)
        depth = sorted(name for name in views if name not in fixed)
        # Keep flow tracks current in every fixed view. A changed or hidden
        # print can make SIFT take two seconds per camera; only one fixed view
        # may do that search in a turn, including a turn that loses support.
        self._observe_fixed(views, fixed, now)
        timings['fixed_views'] = (time.perf_counter()-mark)*1000
        mark = time.perf_counter()
        tracked = self._tracked_by_eligible_view(self._live_patterns())
        wrist_depth = {name for name in depth if name.startswith('wrist_')}
        self.overhead_views = {name: views[name] for name in depth if name not in wrist_depth}
        deferred = []
        for name in depth:
            frame, matrix = views[name]
            # A wrist view is one arm's close look at the page, posed by its
            # own measured joints: it is searched every turn it arrives. When
            # a wrist view is present, first project its current result into
            # the fixed overhead depth instead of also running an independent
            # overhead reference search. If the wrist and fixed views do not
            # support every print, `_cross_current_views` searches the deferred
            # overhead in the same turn. This keeps that fallback while
            # avoiding three simultaneous depth reference searches during a
            # two-arm occlusion.
            # With no fixed view at all nothing else ever searches: an empty live set reads as tracked, and
            # a print lost on its first turn was never searched again (the demo stack's lone D555).
            localize = name.startswith('wrist_') or not fixed or (not tracked and not wrist_depth)
            self._observe(frame, name, matrix, now, localize=localize)
            if not localize:
                deferred.append(name)
        timings['overhead_views'] = (time.perf_counter()-mark)*1000
        mark = time.perf_counter()
        selected = self._cross_current_views(views, fixed, deferred, now)
        timings['cross_views'] = (time.perf_counter()-mark)*1000
        mark = time.perf_counter()
        if self.publish_targets_only:
            result = {'schema': SCHEMA, 'frame': 'root', 'motion_authority': False}
        else:
            result = snapshot(appearance, self._tracking_camera(fixed))
            result['stencil_state'] = self.scan.observe(appearance.reports, now)
            if self.scan.bound is not None or self.scan.error is not None:
                # The binding's own anchors outline the page without a view.
                references = appearance.bank.references if appearance.bank is not None else {}
                result['surfaces'] = self.scan.display(result['stencil_state'], references)
        timings['snapshot_scan'] = (time.perf_counter()-mark)*1000
        mark = time.perf_counter()
        result['targets'] = self.targets(errors)
        result['wrist_views'] = dict(self.wrist_views)
        result['coded_search'] = {'pad_areas': len(appearance.search_areas or ()),
                                  'refused': dict(self.pad_search_refused)}
        result['refused_references'] = dict(self.refused_references)
        timings['target_fits'] = (time.perf_counter()-mark)*1000
        timings['per_camera'] = {name: report['timing_ms'] for name, report in appearance.reports.items()
                                 if 'timing_ms' in report}
        timings['observe_total'] = (time.perf_counter()-started)*1000
        result['timings_ms'] = timings
        result['inventory_sha256'] = self.inventory_sha256
        result['excluded_anchors'] = sorted(self.excluded_anchors)
        result['tracking_views_used'] = sorted(selected)
        self._update_depth_roi(result)
        self.sequence += 1
        result['observation_sequence'] = self.sequence
        result.update(self.provenance)
        self.latest = result
        self._candidate_cache.clear()
        return result

    def _observe_fixed(self, views, fixed, now):
        # The cold census establishes tracks in every view. Afterwards flow
        # is cheap enough to follow all of them, while full reference search
        # rotates through one view every few turns. This bounds publication latency
        # when a moving arm occludes a page and several cameras lose flow at
        # once. A different view can still anchor the current turn from its
        # own tracked pixels and depth; search reacquires lost views in later
        # turns without starving the other page's measurements.
        cold = self.latest is None
        search_turn = cold or len(fixed) == 1 or self.sequence % FIXED_SEARCH_EVERY == 0
        search_slot = self.sequence // FIXED_SEARCH_EVERY if len(fixed) > 1 else self.sequence
        periodic = fixed[(search_slot + self.sequence // 30) % len(fixed)] if fixed else None
        for name in fixed:
            frame, matrix = views[name]
            # A view that lost a coded print where it tracked it looks there every turn: that small
            # box decodes in seconds, where waiting for the rotation left the print lost for 10+
            # min after an arm passed over it (bench 2026-09-27).
            hint = self.appearance.coded_hint(name, frame['timestamp_ns'])
            self._observe(frame, name, matrix, now, localize=True, periodic_search=name == periodic or hint,
                          search=cold or (search_turn and name == periodic) or hint)

    def _lost_patterns(self):
        """The prints the last publication reported lost; none before one."""
        if self.latest is None:
            return set()
        return {target['pattern_id'] for target in self.latest['targets'] if target['source'] != 'measured'}

    def _live_patterns(self):
        """The prints whose absence is worth a search this turn: measured at
        the last publication, or tracked again by a fixed view that may anchor
        it (a hidden print coming back). A print lost at the last publication
        and unseen by every such view waits for its periodic search — the
        wrist view over a page an arm traces sees the print every turn while
        the arm's own cube hides it from the fixed views, and that look is no
        reason to search every fixed view every turn (2026-09-20)."""
        seen = set()
        for name, (frame, observation, _, _) in self.appearance.views.items():
            if 'depth_m' in frame or name in self.excluded_anchors:
                continue
            seen |= {row['pattern_id'] for row in observation['stencils'] if row['image_tracking_valid']}
        return (set(self.references) - self._lost_patterns()) | (seen & set(self.references))

    def _cross_current_views(self, views, fixed, deferred, now):
        appearance = self.appearance
        selected = self._tracking_views(fixed)
        self._candidate_cache.clear()
        appearance.cross_views(selected, depth_roi=self.depth_roi)
        live = self._live_patterns()
        if (self.depth_roi is not None or selected != set(fixed)) and not self._all_patterns_anchored(live):
            # A preferred camera or prior depth footprint can lose support.
            self._candidate_cache.clear()
            appearance.cross_views()
            selected = set(fixed)
        if deferred and not self._all_patterns_anchored(live):
            for name in deferred:
                frame, matrix = views[name]
                self._observe(frame, name, matrix, now, localize=True)
            self._candidate_cache.clear()
            appearance.cross_views()
            selected = set(fixed)
        return selected

    def _update_depth_roi(self, result):
        """One projection region per depth camera (the overhead, each wrist)
        from its own cross-camera attempts; the region is in that camera's
        pixels and never applied to another's."""
        appearance = self.appearance
        boxes = {}
        for source, report in appearance.reports.items():
            for row in report.get('stencils', ()):
                for attempt in row.get('cross_camera_attempts', ()):
                    if 'depth_pixel_bbox' in attempt:
                        boxes.setdefault(source, []).append(attempt['depth_pixel_bbox'])
        if boxes and all(target['source'] == 'measured' for target in result['targets']):
            self.depth_roi = {}
            for source, rows in boxes.items():
                limits = np.asarray(rows, int)
                self.depth_roi[source] = (int(limits[:, 0].min()-40), int(limits[:, 1].min()-40),
                                          int(limits[:, 2].max()+40), int(limits[:, 3].max()+40))
        else:
            self.depth_roi = None

    def _observe(self, frame, name, matrix, now, *, localize, periodic_search=True, search=True):
        self._candidate_cache.clear()
        try:
            if not 0 <= now-frame['timestamp_ns'] <= MAX_AGE_NS:
                raise ValueError('stale or future capture')
            self.appearance.observe(frame, name, matrix, localize=localize,
                                    periodic_search=periodic_search, search=search)
        except (ValueError, KeyError, TypeError, ImportError, cv2.error) as error:
            self.appearance.unavailable(name, error)

    def _tracked_by_eligible_view(self, patterns=None):
        """Every print (of `patterns`) tracked by some fixed view that may anchor it."""
        patterns = set(self.references if patterns is None else patterns)
        for name, (frame, observation, _, _) in self.appearance.views.items():
            if 'depth_m' in frame or name in self.excluded_anchors:
                continue
            patterns -= {row['pattern_id'] for row in observation['stencils'] if row['image_tracking_valid']}
        return not patterns

    def _tracking_views(self, fixed):
        """Current anchors plus independent current support for each print.

        Every fixed image is still tracked; only expensive depth projection is
        selected. A new bank starts with every camera, and missing support
        falls back to every camera in the same turn.
        """
        if self.latest is None:
            return set(fixed)
        selected = set()
        previous = {row['pattern_id']: row['support'] for row in self.latest['targets']}
        visible = self._visible_tracking_views(fixed)
        for pattern in self.references:
            choices = [name for name in visible.get(pattern, ()) if name not in self.excluded_anchors]
            if not choices:
                continue
            support = previous.get(pattern, {})
            anchor = support.get('anchor_camera')
            ranked = support.get('cameras') or {}
            selected.add(anchor if anchor in choices else max(choices, key=lambda name:
                         (ranked.get(name, {}).get('anchors', 0), name)))
        for pattern in self.references:
            choices = sorted(set(visible.get(pattern, ())) - selected)
            if choices and len(selected.intersection(visible.get(pattern, ()))) < 2:
                ranked = previous.get(pattern, {}).get('cameras') or {}
                selected.add(max(choices, key=lambda name: (ranked.get(name, {}).get('anchors', 0), name)))
        audit = sorted({name for names in visible.values() for name in names} - selected)
        if audit and self.sequence % 30 == 0:
            selected.add(audit[(self.sequence // 30) % len(audit)])
        return selected

    def _visible_tracking_views(self, fixed):
        visible = {}
        for name in fixed:
            if name not in self.appearance.views:
                continue
            for row in self.appearance.views[name][1]['stencils']:
                if row['image_tracking_valid']:
                    visible.setdefault(row['pattern_id'], []).append(name)
        return visible

    def _tracking_camera(self, fixed):
        """The fixed view with the most valid tracks, a display label only."""
        counts = {name: sum(row['image_tracking_valid'] for row in self.appearance.views[name][1]['stencils'])
                  for name in fixed if name in self.appearance.views}
        if not counts:
            return None
        return max(sorted(counts), key=counts.get)

    def _all_patterns_anchored(self, patterns=None):
        return all(any(candidate['anchor_eligible'] and candidate['center_supported']
                       for candidate in self._candidates(pattern))
                   for pattern in (self.references if patterns is None else patterns))

    def _candidates(self, pattern):
        """Every camera's measured fit for one print: the RGB-D view's own fit
        and each fixed view's cross-camera fit through that depth."""
        if pattern in self._candidate_cache:
            return self._candidate_cache[pattern]
        reference = self.references[pattern]
        candidates = []
        for source, report in self.appearance.reports.items():
            if 'stencils' not in report or report.get('tracking_only'):
                continue
            row = next((r for r in report['stencils'] if r['pattern_id'] == pattern), None)
            if row is None:
                continue
            root_from_depth = np.asarray(report['root_from_camera'], float)
            # The view's own fit is the report before any cross-camera fill
            # replaced it; every fill is an attempt under its tracking camera.
            direct = next((r for r in self.appearance.direct[source][1]['stencils'] if r['pattern_id'] == pattern), None)
            fits = [(source, direct['surface'], report['capture_timestamp_ns'])] if direct else []
            for attempt in row.get('cross_camera_attempts', []):
                stamps = [attempt.get('depth_timestamp_ns'), attempt.get('tracking_timestamp_ns')]
                fits.append((attempt['tracking_camera'], attempt,
                             min(s for s in stamps if s is not None) if any(s is not None for s in stamps)
                             else report['capture_timestamp_ns']))
            for camera, fit, stamp in fits:
                if (reference.get('physical_instance_encoded')
                        and (not fit.get('physical_instance_identity_verified')
                             or fit.get('physical_instance_id') != reference['physical_instance_id'])):
                    continue
                candidate = fit_target(camera, fit, reference, root_from_depth, stamp,
                                       excluded=self.excluded_anchors)
                if candidate is not None:
                    candidate['physical_instance_id'] = fit.get('physical_instance_id')
                    candidate['physical_instance_identity_verified'] = bool(
                        fit.get('physical_instance_identity_verified'))
                    candidate['physical_instance_capture_ns'] = (
                        fit.get('tracking_timestamp_ns', report['capture_timestamp_ns'])
                        if candidate['physical_instance_identity_verified'] else None)
                    candidates.append(candidate)
        self._candidate_cache[pattern] = candidates
        return candidates

    def _anchor(self, pattern, eligible):
        anchor = max(eligible, key=lambda c: (c['anchors'], -c['loo_p95_m'])) if eligible else None
        previous = next((c for c in eligible if c['camera'] == self.anchor_by_pattern.get(pattern)), None)
        # Depth fits from different calibrated cameras can disagree by
        # millimetres. Retain the current camera when another fit is only
        # marginally better; switch immediately when support is lost or
        # the other view has substantially more measured anchors.
        if (anchor is not None and previous is not None
                and previous['anchors'] + 4 >= anchor['anchors']
                and previous['loo_p95_m'] <= 1.5 * anchor['loo_p95_m']):
            anchor = previous
        if anchor is not None:
            self.anchor_by_pattern[pattern] = anchor['camera']
        else:
            self.anchor_by_pattern.pop(pattern, None)
        return anchor

    def _artwork_anchor(self, pattern, reference, unmeasured):
        """The print placed by its artwork on the table in a fixed RGB-D view (stencil_plane_match), where no fit
        measured it: one page found on exactly one registered arm's pad, and this the only unmeasured print (the
        match cannot tell prints apart). None otherwise, the reason kept for the lost target."""
        self.artwork_reasons.pop(pattern, None)
        found, reasons = self._artwork_pages(pattern, reference, unmeasured)
        if len(found) != 1:
            self.artwork_reasons[pattern] = ('the artwork matched on several pads' if found
                                             else '; '.join(reasons) or 'no registered pad in a fixed RGB-D view')
            return None
        name, arm, result, stamp = found[0]
        pose, page_m = result['pose'], np.asarray(reference['page_mm'], float)/1000.
        self.artwork_x[pattern] = pose[:3, 0].copy()
        u0, v0, u1, v1 = reference.get('clear_center_uv', (0., 0., 1., 1.))
        return {'camera': name, 'anchors': 0, 'loo_p95_m': None, 'loo_rms_m': None, 'model': 'artwork_plane_match',
                'center_supported': True, 'anchor_eligible': True, 'pose': pose,
                'corners': page_corners(pose, page_m, PAGE_CORNERS_UV),
                'clear_corners': page_corners(pose, page_m, [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]),
                'anchor_points': np.zeros((0, 3)),
                'plane': {'point_m': pose[:3, 3].tolist(), 'normal': pose[:3, 2].tolist(),
                          'axes': 'x increasing u, y increasing v, z = x cross y'},
                'translation_sigma_m': TRANSLATION_SIGMA_M, 'rotation_sigma_rad': ROTATION_SIGMA_RAD,
                'sigma_method': 'artwork plane match floor (depth and registration bias)',
                'physical_instance_id': None, 'physical_instance_identity_verified': False,
                'physical_instance_capture_ns': None, 'capture_ns': int(stamp),
                'artwork_match': {'arm': arm, 'score': result['score'], 'margin': result['margin']}}

    def _artwork_pages(self, pattern, reference, unmeasured):
        """[(view, arm, find_page result, exposure)] for every page found, and why each look found none."""
        image = self.reference_images.get(pattern)
        if image is None:
            return [], ['no reference artwork']
        if unmeasured != [pattern]:
            return [], [f'{len(unmeasured)} installed prints are unmeasured and the artwork cannot tell them apart']
        root, page_m = root_from_world(self.robot), np.asarray(reference['page_mm'], float)/1000.
        found, reasons = [], []
        for name, (frame, matrix) in self.overhead_views.items():
            try:
                rgbd, _ = rgbd_frame(frame)
                model = frame['camera_model']
            except (ValueError, KeyError, TypeError) as error:
                reasons.append(f'{name}: {error}')
                continue
            k = np.array([[model['fx'], 0., model['ppx']], [0., model['fy'], model['ppy']], [0., 0., 1.]])
            dist = np.asarray(model['distortion_coefficients'], float)
            for arm, pad in drawing_pads(REPO).items():
                try:
                    world_from_base = self.poser.world_from_arm_base(arm)
                except (OSError, ValueError, KeyError, TypeError):
                    continue    # an arm without a registration in this world has no pad here
                area = pad_area(world_from_base, pad, PAD_SEARCH_HALF_M)@root[:3, :3].T+root[:3, 3]
                result = find_page(frame['image'], rgbd['depth_m'], rgbd['rays'], k, dist, np.asarray(matrix, float),
                                   image, page_m, area, root@world_from_base, self.artwork_x.get(pattern),
                                   clear_uv=reference.get('clear_center_uv'))
                if result['found']:
                    found.append((name, arm, result, frame['timestamp_ns']))
                else:
                    reasons.append(f"{name} on the {arm} pad: {result['reason']}")
        return found, reasons

    def _note_identity_ambiguity(self, pattern, eligible):
        """Latch visible duplicate copies or incompatible current positions."""
        if any(sum(row.get('pattern_id') == pattern for row in report.get('stencils', ())) > 1
               for report in self.appearance.reports.values()):
            self.ambiguous_patterns.add(pattern)
        for index, candidate in enumerate(eligible):
            for other in eligible[index+1:]:
                if abs(candidate['capture_ns']-other['capture_ns']) > 40_000_000:
                    continue
                distance = np.linalg.norm(candidate['pose'][:3, 3]-other['pose'][:3, 3])
                tolerance = max(.01, 4*(candidate['loo_p95_m']+other['loo_p95_m']))
                if distance > tolerance:
                    self.ambiguous_patterns.add(pattern)

    def _lost_reason(self, pattern, candidates, depth_absent, errors):
        if pattern in self.ambiguous_patterns:
            return 'ambiguous same-pattern print positions; new physical target binding required'
        refusals = {row.get('tracking_reason') for report in self.appearance.reports.values()
                    for row in report.get('stencils', ()) if row['pattern_id'] == pattern}
        marked_refusals = sorted(reason for reason in refusals
                                 if reason and (reason.startswith('instance_mark_') or reason in REFUSALS
                                                or reason == 'physical_instance_mismatch'))
        if marked_refusals:
            return '; '.join(marked_refusals)
        if depth_absent:
            return ('no overhead depth: ' + '; '.join(f'{k}: {v}' for k, v in errors.items())
                    if errors else 'no overhead depth')
        if not candidates:
            return 'no measured anchors'
        if not any(c['anchor_eligible'] for c in candidates):
            return 'only excluded cameras measured the print'
        return 'no eligible camera encloses the print centre'

    def targets(self, errors):
        """One target per reference: the anchor camera's measured pose, or
        `lost` with the reason, beside every camera's fit and disagreement."""
        targets = []
        depth_absent = not any('depth_m' in frame for frame, *_ in self.appearance.views.values())
        unmeasured = [p for p in self.references
                      if not any(c['anchor_eligible'] and c['center_supported'] for c in self._candidates(p))]
        for pattern, reference in self.references.items():
            candidates = self._candidates(pattern)
            eligible = [c for c in candidates if c['anchor_eligible'] and c['center_supported']]
            self._note_identity_ambiguity(pattern, eligible)
            anchor = None if pattern in self.ambiguous_patterns else self._anchor(pattern, eligible)
            if anchor is None:
                self.anchor_by_pattern.pop(pattern, None)
                if self.artwork_match and pattern not in self.ambiguous_patterns:
                    anchor = self._artwork_anchor(pattern, reference, unmeasured)
            if anchor is not None and 'artwork_match' not in anchor:
                target_uncertainty(anchor)
            cameras = {}
            for c in candidates:
                row = {key: c[key] for key in ('anchors', 'loo_p95_m', 'loo_rms_m', 'model',
                                                 'center_supported', 'anchor_eligible', 'capture_ns')}
                if anchor is not None:
                    row['centre_delta_mm'] = float(np.linalg.norm(c['pose'][:3, 3]-anchor['pose'][:3, 3])*1000)
                    cosine = np.clip(np.dot(c['pose'][:3, 2], anchor['pose'][:3, 2]), -1, 1)
                    row['normal_delta_deg'] = float(np.degrees(np.arccos(cosine)))
                cameras[c['camera']] = row
            reason = self._lost_reason(pattern, candidates, depth_absent, errors) if anchor is None else None
            if reason is not None and pattern in self.artwork_reasons:
                reason += f'; artwork match: {self.artwork_reasons[pattern]}'
            support = {'schema': SUPPORT_SCHEMA, 'anchor_camera': anchor['camera'] if anchor else None,
                       'anchors': anchor['anchors'] if anchor else 0,
                       'loo_p95_m': anchor['loo_p95_m'] if anchor else None,
                       'loo_rms_m': anchor['loo_rms_m'] if anchor else None,
                       'model': anchor['model'] if anchor else None,
                       'translation_sigma_m': anchor['translation_sigma_m'] if anchor else None,
                       'rotation_sigma_rad': anchor['rotation_sigma_rad'] if anchor else None,
                       'sigma_method': anchor['sigma_method'] if anchor else None,
                       'corners_m': anchor['corners'].tolist() if anchor else None,
                       'clear_corners_m': anchor['clear_corners'].tolist() if anchor else None,
                       'plane': anchor['plane'] if anchor else None,
                       'anchor_points_m': anchor['anchor_points'].tolist() if anchor else [],
                       'cameras': cameras, 'excluded_anchors': sorted(self.excluded_anchors),
                       'reason': reason, 'motion_authority': False,
                       'method': (('artwork_plane_match' if 'artwork_match' in anchor else 'anchor_fit')
                                  if anchor else None),
                       'artwork_match': anchor.get('artwork_match') if anchor else None,
                       'reference_physical_instance_id': reference.get('physical_instance_id'),
                       'physical_instance_id': anchor['physical_instance_id'] if anchor else None,
                       'physical_instance_identity_verified': bool(
                           anchor and anchor['physical_instance_identity_verified']),
                       'physical_instance_capture_ns': anchor['physical_instance_capture_ns'] if anchor else None,
                       'bundle_id': self.calibration.get('bundle_id')}
            targets.append({'pattern_id': pattern, 'reference_id': reference['reference_id'],
                            'reference_physical_instance_id': reference.get('physical_instance_id'),
                            'physical_instance_id': anchor['physical_instance_id'] if anchor else None,
                            'physical_instance_identity_verified': bool(
                                anchor and anchor['physical_instance_identity_verified']),
                            'physical_instance_capture_ns': anchor['physical_instance_capture_ns'] if anchor else None,
                            'seed': str(reference['seed']), 'target_frame': 'world',
                            'source': 'measured' if anchor else 'lost',
                            'world_from_target': anchor['pose'].tolist() if anchor else None,
                            'translation_sigma_m': anchor['translation_sigma_m'] if anchor else None,
                            'rotation_sigma_rad': anchor['rotation_sigma_rad'] if anchor else None,
                            'capture_ns': anchor['capture_ns'] if anchor else int(time.time_ns()),
                            'support': support})
        known = {target['pattern_id'] for target in targets}
        for path, reason in self.refused_references.items():
            target = refused_target(path, reason, self.calibration.get('bundle_id'))
            if target is not None and target['pattern_id'] not in known:
                known.add(target['pattern_id'])
                targets.append(target)
        return targets

    def request(self, request):
        if request.get('kind') == 'support':
            raise ValueError('stroke support is not an observer product; the observer publishes print poses')
        started = time.perf_counter()
        views, errors = {}, dict(request.get('errors', {}))
        # The service names a wrist view it could not pose (no fresh joints,
        # no registration) in `errors`; that refusal is reported with the
        # views this turn did pose, never as a lost print.
        self.wrist_views = {WRIST_ROLE.replace('-', '_')+source[len(WRIST_ROLE):]: {'refused': reason}
                            for source, reason in errors.items() if source.startswith(WRIST_ROLE)}
        for source, path in request['captures'].items():
            try:
                if source.startswith(WRIST_ROLE):
                    name, view, provenance = wrist_view(path, self.calibration, self.robot, self.poser, source)
                    current = {name: view}
                else:
                    current, provenance = self.capture_reader(path, self.calibration, self.robot), None
                if views.keys() & current.keys():
                    raise ValueError('duplicate camera in display captures')
                views.update(current)
                if provenance is not None:
                    self.wrist_views[name] = provenance
            except (OSError, ValueError, KeyError, TypeError, ImportError, cv2.error) as error:
                errors[source] = str(error)
                if source.startswith(WRIST_ROLE):
                    self.wrist_views[WRIST_ROLE.replace('-', '_')+source[len(WRIST_ROLE):]] = {'refused': str(error)}
        decoded = time.perf_counter()
        result = self.observe(views, errors, time.time_ns())
        result['timings_ms'].update(capture_decode=(decoded-started)*1000,
                                    request_total=(time.perf_counter()-started)*1000)
        return result


def fit_target(camera, report, reference, root_from_depth, capture_ns, *, excluded=None):
    """One camera's measured print pose in root from its anchor report, or
    None where the report supports no fit. The anchors are the measured depth
    samples the camera's tracks selected; the pose is the page's UV-centre
    frame of the ordinary local surface fit; the sigmas are the fit's spatial
    bootstrap for the selected anchor and its leave-one-out residual otherwise.
    """
    if not report.get('candidate_valid') or report.get('calibration_warnings'):
        return None
    anchors = report.get('anchors') or []
    if len(anchors) < MIN_ANCHORS:
        return None
    uv = np.asarray([a['reference_uv'] for a in anchors], float)
    xyz = np.asarray([a['point_camera_m'] for a in anchors], float)
    if (uv.shape != (len(anchors), 2) or xyz.shape != (len(anchors), 3)
            or not np.isfinite(uv).all() or not np.isfinite(xyz).all() or np.any((uv < 0) | (uv > 1))):
        return None
    if len(uv) > FIT_ANCHORS:
        selected = np.linspace(0, len(uv)-1, FIT_ANCHORS).astype(int)
        uv, xyz = uv[selected], xyz[selected]
    model = choose_model(uv, xyz)
    if model is None:
        return None
    pose = center_pose(model['coeff'])
    if pose is None:
        return None
    hull = cv2.convexHull(uv.astype(np.float32))
    enclosed = cv2.pointPolygonTest(hull, (.5, .5), False) >= 0
    matrix = np.asarray(root_from_depth, float)

    def placed(points):
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    root_pose = matrix @ pose
    corners = placed(design(PAGE_CORNERS_UV, model['curved']) @ model['coeff'])
    u0, v0, u1, v1 = reference.get('clear_center_uv', (0., 0., 1., 1.))
    clear = placed(design(np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]), model['curved']) @ model['coeff'])
    return {'camera': camera, 'anchors': int(len(uv)), 'loo_p95_m': float(model['loo_p95_m']),
            'loo_rms_m': float(model['loo_rms_m']),
            'model': 'quadratic_uv_surface' if model['curved'] else 'affine_uv_plane',
            'center_supported': bool(enclosed),
            'anchor_eligible': camera not in (excluded or ()),
            'pose': root_pose, 'corners': corners, 'clear_corners': clear,
            'anchor_points': placed(xyz),
            'plane': {'point_m': root_pose[:3, 3].tolist(), 'normal': root_pose[:3, 2].tolist(),
                      'axes': 'x increasing u, y orthogonalized increasing v, z = x cross y'},
            '_uncertainty_input': (uv, xyz, model, pose, reference),
            'capture_ns': int(capture_ns)}


def target_uncertainty(candidate):
    """Bootstrap only the selected anchor, without changing camera ranking."""
    uv, xyz, model, pose, reference = candidate.pop('_uncertainty_input')
    uncertainty = bootstrap_pose(uv, xyz, model, pose, samples=32)
    if uncertainty.get('available'):
        candidate['translation_sigma_m'] = float(max(uncertainty['translation_std_m']))
        candidate['rotation_sigma_rad'] = float(np.radians(max(uncertainty['rotation_std_deg'])))
        candidate['sigma_method'] = uncertainty['method']
    else:
        half_extent = float(min(reference.get('page_mm', [100., 100.])))/2000.
        candidate['translation_sigma_m'] = float(model['loo_rms_m'])
        candidate['rotation_sigma_rad'] = float(model['loo_rms_m']/max(half_extent, 1e-3))
        candidate['sigma_method'] = 'leave-one-out residual of the anchor fit'


def outline(report, pattern, reference):
    """Boundaries, anchors and centre axes in root from the camera report that
    supplied this pattern's interior points; a refused outline names why."""
    try:
        row = next(row for row in report['stencils'] if row['pattern_id'] == pattern)
        geometry = report_outline(row['surface'], reference, report['root_from_camera'])
        if geometry is None:
            return {'outline_reason': 'no supported surface fit'}
        return geometry
    except (ValueError, KeyError, TypeError, StopIteration, cv2.error) as error:
        return {'outline_reason': (str(error) or type(error).__name__)[:200]}


def snapshot(appearance, tracking_camera):
    batches = [(source, xyz, rgb, pattern) for source, rows in appearance.batches.items()
               for xyz, rgb, pattern in rows if len(xyz)]
    budget = max(1, MAX_POINTS//max(1, len(batches)))
    surfaces = []
    for source, xyz, rgb, pattern in batches:
        stride = max(1, int(np.ceil(len(xyz)/budget)))
        report = appearance.reports[source]
        stamps = [report['capture_timestamp_ns']]
        if tracking_camera in appearance.reports and 'capture_timestamp_ns' in appearance.reports[tracking_camera]:
            stamps.append(appearance.reports[tracking_camera]['capture_timestamp_ns'])
        reference = appearance.bank.references[pattern]
        surfaces.append({'source': source, 'pattern_id': pattern, 'seed': str(reference['seed']),
                         'capture_ns': min(stamps), 'points': xyz[::stride].tolist(), 'colors': rgb[::stride].tolist(),
                         **outline(report, pattern, reference)})
    return {'schema': SCHEMA, 'frame': 'root', 'motion_authority': False,
            'dense_material_identity_verified': False, 'surfaces': surfaces,
            'tracking_camera': tracking_camera, 'cameras': copy.deepcopy(appearance.reports),
            'maximum_age_ns': MAX_AGE_NS, 'fusion': 'measured overlay; no uncertainty reduction'}


def main():
    binding, directory = Path(sys.argv[1]), Path(sys.argv[2])
    worker = LiveSurface(json.loads(read(binding, 8*1024*1024)), directory)
    print(json.dumps({'ready': True}), flush=True)
    while line := sys.stdin.buffer.readline(16*1024+1):
        if len(line) > 16*1024:
            raise ValueError('live display request exceeds size limit')
        try:
            response = worker.request(json.loads(line))
        except (OSError, ValueError, KeyError, TypeError, ImportError, cv2.error) as error:
            response = {'schema': SCHEMA, 'frame': 'root', 'motion_authority': False,
                        'surfaces': [], 'targets': [], 'error': str(error)}
        print(json.dumps(response, allow_nan=False, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
