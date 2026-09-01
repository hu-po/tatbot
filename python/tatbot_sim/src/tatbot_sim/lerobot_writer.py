"""Streaming LeRobot v3.0 dataset writer.

Schema is matched field-for-field against a real ``lerobot_robot_tatbot``
recording (local/draw-square, 2026-08-19) so sim and real datasets can be
mixed by a single dataloader during co-training:

- action: fixed_size_list<float>[7], names ``<joint>.pos``
- observation.state: fixed_size_list<float>[14], ``.pos`` then ``.ext_eff``
- data parquet carries no ``task`` string column (only ``task_index``)
- meta/tasks.parquet indexes task_index by the task string, one row per
  distinct task; episodes carry their own task, because a language-conditioned
  policy must not be able to tell sim from real by reading the prompt
- per-episode stats include the q01/q10/q50/q90/q99 quantiles LeRobot expects

Writing streams: frames are encoded into per-episode mp4s as they arrive, so
memory stays flat regardless of dataset size. Encoding runs on a thread pool
(PyAV releases the GIL) because serial encoding of B x 2 cameras per control
step is otherwise the generation bottleneck.

Codec note: sim encodes AV1 with the same settings LeRobot uses for real
recordings (crf 30, preset 12, gop 2). Matching matters for more than tidiness
— ``aggregate_datasets`` requires the two datasets' feature dicts to be equal,
so a different codec makes sim and real impossible to co-train from. It also
gives sim the same compression artifacts as real, which is one less domain gap.

Depth note: with ``depth=True`` each camera gains an
``observation.images.<cam>_depth`` feature, matching what the real follower
records when the D405s run with ``use_depth`` — (H, W, 1) millimetre depth,
log-quantized to 12-bit codes and encoded HEVC gray12le lossless. The
quantization math and encoder settings are ported from LeRobot's
``DepthEncoderConfig`` / ``quantize_depth`` (lerobot is deliberately not a
dependency of this venv), and the feature's info dict mirrors LeRobot's
``get_video_info`` output key for key. If lerobot's depth defaults ever move,
re-verify parity against a fresh real recording before aggregating.
"""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
import pandas as pd

JOINTS = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
]
ACTION_NAMES = [f"{j}.pos" for j in JOINTS]
STATE_NAMES = [f"{j}.pos" for j in JOINTS] + [f"{j}.ext_eff" for j in JOINTS]
QUANTILES = ("q01", "q10", "q50", "q90", "q99")
_Q_LEVELS = (0.01, 0.10, 0.50, 0.90, 0.99)

# LeRobot's depth quantization defaults (configs/video.py, DepthEncoderConfig):
# 12-bit codes over a log-spaced [depth_min, depth_max] metre range. Real
# recordings use exactly these, and dequantization on load reads them back
# from the feature's info dict — so they are part of the schema contract.
DEPTH_QMAX = 4095
DEPTH_MIN_M = 0.01
DEPTH_MAX_M = 10.0
DEPTH_SHIFT_M = 3.5
_DEPTH_LOG_MIN = np.log(np.float32((DEPTH_MIN_M + DEPTH_SHIFT_M) * 1000.0))
_DEPTH_LOG_MAX = np.log(np.float32((DEPTH_MAX_M + DEPTH_SHIFT_M) * 1000.0))


def quantize_depth_codes(depth_mm):
    """Millimetre depth (torch tensor, any device) -> 12-bit codes.

    Same math as LeRobot's quantize_depth; run on the GPU so the encode
    workers receive ready-to-mux codes — per-frame numpy quantization in 64
    worker threads was pure GIL contention against the generation loop."""
    import torch

    norm = (torch.log(depth_mm + DEPTH_SHIFT_M * 1000.0) - float(_DEPTH_LOG_MIN)) / float(
        _DEPTH_LOG_MAX - _DEPTH_LOG_MIN
    )
    return torch.round(norm * DEPTH_QMAX).clamp(0, DEPTH_QMAX)


def dequantize_depth_codes_mm(codes: np.ndarray) -> np.ndarray:
    """12-bit codes -> float32 millimetres (for the stats subsample only)."""
    norm = codes.astype(np.float32) / DEPTH_QMAX
    return np.exp(norm * float(_DEPTH_LOG_MAX - _DEPTH_LOG_MIN) + float(_DEPTH_LOG_MIN)) - (
        DEPTH_SHIFT_M * 1000.0
    )


def _stats_dict(rows: np.ndarray) -> dict:
    """min/max/mean/std/count + quantiles over (N, dim) rows, as plain lists."""
    rows = np.asarray(rows, dtype=np.float64)
    out = {
        "min": rows.min(0).tolist(),
        "max": rows.max(0).tolist(),
        "mean": rows.mean(0).tolist(),
        "std": rows.std(0).tolist(),
        "count": [len(rows)],
    }
    qs = np.quantile(rows, _Q_LEVELS, axis=0)
    for name, q in zip(QUANTILES, qs, strict=True):
        out[name] = q.tolist()
    return out


# LeRobot subsamples images 4x per axis when it computes statistics, so an
# image "count" is frames x (H/4) x (W/4) pixels — and it is a flat 1-element
# list, unlike the per-channel [[[v]]] layout the other fields use. Getting
# this wrong makes datasets load fine but aggregation reject them outright.
STATS_PIXELS_PER_FRAME = (480 // 4) * (640 // 4)


def _img_stats_dict(pixels: np.ndarray, frame_count: int) -> dict:
    """Per-channel image stats in LeRobot's nested [[[v]]] layout."""
    base = _stats_dict(pixels)
    out = {}
    for k, v in base.items():
        if k == "count":
            out[k] = [frame_count * STATS_PIXELS_PER_FRAME]
        else:
            out[k] = [[[float(x)]] for x in v]
    return out


class _Accum:
    """Running moments + a capped reservoir of rows for global quantiles."""

    def __init__(self, dim: int, reservoir: int = 200_000, seed: int = 0):
        self.dim = dim
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf)
        self.max = np.full(dim, -np.inf)
        self.cap = reservoir
        self.buf: list[np.ndarray] = []
        self.buf_n = 0
        self.rng = np.random.default_rng(seed)

    def update(self, rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, self.dim)
        self.n += len(rows)
        self.sum += rows.sum(0)
        self.sumsq += (rows**2).sum(0)
        self.min = np.minimum(self.min, rows.min(0))
        self.max = np.maximum(self.max, rows.max(0))
        if self.buf_n < self.cap:
            take = min(len(rows), self.cap - self.buf_n)
            self.buf.append(rows[:take].copy())
            self.buf_n += take
        elif self.rng.random() < 0.05:  # thin tail sampling once full
            idx = self.rng.integers(0, len(self.buf))
            j = self.rng.integers(0, len(rows))
            self.buf[idx][self.rng.integers(0, len(self.buf[idx]))] = rows[j]

    def result(self) -> dict:
        mean = self.sum / max(self.n, 1)
        var = np.maximum(self.sumsq / max(self.n, 1) - mean**2, 0)
        out = {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [self.n],
        }
        sample = np.concatenate(self.buf) if self.buf else np.zeros((1, self.dim))
        for name, q in zip(QUANTILES, np.quantile(sample, _Q_LEVELS, axis=0), strict=True):
            out[name] = q.tolist()
        return out


class _RgbStream:
    """One RGB mp4 stream: plain codec state, no threading. Streams are
    encoded single-threaded (``threads=1`` where the codec would fan out);
    with dozens of concurrent streams that beats letting each encoder
    instance spawn its own pool, which just oversubscribes the CPU."""

    def __init__(self, path, w: int, h: int, fps: int, codec: str, crf: int, preset: str):
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = w
        self.stream.height = h
        self.stream.pix_fmt = "yuv420p"
        opts = {"crf": str(crf), "preset": str(preset), "g": "2"}
        if codec == "libx264":
            opts["threads"] = "1"
        self.stream.options = opts

    def encode(self, frame: np.ndarray):
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        for packet in self.stream.encode(vf):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


class _DepthStream:
    """HEVC Main 12 gray12le stream for millimetre depth maps.

    Encoder settings mirror LeRobot's ``DepthEncoderConfig`` (hevc, gray12le,
    lossless x265, gop 2), so real and sim depth clips decode identically.
    Frames arrive as (H, W) or (H, W, 1) uint16 12-bit codes (GPU-quantized).
    """

    def __init__(self, path, w: int, h: int, fps: int):
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("hevc", rate=fps)
        self.stream.width = w
        self.stream.height = h
        self.stream.pix_fmt = "gray12le"
        # ultrafast: for LOSSLESS x265 the preset trades encode time for
        # file size only — decoded pixels are bit-identical — and it is 5.6x
        # faster than the default medium (109 vs 19.6 fps/stream measured on
        # the 5900X). Depth encoding is the generation bottleneck, so this is
        # the single biggest throughput lever. Not declared in the feature
        # info dict: the preset field there mirrors DepthEncoderConfig, and
        # decode needs only codec/pix_fmt/quantization, all unchanged.
        self.stream.options = {
            "g": "2", "crf": "30", "x265-params": "lossless=1",
            "preset": "ultrafast", "threads": "1",
        }

    def encode(self, frame: np.ndarray):
        codes = np.squeeze(frame)
        vf = av.VideoFrame.from_ndarray(codes, format="gray12le")
        # from_ndarray does not account for the plane's row padding at every
        # width; rewrite the plane row by row the way LeRobot does.
        h, w = codes.shape
        stride = vf.planes[0].line_size // 2
        dst = np.frombuffer(vf.planes[0], dtype=np.uint16).reshape(h, stride)
        dst[:, :w] = codes
        for packet in self.stream.encode(vf):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


class _ThreadStream:
    """Legacy mode: one dedicated encode thread per stream, so encoding
    overlaps simulation while frames stay strictly ordered (muxing a stream
    from two threads at once would interleave packets and corrupt it)."""

    def __init__(self, stream):
        self._s = stream
        self._ex = ThreadPoolExecutor(max_workers=1)

    def write(self, frame: np.ndarray):
        return self._ex.submit(self._s.encode, frame)

    def close(self):
        fut = self._ex.submit(self._s.close)
        self._ex.shutdown(wait=True)
        return fut


def _encoder_worker(inq, outq):
    """Encoder process main loop: owns a subset of the batch's streams.
    Messages: (op, ...) — "open" creates a stream, "frame" encodes (order
    within a stream is the queue order), "close" flushes and acks, "stop"
    exits. Runs the same codec objects as thread mode, so output files are
    byte-identical either way.

    Exits if the parent disappears: daemon= only covers clean parent exits,
    and a SIGKILLed parent (e.g. the kernel OOM killer) once left three
    orphans holding ~3 GB of codec state each, starving the next run."""
    import os
    import queue as _queue

    parent = os.getppid()
    streams = {}
    while True:
        try:
            msg = inq.get(timeout=10)
        except _queue.Empty:
            if os.getppid() != parent:
                return
            continue
        op = msg[0]
        if op == "open":
            _, sid, kind, path, w, h, fps, codec, crf, preset = msg
            streams[sid] = (_DepthStream(path, w, h, fps) if kind == "depth"
                            else _RgbStream(path, w, h, fps, codec, crf, preset))
        elif op == "frame":
            streams[msg[1]].encode(msg[2])
        elif op == "close":
            streams.pop(msg[1]).close()
            outq.put(("closed", msg[1]))
        elif op == "stop":
            outq.put(("stopped",))
            return


class _EncoderPool:
    """N encoder processes; streams are assigned round-robin at open. Moves
    all codec CPU (and its GIL) out of the generation process — on encode-
    bound hosts the sim loop no longer time-shares with the encoders.
    Backpressure comes from the bounded per-worker queues: when encoders
    lag, submit_frame blocks the sim loop instead of buffering unboundedly.
    """

    def __init__(self, n_procs: int, queue_frames: int = 48):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")  # no fork: parent holds CUDA + av state
        self._inqs = [ctx.Queue(maxsize=queue_frames) for _ in range(n_procs)]
        self._outq = ctx.Queue()
        self._procs = [
            ctx.Process(target=_encoder_worker, args=(q, self._outq), daemon=True)
            for q in self._inqs
        ]
        for p in self._procs:
            p.start()
        self._route: dict[str, int] = {}
        self._rr = 0
        self._closed_acks: set[str] = set()

    def _put(self, worker: int, msg):
        """Bounded put that raises instead of hanging if the worker died —
        a dead consumer once left the generation loop blocked overnight."""
        import queue as _queue

        while True:
            try:
                self._inqs[worker].put(msg, timeout=30)
                return
            except _queue.Full:
                if not self._procs[worker].is_alive():
                    raise RuntimeError(
                        "encoder worker died (traceback above) — aborting rather "
                        "than blocking; check RAM: codec state is ~100 MB/stream"
                    ) from None

    def open_stream(self, sid: str, kind: str, path, w, h, fps,
                    codec="", crf=0, preset=""):
        self._route[sid] = self._rr % len(self._inqs)
        self._rr += 1
        self._put(self._route[sid],
                  ("open", sid, kind, str(path), w, h, fps, codec, crf, preset))

    def submit_frame(self, sid: str, frame: np.ndarray):
        self._put(self._route[sid], ("frame", sid, frame))

    def close_stream(self, sid: str):
        self._put(self._route[sid], ("close", sid))

    def wait_closed(self, sid: str):
        import queue as _queue

        while sid not in self._closed_acks:
            try:
                msg = self._outq.get(timeout=30)
            except _queue.Empty:
                if not all(p.is_alive() for p in self._procs):
                    raise RuntimeError("encoder worker died while flushing streams") from None
                continue
            if msg[0] == "closed":
                self._closed_acks.add(msg[1])
        self._closed_acks.discard(sid)
        self._route.pop(sid, None)

    def shutdown(self):
        for q in self._inqs:
            q.put(("stop",))
        for p in self._procs:
            p.join(timeout=30)


class _PoolStream:
    """Stream handle backed by the encoder pool; mirrors _ThreadStream's API
    (write returns nothing to wait on — ordering and backpressure live in
    the pool queues; close returns an ack to .result() on)."""

    def __init__(self, pool: _EncoderPool, sid: str):
        self._pool = pool
        self.sid = sid

    def write(self, frame: np.ndarray):
        self._pool.submit_frame(self.sid, frame)
        return None

    def close(self):
        self._pool.close_stream(self.sid)
        return self

    def result(self):  # ack object protocol, matched with _ThreadStream futures
        self._pool.wait_closed(self.sid)


@dataclass
class _Episode:
    index: int
    task: str
    actions: list = field(default_factory=list)
    states: list = field(default_factory=list)
    videos: dict = field(default_factory=dict)  # RGB and depth streams, keyed by feature cam name
    pixels: dict = field(default_factory=dict)
    frames: int = 0


class LeRobotWriter:
    def __init__(
        self,
        out_dir: str,
        fps: int = 30,
        cameras: tuple[str, ...] = ("wrist_upper", "wrist_lower"),
        depth: bool = False,
        image_size: tuple[int, int] = (640, 480),  # (W, H)
        task_name: str = "Draw the shape on the skin pad.",
        robot_type: str = "tatbot_follower",
        chunks_size: int = 1000,
        stats_stride: int = 5,
        codec: str = "libsvtav1",
        crf: int = 30,
        preset: str = "12",
        max_pending: int = 3,
        encoder_procs: int = 0,
    ):
        """``encoder_procs`` > 0 encodes video in that many separate
        processes instead of in-process threads — identical output bytes,
        but the codec CPU (and its GIL) leaves the generation process. Use
        on encode-bound hosts (few cores); 0 keeps the thread mode."""
        self.base = Path(out_dir)
        if self.base.exists() and any(self.base.iterdir()):
            raise FileExistsError(f"output dir {self.base} is not empty")
        self.fps = fps
        self.cameras = cameras
        self.w, self.h = image_size
        self.task_name = task_name
        # task string -> index, assigned in first-seen order
        self._tasks: dict[str, int] = {}
        self.robot_type = robot_type
        self.chunks_size = chunks_size
        self.stats_stride = stats_stride
        self.codec = codec
        self.crf = crf
        self.preset = preset

        self.action_dim = len(ACTION_NAMES)
        self.state_dim = len(STATE_NAMES)
        self.episodes_meta: list[dict] = []
        self.frames_per_episode: list[int] = []
        # Episode rows are flushed per chunk instead of retained to the end:
        # holding every DataFrame cost ~4 MB/episode, which does not survive a
        # 10k-episode run.
        self._chunk_dfs: list[pd.DataFrame] = []
        self._chunk_idx = 0
        self._scalar_acc = {
            f: _Accum(1) for f in ("timestamp", "frame_index", "episode_index", "index", "task_index")
        }
        # Depth cameras appear as "<cam>_depth", the name the real follower
        # gives them when the D405s record with use_depth.
        self.depth_cams = tuple(f"{c}_depth" for c in self.cameras) if depth else ()
        self.g_action = _Accum(self.action_dim)
        self.g_state = _Accum(self.state_dim)
        self.g_img = {c: _Accum(3) for c in self.cameras}
        self.g_img.update({c: _Accum(1) for c in self.depth_cams})
        self.g_img_frames = dict.fromkeys(self.cameras + self.depth_cams, 0)
        self._open: list[_Episode] = []
        self._pending: deque[list] = deque()
        self.max_pending = max_pending
        self.pool = _EncoderPool(encoder_procs) if encoder_procs > 0 else None

        (self.base / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    @property
    def num_episodes(self) -> int:
        return len(self.frames_per_episode)

    # -- batch-parallel episode API -------------------------------------------------

    def open_batch(self, batch_size: int, tasks: list[str] | None = None) -> list[_Episode]:
        """Open ``batch_size`` episodes recorded in lockstep (one per parallel env).

        ``tasks`` gives each episode its own task string; without it they all
        share the writer's default.
        """
        assert not self._open, "previous batch not closed"
        base_idx = self.num_episodes
        eps = []
        for i in range(batch_size):
            ep = _Episode(index=base_idx + i, task=(tasks[i] if tasks else self.task_name))
            chunk = ep.index // self.chunks_size
            for cam in self.cameras:
                vdir = self.base / "videos" / f"observation.images.{cam}" / f"chunk-{chunk:03d}"
                vdir.mkdir(parents=True, exist_ok=True)
                path = vdir / f"file-{ep.index:03d}.mp4"
                if self.pool:
                    sid = f"{ep.index}:{cam}"
                    self.pool.open_stream(sid, "rgb", path, self.w, self.h, self.fps,
                                          self.codec, self.crf, self.preset)
                    ep.videos[cam] = _PoolStream(self.pool, sid)
                else:
                    ep.videos[cam] = _ThreadStream(_RgbStream(
                        path, self.w, self.h, self.fps, self.codec, self.crf, self.preset))
                ep.pixels[cam] = []
            for cam in self.depth_cams:
                vdir = self.base / "videos" / f"observation.images.{cam}" / f"chunk-{chunk:03d}"
                vdir.mkdir(parents=True, exist_ok=True)
                path = vdir / f"file-{ep.index:03d}.mp4"
                if self.pool:
                    sid = f"{ep.index}:{cam}"
                    self.pool.open_stream(sid, "depth", path, self.w, self.h, self.fps)
                    ep.videos[cam] = _PoolStream(self.pool, sid)
                else:
                    ep.videos[cam] = _ThreadStream(_DepthStream(path, self.w, self.h, self.fps))
                ep.pixels[cam] = []
            eps.append(ep)
        self._open = eps
        return eps

    def add_steps(
        self,
        actions: np.ndarray,
        states: np.ndarray,
        frames: dict[str, np.ndarray],
        depth: dict[str, np.ndarray] | None = None,
        active: list[bool] | None = None,
    ):
        """Add one control step for every open episode.

        actions: (B, action_dim); states: (B, state_dim);
        frames[cam]: (B, H, W, 3) uint8 RGB;
        depth[cam]: (B, H, W, 1) uint16 12-BIT CODES (see
        quantize_depth_codes; code 0 = no measurement).
        ``active[i]`` False skips episode i this step — its recording has
        ended (the drawing is done) while the batch holds for its longest
        episode; the episode simply closes shorter at close_batch.
        """
        assert (depth is not None) == bool(self.depth_cams), "depth frames must match writer config"
        jobs = []
        for i, ep in enumerate(self._open):
            if active is not None and not active[i]:
                continue
            ep.actions.append(actions[i].astype(np.float32))
            ep.states.append(states[i].astype(np.float32))
            ep.frames += 1
            do_stats = ep.frames % self.stats_stride == 0
            for cam in self.cameras:
                frame = frames[cam][i]
                jobs.append(ep.videos[cam].write(frame))
                if do_stats:
                    ep.pixels[cam].append(frame[::8, ::8].reshape(-1, 3).astype(np.float32) / 255.0)
            if self.depth_cams and depth is not None:
                for src, cam in zip(self.cameras, self.depth_cams, strict=True):
                    frame = depth[src][i]
                    jobs.append(ep.videos[cam].write(frame))
                    if do_stats:
                        # depth stats stay in stored units (mm) — LeRobot only
                        # rescales RGB by 255, never depth maps. Frames arrive as
                        # codes, so dequantize just this small subsample.
                        ep.pixels[cam].append(
                            dequantize_depth_codes_mm(frame[::8, ::8]).reshape(-1, 1)
                        )
        # Let encoding run behind simulation, but only by max_pending steps so
        # queued frames cannot grow without bound. (Pool streams return None —
        # their backpressure is the bounded worker queues.)
        self._pending.append(jobs)
        while len(self._pending) > self.max_pending:
            for j in self._pending.popleft():
                if j is not None:
                    j.result()

    def close_batch(self):
        while self._pending:
            for j in self._pending.popleft():
                if j is not None:
                    j.result()
        for fut in [v.close() for ep in self._open for v in ep.videos.values()]:
            fut.result()
        for ep in self._open:
            self._finish_episode(ep)
        self._open = []

    # -- internal -------------------------------------------------------------------

    def _task_index(self, task: str) -> int:
        return self._tasks.setdefault(task, len(self._tasks))

    def _finish_episode(self, ep: _Episode):
        actions = np.stack(ep.actions)
        states = np.stack(ep.states)
        n = len(actions)
        ts = np.arange(n, dtype=np.float32) / self.fps
        prev_frames = int(sum(self.frames_per_episode))

        df = pd.DataFrame(
            {
                "action": [r.tolist() for r in actions],
                "observation.state": [r.tolist() for r in states],
                "timestamp": ts,
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, ep.index, dtype=np.int64),
                "index": np.arange(prev_frames, prev_frames + n, dtype=np.int64),
                "task_index": np.full(n, self._task_index(ep.task), dtype=np.int64),
            }
        )
        self._chunk_dfs.append(df)
        for f, acc in self._scalar_acc.items():
            acc.update(df[f].values.reshape(-1, 1))
        self.g_action.update(actions)
        self.g_state.update(states)

        chunk = ep.index // self.chunks_size
        meta = {
            "episode_index": ep.index,
            "tasks": [ep.task],
            "length": n,
            "data/chunk_index": chunk,
            "data/file_index": 0,
            "dataset_from_index": prev_frames,
            "dataset_to_index": prev_frames + n,
        }
        for cam in self.cameras + self.depth_cams:
            prefix = f"videos/observation.images.{cam}"
            meta[f"{prefix}/chunk_index"] = chunk
            meta[f"{prefix}/file_index"] = ep.index
            meta[f"{prefix}/from_timestamp"] = float(ts[0])
            meta[f"{prefix}/to_timestamp"] = float(ts[-1])
        meta["meta/episodes/chunk_index"] = 0
        meta["meta/episodes/file_index"] = 0

        for name, rows in (("action", actions), ("observation.state", states)):
            for k, v in _stats_dict(rows).items():
                meta[f"stats/{name}/{k}"] = v
        for cam in self.cameras + self.depth_cams:
            dim = self.g_img[cam].dim
            px = (
                np.concatenate(ep.pixels[cam])
                if ep.pixels[cam]
                else np.zeros((1, dim), dtype=np.float32)
            )
            self.g_img[cam].update(px)
            self.g_img_frames[cam] += n
            for k, v in _img_stats_dict(px, n).items():
                meta[f"stats/observation.images.{cam}/{k}"] = v
        for f in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            for k, v in _stats_dict(df[f].values.reshape(-1, 1)).items():
                meta[f"stats/{f}/{k}"] = v

        self.episodes_meta.append(meta)
        self.frames_per_episode.append(n)
        if len(self._chunk_dfs) >= self.chunks_size:
            self._flush_chunk()

    def _flush_chunk(self):
        """Write buffered episode rows as one data chunk and release them."""
        if not self._chunk_dfs:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema(
            [
                pa.field("action", pa.list_(pa.float32(), self.action_dim)),
                pa.field("observation.state", pa.list_(pa.float32(), self.state_dim)),
                pa.field("timestamp", pa.float32()),
                pa.field("frame_index", pa.int64()),
                pa.field("episode_index", pa.int64()),
                pa.field("index", pa.int64()),
                pa.field("task_index", pa.int64()),
            ]
        )
        cdir = self.base / "data" / f"chunk-{self._chunk_idx:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(self._chunk_dfs, ignore_index=True)
        pq.write_table(
            pa.Table.from_pandas(combined, schema=schema, preserve_index=False),
            cdir / "file-000.parquet",
        )
        self._chunk_dfs = []
        self._chunk_idx += 1

    # -- finalization ---------------------------------------------------------------

    def finalize(self):
        assert not self._open, "close the open batch before finalizing"
        if self.pool:
            self.pool.shutdown()
        self._flush_chunk()
        num_eps = self.num_episodes

        pd.DataFrame(self.episodes_meta).to_parquet(
            self.base / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
        )
        # LeRobot's load_tasks() reads this frame and names its INDEX "task",
        # then looks tasks up with .loc[task_string] — so the task text must be
        # the index, not a column, or batches surface task as an int.
        tasks = self._tasks or {self.task_name: 0}
        names = sorted(tasks, key=lambda k: tasks[k])
        pd.DataFrame(
            {"task_index": [tasks[t] for t in names]}, index=pd.Index(names, name="task")
        ).to_parquet(self.base / "meta" / "tasks.parquet")

        stats = {"action": self.g_action.result(), "observation.state": self.g_state.result()}
        for cam in self.cameras + self.depth_cams:
            g = self.g_img[cam]
            sample = np.concatenate(g.buf) if g.buf else np.zeros((1, g.dim))
            stats[f"observation.images.{cam}"] = _img_stats_dict(sample, self.g_img_frames[cam])
        for f, acc in self._scalar_acc.items():
            stats[f] = acc.result()
        with open(self.base / "meta" / "stats.json", "w") as fh:
            json.dump(stats, fh, indent=2)

        codec_name = {"libsvtav1": "av1", "libx264": "h264", "h264_nvenc": "h264"}.get(
            self.codec, self.codec
        )
        # Mirrors what LeRobot writes for real recordings, key for key: the
        # aggregate step compares these dicts for equality.
        video_info = {
            "is_depth_map": False,
            "video.height": self.h,
            "video.width": self.w,
            "video.codec": codec_name,
            "video.pix_fmt": "yuv420p",
            "video.fps": self.fps,
            "video.channels": 3,
            "has_audio": False,
            "video.g": 2,
            "video.crf": self.crf,
            "video.preset": int(self.preset) if str(self.preset).isdigit() else self.preset,
            "video.fast_decode": 0,
            "video.video_backend": "pyav",
            "video.extra_options": {},
        }
        features = {
            "action": {"dtype": "float32", "shape": [self.action_dim], "names": ACTION_NAMES},
            "observation.state": {
                "dtype": "float32",
                "shape": [self.state_dim],
                "names": STATE_NAMES,
            },
        }
        for cam in self.cameras:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [self.h, self.w, 3],
                "names": ["height", "width", "channels"],
                "info": dict(video_info),
            }
        # Depth feature info mirrors LeRobot's get_video_info() on a stream
        # encoded by DepthEncoderConfig: stream facts, then every config field
        # as video.<name> (vcodec excluded, preset genuinely None), then the
        # depth-map flag. The quantization fields are read back on load.
        depth_info = {
            "video.height": self.h,
            "video.width": self.w,
            "video.codec": "hevc",
            "video.pix_fmt": "gray12le",
            "video.fps": self.fps,
            "video.channels": 1,
            "has_audio": False,
            "video.g": 2,
            "video.crf": 30,
            "video.preset": None,
            "video.fast_decode": 0,
            "video.video_backend": "pyav",
            "video.extra_options": {"x265-params": "lossless=1"},
            "video.depth_min": DEPTH_MIN_M,
            "video.depth_max": DEPTH_MAX_M,
            "video.shift": DEPTH_SHIFT_M,
            "video.use_log": True,
            "is_depth_map": True,
            # the recorder infers this from the first frame's dtype (uint16 -> mm)
            "depth_unit": "mm",
        }
        for cam in self.depth_cams:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [self.h, self.w, 1],
                "names": ["height", "width", "channels"],
                "info": dict(depth_info),
            }
        for f, dt in (
            ("timestamp", "float32"),
            ("frame_index", "int64"),
            ("episode_index", "int64"),
            ("index", "int64"),
            ("task_index", "int64"),
        ):
            features[f] = {"dtype": dt, "shape": [1], "names": None}

        # LeRobot rejects a zero size and ignores unknown keys, so report
        # ceil-MB and mirror exactly the key set a real recording carries.
        data_size = sum(f.stat().st_size for f in (self.base / "data").rglob("*.parquet"))
        video_size = sum(f.stat().st_size for f in (self.base / "videos").rglob("*.mp4"))
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "total_episodes": num_eps,
            "total_frames": int(sum(self.frames_per_episode)),
            "total_tasks": max(len(self._tasks), 1),
            "chunks_size": self.chunks_size,
            "fps": self.fps,
            "data_files_size_in_mb": max(1, -(-data_size // 2**20)),
            "video_files_size_in_mb": max(1, -(-video_size // 2**20)),
            "splits": {"train": f"0:{num_eps}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }
        with open(self.base / "meta" / "info.json", "w") as fh:
            json.dump(info, fh, indent=2)
