"""flight_summary.py on a synthetic ring: periods, a latch, the resting carriage effort, the signed tip lag."""
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "flight_summary.py"
spec = importlib.util.spec_from_file_location("flight_summary", SCRIPT)
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)


class Kin:
    """tcp at (q0, q1, q2) m with +z pointing down (toward the paper)."""

    def fk(self, q):
        m = np.diag([1.0, -1.0, -1.0, 1.0])
        m[:3, 3] = q[:3]
        return m


def test_summary_of_a_synthetic_ring(tmp_path):
    n = 400
    r = np.zeros(n, dtype=fs.DTYPE)
    r["t"] = 100.0 + np.arange(n) * 0.0025
    r["t"][200:] += 0.0005  # one 3.0 ms period
    r["phase"] = fs.RUNNING
    r["estop_ok"] = 1
    r["effort"][:, 6] = 1.5
    r["qd"][:150, 0] = 0.2  # moving, then resting
    r["cmd_q"][:, 2] = -0.0003  # the command 0.3 mm below the measured tip: lag toward the paper
    r["latched"][300:], r["latch_reason"][300:], r["estop_ok"][300:] = 1, 1, 0
    path = tmp_path / "right-flight.bin"
    path.write_bytes(f"tatbot-flight 1 right {fs.DTYPE.itemsize}\n".encode() + r.tobytes())
    ring = fs.read(path)
    assert len(ring) == n and fs.DTYPE.itemsize == 188
    out = fs.summarize(ring, Kin())
    assert out["period_max_ms"] == 3.0 and out["jitter_ms"] == 0.5
    (latch,) = out["latches"]
    assert latch["reason"] == 1 and latch["estop_ok"] == 0 and latch["still_after_ms"] == 0.0
    assert out["carriage_effort_n"]["median"] == 1.5
    assert abs(out["tip_lag_mm"]["p50"] - 0.3) < 1e-3
    assert fs.main([str(tmp_path / "absent")]) == 2
