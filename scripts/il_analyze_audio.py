#!/usr/bin/env python3
"""Score a rollout's contact-mic recording: when did the pen actually touch?

    il_analyze_audio.py <run-dir | audio.wav> [--settle S] [--json]
    il_analyze_audio.py bench.wav --label hover:0-10,draw:10-20
    il_analyze_audio.py --compare <audio_analysis.json>...

Hover and drawing sound different — pen-motor load shift plus tip-on-paper
friction — and the microphone attached to the new EE hears it through the
structure. Unlike the FK proxy in il_analyze_rollout.py this needs no plane
calibration, so it survives the base/table/pad drift that made "in-band 100%"
runs never mark the paper (2026-08-24). Where analysis.json exists the two
are cross-checked frame-by-frame.

The contact threshold is fit PER RUN on the mono EE channel (Otsu on log
friction-band energy) — never carried across sessions: the acoustic signature
moves with the EE mount and the interface gain. A run whose two classes are not
separated (< MIN_SEPARATION_DB) reports threshold.valid=false and its contact
numbers are flagged, mirroring the contact_basis honesty of the geometric
analyzer. Alignment: audio_start.json unix_seconds + frame offset against the
flight CSV's t_wall column (same host, same clock).

Never imports lerobot/torch, and numpy is the only non-stdlib dependency —
this runs in the post-rollout hook while the operator is waiting.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

try:
    import numpy as np
except ImportError:  # a bare system python3 has no numpy; uv always can get it.
    sys.exit("il_analyze_audio needs numpy — run:\n"
             "  uv run --no-project --with numpy python scripts/il_analyze_audio.py "
             + " ".join(sys.argv[1:]))

# --- DSP -------------------------------------------------------------------
WIN = 1024               # 21 ms @ 48 kHz — several pen-motor cycles per frame
HOP = 256                # 5.3 ms frame pitch
FRICTION_BAND_HZ = (2000.0, 8000.0)  # tip-on-paper broadband; above motor harmonics
MIN_SEG_S = 0.10         # shorter than this is a tap/transient, not drawing
MERGE_GAP_S = 0.15       # micro-dropouts inside one stroke are still one stroke
MIN_SEPARATION_DB = 6.0  # Otsu classes closer than this = no contact evidence
SETTLE_S = 8.0           # same fixed startup-traverse window as il_analyze_rollout

STATUS_MARK = {"ok": "  ok ", "warn": "WARN ", "fail": "FAIL "}


def load_wav(path: Path):
    """(fs, float array [n, 1] in [-1, 1]) — mono 16-bit PCM only."""
    with wave.open(str(path), "rb") as w:
        ch, width, fs, n = (w.getnchannels(), w.getsampwidth(),
                            w.getframerate(), w.getnframes())
        if ch != 1:
            raise SystemExit(f"{path}: expected mono EE audio, got {ch} channels")
        if width != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM, got {8 * width}-bit "
                             "(capture uses arecord -f S16_LE)")
        raw = np.frombuffer(w.readframes(n), dtype="<i2")
    return fs, (raw.reshape(-1, 1).astype(np.float64) / 32768.0)


def frame_features(x: np.ndarray, fs: float) -> dict:
    """Per-frame features from the mono EE channel. x is [n, 1] float.

    The singleton channel axis is retained in the output for compatibility
    with existing ``tatbot.rollout.audio/1`` reports.
    """
    n = x.shape[0]
    if n < WIN:
        raise SystemExit(f"recording too short: {n / fs:.2f}s < one {WIN / fs:.3f}s window")
    n_frames = 1 + (n - WIN) // HOP
    window = np.hanning(WIN)
    freqs = np.fft.rfftfreq(WIN, 1.0 / fs)
    band = (freqs >= FRICTION_BAND_HZ[0]) & (freqs <= FRICTION_BAND_HZ[1])
    frames = np.lib.stride_tricks.sliding_window_view(x[:, 0], WIN)[::HOP][:n_frames]
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    friction = 10.0 * np.log10(spec[:, band].sum(axis=1) + 1e-12)[:, None]
    rms = 20.0 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-12)[:, None]
    t_rel = (np.arange(n_frames) * HOP + WIN / 2) / fs
    clip_pct = [round(100.0 * float((np.abs(x[:, 0]) >= 32767 / 32768).mean()), 3)]
    return {"t_rel": t_rel, "friction_db": friction, "rms_db": rms, "clip_pct": clip_pct}


def otsu_split(values_db: np.ndarray):
    """(threshold_db, separation_db) — Otsu on a 1-D dB feature, per run.

    separation_db is the distance between the two class means; when the
    distribution is unimodal (all-hover, all-draw, or mic dead) Otsu still
    returns a cut, so the caller must gate on separation before believing it.
    """
    v = values_db[np.isfinite(values_db)]
    if len(v) < 32 or float(v.max() - v.min()) < 1e-6:
        return float("nan"), 0.0
    hist, edges = np.histogram(v, bins=128)
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(np.float64)
    p = w / w.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / np.where(denom > 0, denom, np.nan)
    k = int(np.nanargmax(sigma_b))
    thr = float(centers[k])
    lo, hi = v[v <= thr], v[v > thr]
    if not len(lo) or not len(hi):
        return thr, 0.0
    return thr, float(hi.mean() - lo.mean())


def find_segments(mask: np.ndarray, t_rel: np.ndarray) -> list[tuple[float, float]]:
    """Contact segments (start_s, end_s): gaps <= MERGE_GAP_S merged, then
    segments < MIN_SEG_S dropped."""
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    spans = [(float(t_rel[a]), float(t_rel[min(b, len(t_rel)) - 1]))
             for a, b in zip(idx[::2], idx[1::2], strict=True)]
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= MIN_SEG_S]


def geometric_mask(run_dir: Path, t_unix: np.ndarray):
    """Per-audio-frame geometric contact (bool) from the flight CSV, or None.

    analysis.json stores only aggregates, so the per-tick mask is recomputed
    with il_analyze_rollout's own loaders — but the PLANE is taken from
    analysis.json when present, so both reports argue about the same plane.
    """
    csvs = sorted(run_dir.glob("flight-*.csv"))
    if not csvs:
        return None, None
    import il_analyze_rollout as ana  # noqa: PLC0415 — lazy: pulls in the URDF chain

    rows = ana.load_rows(csvs[-1])
    if len(rows) < 30:
        return None, None
    t_wall = np.array([float(r["t_wall"]) for r in rows])

    tip_offset, plane = None, None
    try:
        geom = json.loads((run_dir / "analysis.json").read_text())["geometry"]
        plane = float(geom["plane_z_mm"])
        if geom.get("pen_tip_offset_mm") is not None:
            tip_offset = [v / 1000.0 for v in geom["pen_tip_offset_mm"]]
    except Exception:
        pass

    chain = ana.UrdfChain(str(REPO / "urdf" / "tatbot.urdf"))
    names = chain.arm_joint_names("right")
    z = ana.pen_path(rows, chain, names, tip_offset)[:, 2]
    if plane is None:
        plane = float(np.percentile(z, ana.INFERRED_PLANE_PCT))
    tick_mask = (z <= plane + ana.CONTACT_TOL_MM).astype(np.float64)
    frame_mask = np.interp(t_unix, t_wall, tick_mask) > 0.5
    in_flight = (t_unix >= t_wall[0]) & (t_unix <= t_wall[-1])
    return frame_mask, in_flight


def cross_check(audio_mask: np.ndarray, geo_mask: np.ndarray,
                t_rel: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Frame-by-frame agreement between the two contact opinions.

    audio_only = the mic hears drawing the FK plane calls airborne (plane too
    low / drifted); geo_only = FK says in-band but nothing is heard — exactly
    the "in-band 100%, never marked" failure this feature exists to catch.
    """
    keep = valid if valid is not None else np.ones(len(t_rel), dtype=bool)
    a, g = audio_mask[keep], geo_mask[keep]
    n = int(keep.sum())
    if n == 0:
        return {"available": False}
    disagree = a != g
    windows = [{"t_start_s": round(s, 2), "t_end_s": round(e, 2),
                "who": "audio_only" if a[np.searchsorted(t_rel[keep], s)] else "geo_only"}
               for s, e in find_segments(disagree, t_rel[keep])]
    windows.sort(key=lambda w: w["t_end_s"] - w["t_start_s"], reverse=True)
    return {
        "available": True,
        "frames_compared": n,
        "agreement_pct": round(100.0 * float((a == g).mean()), 1),
        "audio_only_pct": round(100.0 * float((a & ~g).mean()), 1),
        "geo_only_pct": round(100.0 * float((~a & g).mean()), 1),
        "audio_contact_pct": round(100.0 * float(a.mean()), 1),
        "geometric_contact_pct": round(100.0 * float(g.mean()), 1),
        "disagreement_windows": windows[:10],
    }


def parse_labels(spec: str) -> list[tuple[str, float, float]]:
    """'hover:0-10,draw:10-20' -> [(label, start_s, end_s), ...]"""
    out = []
    for part in spec.split(","):
        name, _, span = part.partition(":")
        a, _, b = span.partition("-")
        out.append((name.strip(), float(a), float(b)))
    return out


def score_labels(feat: dict, mask: np.ndarray, thr_db: list[float],
                 labels: list[tuple[str, float, float]]) -> dict:
    """Bench separability: operator hovered/drew in known spans — how cleanly
    do the features split, and does the fitted threshold agree with the labels?
    Drawing-ish label names (draw/contact/touch/ink) count as contact=True."""
    t = feat["t_rel"]
    fr = feat["friction_db"]
    per_label, truth, pred = {}, [], []
    for name, a, b in labels:
        sel = (t >= a) & (t < b)
        if not sel.any():
            continue
        is_contact = any(k in name.lower() for k in ("draw", "contact", "touch", "ink"))
        per_label[name] = {
            "span_s": [a, b],
            "contact": is_contact,
            "friction_db_mean": [round(float(fr[sel, 0].mean()), 1)],
            "friction_db_std": [round(float(fr[sel, 0].std()), 1)],
            "classified_contact_pct": round(100.0 * float(mask[sel].mean()), 1),
        }
        truth.append(np.full(int(sel.sum()), is_contact))
        pred.append(mask[sel])
    if not per_label:
        return {"labels": {}, "accuracy_pct": None, "d_prime": None}
    truth, pred = np.concatenate(truth), np.concatenate(pred)
    d_prime = None
    hov = [v for v in per_label.values() if not v["contact"]]
    drw = [v for v in per_label.values() if v["contact"]]
    if hov and drw:  # d' on the EE channel's band energy between the two classes
        mu_h, sd_h = hov[0]["friction_db_mean"][0], max(hov[0]["friction_db_std"][0], 1e-6)
        mu_d, sd_d = drw[0]["friction_db_mean"][0], max(drw[0]["friction_db_std"][0], 1e-6)
        d_prime = round(abs(mu_d - mu_h) / np.sqrt(0.5 * (sd_h**2 + sd_d**2)), 2)
    return {"labels": per_label,
            "accuracy_pct": round(100.0 * float((truth == pred).mean()), 1),
            "d_prime": d_prime,
            "threshold_db": [round(v, 1) for v in thr_db]}


def analyze(target: Path, settle: float = SETTLE_S, label_spec: str | None = None) -> dict:
    run_dir = target if target.is_dir() else target.parent
    wav_path = (target if target.is_file() else run_dir / "audio.wav")
    if not wav_path.is_file():
        raise SystemExit(f"no audio.wav in {run_dir}")

    start_unix = None
    for cand in (wav_path.with_name(wav_path.stem + "_start.json"),
                 run_dir / "audio_start.json"):
        if cand.is_file():
            try:
                start_unix = float(json.loads(cand.read_text())["unix_seconds"])
                break
            except Exception:
                pass
    if start_unix is None:
        print(f"WARN: no audio_start.json beside {wav_path.name} — "
              "unix alignment unavailable, relative times only", file=sys.stderr)

    meta = {}
    if (run_dir / "meta.json").is_file():
        with __import__("contextlib").suppress(Exception):
            meta = json.loads((run_dir / "meta.json").read_text())

    fs, x = load_wav(wav_path)
    feat = frame_features(x, fs)
    t_rel = feat["t_rel"]

    # Threshold on the post-settle window — the startup traverse is transit
    # noise in the geometric analyzer and cable/servo rustle here.
    scored = t_rel >= (0.0 if label_spec else settle)
    threshold_db, separation_db = otsu_split(feat["friction_db"][scored, 0])
    thr = [threshold_db]
    sep = [separation_db]
    ch_valid = [bool(separation_db >= MIN_SEPARATION_DB)]
    valid = ch_valid[0]

    # Silence is not evidence of hover. With no believable split the mask is
    # empty and metrics are flagged rather than invented.
    mask = (feat["friction_db"][:, 0] > threshold_db
            if valid else np.zeros(len(t_rel), dtype=bool))
    mask &= scored

    segments = find_segments(mask, t_rel)
    seg_mask = np.zeros(len(t_rel), dtype=bool)  # post merge/min-length pruning
    for a, b in segments:
        seg_mask |= (t_rel >= a) & (t_rel <= b)

    scored_s = float(t_rel[scored][-1] - t_rel[scored][0]) if scored.any() else 0.0
    contact_s = sum(b - a for a, b in segments)
    hov_db = [float(np.median(feat["friction_db"][scored & ~seg_mask, 0]))
              if (scored & ~seg_mask).any() else float("nan")]
    con_db = [float(np.median(feat["friction_db"][seg_mask, 0]))
              if seg_mask.any() else float("nan")]

    def seg_json(a, b):
        sel = (t_rel >= a) & (t_rel <= b)
        # unix = start + the ROUNDED relative time, so the two fields never
        # disagree — downstream joins use unix, humans read the relative one.
        return {
            "t_start_s": round(a, 2), "t_end_s": round(b, 2),
            "t_start_unix": round(start_unix + round(a, 2), 3) if start_unix else None,
            "t_end_unix": round(start_unix + round(b, 2), 3) if start_unix else None,
            "peak_db": round(float(feat["friction_db"][sel].max()), 1) if sel.any() else None,
        }

    xc = {"available": False}
    in_a_run = target.is_dir() or (run_dir / "meta.json").is_file()
    if in_a_run and start_unix is not None:
        geo, in_flight = geometric_mask(run_dir, start_unix + t_rel)
        if geo is not None:
            xc = cross_check(seg_mask, geo, t_rel, valid=in_flight & scored)
            with __import__("contextlib").suppress(Exception):
                a_json = json.loads((run_dir / "analysis.json").read_text())
                xc["analysis_contact_pct"] = a_json["metrics"]["contact_pct"]
                xc["geometric_valid"] = a_json["geometry"]["valid"]

    label_block = None
    if label_spec:
        label_block = score_labels(feat, seg_mask, thr, parse_labels(label_spec))

    m = {
        "contact_pct_audio": round(100.0 * contact_s / scored_s, 1) if scored_s else 0.0,
        "contact_s_total": round(contact_s, 2),
        "longest_contact_s": round(max((b - a for a, b in segments), default=0.0), 2),
        "n_contact_segments": len(segments),
        "hover_level_db": [round(v, 1) for v in hov_db],
        "contact_level_db": [round(v, 1) for v in con_db],
    }

    out = {
        "schema": "tatbot.rollout.audio/1",
        "run_id": meta.get("run_id") or (run_dir.name if target.is_dir() else None),
        "wav": str(wav_path),
        "git_sha": meta.get("git", {}).get("short"),
        "audio": {"fs": fs, "channels": 1,
                  "duration_s": round(float(len(x) / fs), 2),
                  "start_unix": start_unix, "clipped_pct": feat["clip_pct"]},
        "dsp": {"win": WIN, "hop": HOP, "friction_band_hz": list(FRICTION_BAND_HZ),
                "settle_s": 0.0 if label_spec else settle,
                "min_seg_s": MIN_SEG_S, "merge_gap_s": MERGE_GAP_S},
        "threshold": {"method": "otsu_log_friction_energy",
                      "per_channel_db": [round(v, 1) if np.isfinite(v) else None for v in thr],
                      "separation_db": [round(v, 1) for v in sep],
                      "channel_valid": ch_valid,
                      "min_separation_db": MIN_SEPARATION_DB,
                      "valid": valid},
        "metrics": m,
        "segments": [seg_json(a, b) for a, b in segments],
        "cross_check": xc,
        "labels": label_block,
    }
    out["checks"] = build_checks(out)
    return out


def build_checks(a: dict) -> list[dict]:
    def check(name, value, expected, status, note=""):
        return {"name": name, "value": value, "expected": expected,
                "status": status, "note": note}

    checks = []
    rms_repr = a["metrics"]["hover_level_db"]
    quiet = all(not np.isfinite(v) or v < -85 for v in rms_repr)
    checks.append(check(
        "signal_present", rms_repr, "> -85 dB band energy",
        "warn" if quiet else "ok",
        "hover-level friction-band energy is at the EE-mic noise floor — "
        "check piezo wiring and interface gain (probe: "
        "scripts/il_audio_record.sh probe)." if quiet else ""))

    clip = max(a["audio"]["clipped_pct"])
    checks.append(check(
        "clipping", clip, "<1%", "warn" if clip >= 1.0 else "ok",
        "clipped samples flatten the friction band and drag the threshold — "
        "lower the interface input gain." if clip >= 1.0 else ""))

    t = a["threshold"]
    checks.append(check(
        "separation", t["separation_db"], f">={MIN_SEPARATION_DB} dB on the EE channel",
        "ok" if t["valid"] else "warn",
        "" if t["valid"] else
        "the EE channel does not show two acoustic classes — either the pen never touched, "
        "never lifted, or the microphone is not coupled. Contact metrics are "
        "reported but NOT evidence either way."))

    xc = a["cross_check"]
    if xc.get("available") and t["valid"]:
        agree = xc["agreement_pct"]
        checks.append(check(
            "geometry_agreement", agree, ">=70%",
            "ok" if agree >= 70 else "warn",
            "" if agree >= 70 else
            f"audio and FK disagree {round(100 - agree, 1)}% of frames "
            f"(audio-only {xc['audio_only_pct']}%, geo-only {xc['geo_only_pct']}%). "
            "geo-only = in-band but silent (plane drift / never marked); "
            "audio-only = drawing the plane calls airborne."))
    return checks


def print_report(a: dict) -> None:
    m, t = a["metrics"], a["threshold"]
    print(f"\n{a['run_id'] or a['wav']}")
    print(f"audio {a['audio']['duration_s']:.1f}s  {a['audio']['channels']} ch @ "
          f"{a['audio']['fs']} Hz   settle {a['dsp']['settle_s']:.0f}s   "
          f"threshold valid={t['valid']} (sep {t['separation_db']} dB)")
    print(f"\n  contact    {m['contact_pct_audio']:>5.1f}% of scored window (AUDIO)"
          f"   {m['n_contact_segments']} segments"
          f"   longest {m['longest_contact_s']:.1f}s   total {m['contact_s_total']:.1f}s")
    print(f"  levels     hover {m['hover_level_db']} dB   contact {m['contact_level_db']} dB"
          f"   (friction band {a['dsp']['friction_band_hz'][0]:.0f}-"
          f"{a['dsp']['friction_band_hz'][1]:.0f} Hz)")
    xc = a["cross_check"]
    if xc.get("available"):
        print(f"  vs FK      agree {xc['agreement_pct']:.1f}%"
              f"   audio-only {xc['audio_only_pct']:.1f}%   geo-only {xc['geo_only_pct']:.1f}%"
              f"   (FK in-band {xc['geometric_contact_pct']:.1f}%)")
        for w in xc["disagreement_windows"][:3]:
            print(f"             {w['who']:<11} {w['t_start_s']:.1f}-{w['t_end_s']:.1f}s")
    lb = a["labels"]
    if lb:
        print(f"  bench      accuracy {lb['accuracy_pct']}%   d' {lb['d_prime']}")
        for name, v in lb["labels"].items():
            print(f"             {name:<8} {v['span_s'][0]:.0f}-{v['span_s'][1]:.0f}s"
                  f"   band {v['friction_db_mean']} dB"
                  f"   classified contact {v['classified_contact_pct']}%")
    print()
    for c in a["checks"]:
        print(f"  {STATUS_MARK.get(c['status'], '  ?  ')}{c['name']:<20} "
              f"{str(c['value']):<24} expected {c['expected']}")
        if c["note"] and c["status"] != "ok":
            words, cur = c["note"].split(), ""
            for w in words:
                if len(cur) + len(w) + 1 > 66:
                    print(f"        {cur}")
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                print(f"        {cur}")


def compare(paths: list[Path]) -> None:
    rows, seen = [], set()
    for p in paths:
        with __import__("contextlib").suppress(Exception):
            row = json.loads(Path(p).read_text())
            key = row.get("run_id") or str(Path(p).resolve())
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if not rows:
        raise SystemExit("no readable audio_analysis.json files")
    print(f"\n{'run':<26}{'contact%':>10}{'segs':>6}{'longest':>9}{'sep dB':>14}"
          f"{'valid':>7}{'vs FK':>8}")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: -x["metrics"]["contact_pct_audio"]):
        m, t = r["metrics"], r["threshold"]
        name = str(r.get("run_id") or r["wav"]).rstrip("/").split("/")[-1][:25]
        agree = r["cross_check"].get("agreement_pct")
        print(f"{name:<26}{m['contact_pct_audio']:>9.1f}%{m['n_contact_segments']:>6}"
              f"{m['longest_contact_s']:>8.1f}s{str(t['separation_db']):>14}"
              f"{str(t['valid']):>7}{(f'{agree:.0f}%' if agree is not None else '-'):>8}")
    invalid = [r for r in rows if not r["threshold"]["valid"]]
    if invalid:
        print(f"\n!! {len(invalid)}/{len(rows)} runs have no believable acoustic split — "
              "their contact% is not evidence.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="*", help="run directory or wav file")
    ap.add_argument("--settle", type=float, default=SETTLE_S)
    ap.add_argument("--label", default=None,
                    help='bench spans, e.g. "hover:0-10,draw:10-20" — disables settle')
    ap.add_argument("--compare", action="store_true",
                    help="treat targets as audio_analysis.json files and tabulate them")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if not args.target:
        ap.error("need a run directory, wav file, or --compare audio_analysis.json...")
    if args.compare:
        compare([Path(p) for p in args.target])
        return 0
    for target in args.target:
        tp = Path(target).expanduser()
        a = analyze(tp, settle=args.settle, label_spec=args.label)
        run_dir = tp if tp.is_dir() else tp.parent
        out = run_dir / "audio_analysis.json"
        with __import__("contextlib").suppress(Exception):
            out.write_text(json.dumps(a, indent=2))
        if args.json:
            print(json.dumps(a, indent=2))
        else:
            print_report(a)
            print(f"\n  wrote {out}")
    return 0  # a report, never a gate — see il_rollout_async.sh


if __name__ == "__main__":
    sys.exit(main())
