"""The search (docs/lab.md "Searching"): where in mark space a study's read trials landed, a model of where untried
settings would land, and batches of trials aimed at what is not yet known.

- The archive (MAP-Elites): for one technique, cartridge and ink, two descriptor axes read from a trial's metrics,
  darkness and the technique's own (AXES), each cut into bins. A bin's elite is its cleanest trial, the least
  speckle about its strokes, and of two as clean the faster.
- The surrogate: per axis, a linear trend over the settings the study bounds, each scaled to [0, 1] by its bounds,
  and a Gaussian process over what the trend leaves, with the length scale and noise its read trials make likeliest.
  The trend carries a response past the trials (deeper is darker); the process alone falls back to the mean there.
  Under MIN_FIT read trials there is none.
- A batch mixes three kinds of trial. Novelty: settings the surrogate puts in a bin no trial holds, or, before
  there is a surrogate, settings far from every trial. Quality: a near neighbour of a weak elite. Surprise: an A/A
  repeat of the read trial the surrogate, fitted without it, predicted worst, once MIN_SURPRISE trials are read.

Each distinct depth trim and speed is a run of its own, one page set-up, so a batch holds at most `max_runs` of
them, snapped to TRIM_STEP_MM and SPEED_STEP_MM_S; past that, its trials take the nearest run already in it. Where
the runs are all the study lets vary and every one is picked, the rest of the batch repeats its own picks, for
their noise floor.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from tatbot_contracts import lab

from tatbot_lab import space

MIN_FIT = 4                          # read trials a surrogate needs
MIN_SURPRISE = 8                     # read trials before a misprediction is taken for a bad read, not a sparse model
RIDGE = 0.1                          # the trend's slopes are shrunk by this, so a few trials cannot tilt it far
CANDIDATES = 2048                    # random settings a novelty pick is chosen from
LENGTH_SCALES = (0.15, 0.3, 0.5, 0.8, 1.2)   # in units of a setting's bounds
NOISES = (0.01, 0.05, 0.15, 0.4)     # of the axis's variance
SURPRISE_Z = 3.0                     # a trial is surprising past this many predicted sds
STEP = 0.08                          # a quality pick's step from its elite, in units of the bounds
TRIM_STEP_MM, SPEED_STEP_MM_S = 0.25, 0.5
RUN_SETTINGS = {"trim_mm": TRIM_STEP_MM, "speed_mm_s": SPEED_STEP_MM_S}


@dataclass(frozen=True)
class Axis:
    """One descriptor: its name, its bin edges and how it is read from a trial's metrics."""
    name: str
    edges: tuple[float, ...]
    read: Callable[[dict], float | None]

    def value(self, metrics: dict) -> float | None:
        v = self.read(metrics or {})
        return None if v is None or not np.isfinite(v) else float(v)

    def bin(self, value: float | None) -> int | None:
        return None if value is None else int(np.searchsorted(self.edges, value, side="right"))

    def label(self, k: int) -> str:
        if k == 0:
            return f"<{self.edges[0]:g}"
        if k == len(self.edges):
            return f">{self.edges[-1]:g}"
        return f"{self.edges[k - 1]:g}-{self.edges[k]:g}"


def _width(m: dict) -> float | None:
    widths = [w for w in m.get("ink_width_mm") or [] if w is not None]
    return float(np.median(widths)) if widths else None


def _carry(m: dict) -> float | None:
    cycles = m.get("carry") or []
    return cycles[0].get("visible_mm") if cycles else None


def _speckle(m: dict) -> float | None:
    return (m.get("speckle") or {}).get("share")


DARKNESS = Axis("darkness", (3.0, 8.0, 15.0, 25.0, 40.0), lambda m: m.get("dL_mean"))   # under 3: no mark
AXES: dict[str, tuple[Axis, Axis]] = {
    "line": (DARKNESS, Axis("width_mm", (0.3, 0.45, 0.6, 0.8, 1.1), _width)),
    "carry": (DARKNESS, Axis("carry_mm", (40.0, 80.0, 160.0, 320.0), _carry)),
    "solid": (DARKNESS, Axis("speckle", (0.02, 0.05, 0.1, 0.2), _speckle)),
    "whip": (DARKNESS, Axis("fade", (1.5, 3.0, 6.0, 12.0), lambda m: (m.get("along") or {}).get("first_over_last"))),
    "corners": (DARKNESS, Axis("ends", (1.2, 1.6, 2.4), lambda m: m.get("ends_over_middles"))),
}


def members(trials: dict, technique: str, cartridge: str, ink: str) -> list[dict]:
    """The study's trials of one archive: this technique, cartridge and (first) ink, in the order written."""
    return [t for t in trials.values() if t["technique"] == technique and t["cartridge"]["type"] == cartridge
            and t["inks"][:1] == [ink]]


def descriptors(trial: dict) -> tuple[float | None, float | None]:
    return tuple(axis.value(trial.get("metrics")) for axis in AXES[trial["technique"]])


def quality(trial: dict) -> tuple[float, float]:
    """Higher is better: the least speckle about the strokes (no blowout, clean edges), then the fastest."""
    return -(_speckle(trial.get("metrics") or {}) or 0.0), float(trial["settings"]["speed_mm_s"])


def archive(group: list[dict]) -> dict[tuple[int, int], dict]:
    """(darkness bin, second bin) -> the elite trial among the group's read trials."""
    out: dict[tuple[int, int], dict] = {}
    for trial in group:
        bins = tuple(axis.bin(v) for axis, v in zip(AXES[trial["technique"]], descriptors(trial), strict=True))
        if None in bins:
            continue
        if bins not in out or quality(trial) > quality(out[bins]):
            out[bins] = trial
    return out


class Surrogate:
    """A linear trend over unit settings and a Gaussian process (squared-exponential kernel) over its residuals: the
    trend a ridge fit, the process's length scale and noise the grid's likeliest for the residuals."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, float)
        self.mean, self.spread = float(np.mean(y)), float(np.std(y)) or 1.0
        z = (np.asarray(y, float) - self.mean) / self.spread
        design = np.column_stack([np.ones(len(self.x)), self.x])
        shrink = RIDGE * np.diag([0.0] + [1.0] * self.x.shape[1])
        self.trend = np.linalg.solve(design.T @ design + shrink, design.T @ z)
        self.z = z - design @ self.trend
        self.length, self.noise = max(((ls, nz) for ls in LENGTH_SCALES for nz in NOISES),
                                      key=lambda p: self._evidence(*p))
        self.inverse = np.linalg.inv(self._gram(self.length, self.noise))
        self.alpha = self.inverse @ self.z

    def _line(self, q: np.ndarray) -> np.ndarray:
        return self.trend[0] + np.atleast_2d(q) @ self.trend[1:]

    def _kernel(self, a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-0.5 * d2 / length ** 2)

    def _gram(self, length: float, noise: float) -> np.ndarray:
        return self._kernel(self.x, self.x, length) + (noise + 1e-9) * np.eye(len(self.x))

    def _evidence(self, length: float, noise: float) -> float:
        chol = np.linalg.cholesky(self._gram(length, noise))
        w = np.linalg.solve(chol, self.z)
        return float(-0.5 * w @ w - np.log(np.diag(chol)).sum())

    def predict(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and sd at unit settings `q` (rows), in the axis's own units."""
        k = self._kernel(np.atleast_2d(q), self.x, self.length)
        mean = self._line(q) + k @ self.alpha
        var = np.clip(1.0 + self.noise - np.einsum("ij,jk,ik->i", k, self.inverse, k), 1e-12, None)
        return self.mean + self.spread * mean, self.spread * np.sqrt(var)

    def observed(self) -> np.ndarray:
        return self.mean + self.spread * (self._line(self.x) + self.z)

    def held_out(self) -> tuple[np.ndarray, np.ndarray]:
        """Each data point's prediction (mean, sd, in the axis's units) from a surrogate fitted afresh to all the
        others, its trend, length scale and noise included: a bad read fitted with the rest inflates their spread
        and hides itself (a +40 dL* cell read 2.5 sds off that way, 2026-11-09's tests)."""
        y = self.observed()
        out = [Surrogate(np.delete(self.x, i, axis=0), np.delete(y, i)).predict(self.x[i:i + 1])
               for i in range(len(self.x))]
        return np.array([m[0] for m, _ in out]), np.array([sd[0] for _, sd in out])


@dataclass
class Space:
    """The settings a study lets one technique's search move, each scaled to [0, 1] by its bounds, and those it
    holds (bounds with low equal to high)."""
    names: list[str]
    low: np.ndarray
    high: np.ndarray
    held: dict[str, float]

    @classmethod
    def of(cls, study: dict, technique: str) -> Space:
        uses = {*space.TECHNIQUES[technique], *space.REQUIRED, *space.DIP}
        bounds = {n: (float(lo), float(hi)) for n, (lo, hi) in study["bounds"].items() if n in uses}
        if missing := [n for n in space.REQUIRED if n not in bounds]:
            raise ValueError(f"the search moves within the study's bounds, and it bounds no {', '.join(missing)}")
        names = sorted(n for n, (lo, hi) in bounds.items() if hi > lo)
        return cls(names, np.array([bounds[n][0] for n in names]), np.array([bounds[n][1] for n in names]),
                   {n: lo for n, (lo, hi) in bounds.items() if hi <= lo})

    def unit(self, settings: dict, technique: str) -> np.ndarray:
        full = space.resolve(technique, settings)
        return np.clip((np.array([float(full[n]) for n in self.names]) - self.low) / (self.high - self.low), 0, 1)

    def settings(self, x: np.ndarray) -> dict:
        out = dict(self.held)
        for name, lo, hi, v in zip(self.names, self.low, self.high, x, strict=True):
            value = float(lo) + float(v) * float(hi - lo)
            out[name] = int(round(value)) if lab.SETTINGS[name] is int else round(value, 3)
        return out

    def snap(self, x: np.ndarray) -> np.ndarray:
        """`x` with its run settings on their steps, inside the bounds."""
        x = np.array(x, float)
        for name, step in RUN_SETTINGS.items():
            if name in self.names:
                i = self.names.index(name)
                value = np.clip(round((self.low[i] + x[i] * (self.high[i] - self.low[i])) / step) * step,
                                self.low[i], self.high[i])
                x[i] = (value - self.low[i]) / (self.high[i] - self.low[i])
        return x

    def run_of(self, x: np.ndarray) -> tuple[float, ...]:
        return tuple(round(float(x[self.names.index(n)]), 6) for n in RUN_SETTINGS if n in self.names)

    def onto(self, x: np.ndarray, runs: list[tuple[float, ...]]) -> np.ndarray:
        """`x` moved onto the nearest of `runs` (unit run settings)."""
        idx = [self.names.index(n) for n in RUN_SETTINGS if n in self.names]
        if not idx or not runs:
            return x
        best = min(runs, key=lambda r: float(np.linalg.norm(np.asarray(r) - x[idx])))
        x = np.array(x, float)
        x[idx] = best
        return x


@dataclass
class Batch:
    """A batch being chosen: its picks so far, the runs they need and the empty bins they aim at."""
    space: Space
    max_runs: int
    runs: list[tuple[float, ...]]
    points: list[np.ndarray]
    proposals: list[dict]
    aimed: set[tuple[int, int]]

    def place(self, x: np.ndarray) -> np.ndarray:
        x = self.space.snap(x)
        if self.space.run_of(x) not in self.runs and len(self.runs) >= self.max_runs:
            x = self.space.onto(x, self.runs)
        return x

    def add(self, x: np.ndarray, proposal: dict) -> None:
        if (run := self.space.run_of(x)) not in self.runs:
            self.runs.append(run)
        self.points.append(np.asarray(x, float))
        self.proposals.append(proposal)


def _surrogates(sp: Space, group: list[dict], technique: str) -> list[tuple[Surrogate, list[dict]] | None]:
    out = []
    for k, _ in enumerate(AXES[technique]):
        read = [t for t in group if descriptors(t)[k] is not None]
        if len(read) < MIN_FIT:
            out.append(None)
            continue
        x = np.stack([sp.unit(t["settings"], technique) for t in read])
        out.append((Surrogate(x, np.array([descriptors(t)[k] for t in read])), read))
    return out


def _proposal(technique: str, cartridge: str, ink: str, settings: dict, why: str, repeat_of: str | None = None):
    return {"technique": technique, "cartridge": cartridge, "inks": [ink], "settings": settings, "hypothesis": why,
            **({"repeat_of": repeat_of} if repeat_of else {})}


def _surprises(models, technique: str) -> list[tuple[float, dict, str]]:
    """(|z|, trial, why) for each read trial its held-out prediction missed by more than SURPRISE_Z sds."""
    out: dict[str, tuple[float, dict, str]] = {}
    for axis, model in zip(AXES[technique], models, strict=True):
        if model is None:
            continue
        surrogate, read = model
        mean, sd = surrogate.held_out()
        if len(read) < MIN_SURPRISE:
            continue
        for trial, m, s, y in zip(read, mean, sd, surrogate.observed(), strict=True):
            z = abs(y - m) / s
            if z > SURPRISE_Z and z > out.get(trial["id"], (0.0,))[0]:
                out[trial["id"]] = (z, trial, f"surprise: {axis.name} read {y:.3g}, the other trials predict "
                                              f"{m:.3g} +- {s:.2g}; an A/A repeat says which to trust")
    return sorted(out.values(), key=lambda item: -item[0])


def _weak_elites(elites: dict) -> list[tuple[tuple[int, int], dict]]:
    """Elites whose speckle is above the archive's median elite's, the worst first."""
    if len(elites) < 2:
        return []
    shares = {bins: _speckle(t.get("metrics") or {}) or 0.0 for bins, t in elites.items()}
    median = float(np.median(list(shares.values())))
    return sorted(((b, elites[b]) for b in elites if shares[b] > median), key=lambda item: -shares[item[0]])


def _novelty(batch: Batch, models, elites: dict, technique: str,
             rng: np.random.Generator) -> tuple[np.ndarray, str, tuple[int, int] | None]:
    """The candidate that most adds to what is known, why, and the bin it aims at: an empty bin the surrogate
    predicts, its uncertainty, and its distance from every trial and pick."""
    sp, axes = batch.space, AXES[technique]
    cands = np.stack([batch.place(x) for x in rng.random((CANDIDATES, len(sp.names)))])
    known = np.stack(batch.points) if batch.points else np.empty((0, len(sp.names)))
    spacing = (np.min(np.linalg.norm(cands[:, None, :] - known[None, :, :], axis=-1), axis=1) if len(known)
               else np.full(len(cands), np.sqrt(len(sp.names))))
    if all(m is not None for m in models):
        predicted = [m[0].predict(cands) for m in models]
        bins = [np.searchsorted(axis.edges, mean, side="right") for axis, (mean, _) in zip(axes, predicted, strict=True)]
        taken = set(elites) | batch.aimed
        empty = np.array([(int(a), int(b)) not in taken for a, b in zip(*bins, strict=True)])
        doubt = np.mean([sd / m[0].spread for (_, sd), m in zip(predicted, models, strict=True)], axis=0)
        k = int(np.argmax(2.0 * empty + doubt + spacing))
        where = ", ".join(f"{axis.name} ~{mean[k]:.3g} ({axis.label(int(b[k]))})"
                          for axis, (mean, _), b in zip(axes, predicted, bins, strict=True))
        why = f"novelty: the model puts it at {where}, " + ("a bin no trial holds" if empty[k] else "where it is least sure")
        return cands[k], why, (int(bins[0][k]), int(bins[1][k]))
    k = int(np.argmax(spacing))
    return cands[k], f"novelty: as far as it gets from every trial so far (too few read for a model, {MIN_FIT} needed)", None


def batch(study: dict, trials: dict, technique: str, cartridge: str, ink: str, n: int, *, seed: int = 0,
          max_runs: int = 3) -> list[dict]:
    """`n` proposals for one archive (propose.add writes them): surprises first, then weak elites' neighbours, the
    rest novelty, in at most `max_runs` runs of depth and speed."""
    if technique not in AXES:
        raise ValueError(f"technique {technique!r} has no descriptors; it is one of {', '.join(AXES)}")
    sp, rng = Space.of(study, technique), np.random.default_rng(seed)
    group = members(trials, technique, cartridge, ink)
    read = [t for t in group if t.get("metrics")]
    models = _surrogates(sp, read, technique)
    elites = archive(read)
    out = Batch(sp, max_runs, [], [sp.unit(t["settings"], technique) for t in group], [], set())
    repeats = _surprises(models, technique)[:max(1, n // 5) if n >= 3 else 0]
    for _, trial, why in repeats:
        x = sp.unit(trial["settings"], technique)
        out.add(x, _proposal(technique, cartridge, ink, dict(trial["settings"]), why, repeat_of=trial["id"]))
    for bins, elite in _weak_elites(elites)[:max(1, n // 5) if n >= 3 else 0]:
        if len(out.proposals) >= n:
            break
        x = out.place(np.clip(sp.unit(elite["settings"], technique) + rng.normal(0.0, STEP, len(sp.names)), 0, 1))
        cells = " x ".join(axis.label(b) for axis, b in zip(AXES[technique], bins, strict=True))
        out.add(x, _proposal(technique, cartridge, ink, sp.settings(x),
                             f"quality: {elite['id']} holds {cells} with speckle {_speckle(elite['metrics']) or 0:.3g}; "
                             "a neighbour may hold it cleaner"))
    while len(out.proposals) < n:
        x, why, aim = _novelty(out, models, elites, technique, rng)
        if (k := _same_pick(out, x)) is not None:   # nothing new left inside the runs: repeat a pick instead
            why = f"repeat of this batch's pick {k + 1}: its runs hold nothing else new, so it measures the floor"
        out.add(x, _proposal(technique, cartridge, ink, sp.settings(x), why))
        if aim is not None:
            out.aimed.add(aim)
    return out.proposals


def _same_pick(batch: Batch, x: np.ndarray) -> int | None:
    """The batch's earlier pick at these very settings, if any."""
    return next((k for k, p in enumerate(batch.proposals) if p["settings"] == batch.space.settings(x)), None)


def grid(group: list[dict]) -> dict:
    """The archive as a table: each bin's elite id and descriptors, for `tatbot lab map`."""
    technique = group[0]["technique"]
    elites = archive([t for t in group if t.get("metrics")])
    rows, cols = AXES[technique]
    return {"technique": technique, "cartridge": group[0]["cartridge"]["type"], "ink": group[0]["inks"][0],
            "axes": [rows.name, cols.name], "rows": [rows.label(k) for k in range(len(rows.edges) + 1)],
            "cols": [cols.label(k) for k in range(len(cols.edges) + 1)],
            "read": sum(1 for t in group if t.get("metrics")), "trials": len(group),
            "elites": {f"{a},{b}": {"trial": t["id"], "descriptors": list(descriptors(t)), "quality": list(quality(t))}
                       for (a, b), t in sorted(elites.items())},
            "bins": (len(rows.edges) + 1) * (len(cols.edges) + 1)}


def groups(trials: dict) -> list[list[dict]]:
    """The study's trials by archive: technique, cartridge and first ink, in the order first written."""
    out: dict[tuple, list[dict]] = {}
    for t in trials.values():
        if t["technique"] in AXES:
            out.setdefault((t["technique"], t["cartridge"]["type"], t["inks"][0]), []).append(t)
    return list(out.values())
