"""Decoder for the coded flower-of-life frame (`scripts/lib/stencil_coded.py`): the `coded` tracker.

1. **Ink and skin.** Both transfer inks absorb red and green, so the ink map is
   the negative log of those channels. The skin is where the image is warm (red
   over blue) after smoothing; the grey table, mat and clutter are not.
2. **Knots.** Every junction carries a small solid knot. A scale-normalised
   Laplacian of Gaussian finds knot candidates as signal over the skin's noise.
   Bare line junctions are not distinctive at 2-3 px/mm; the knots are.
3. **Lattice.** A strong knot seeds a local lattice: the spacing and orientation
   whose first, sqrt(3) and second rings of neighbours hold the most candidate
   score. The lattice grows one step at a time from local affine fits, snapping
   each prediction to a nearby candidate or leaving a virtual node, so it
   crosses small washed-off gaps.
4. **Bits.** Each lattice edge compares the ink on its two candidate arcs. A
   washed-off or pooled edge reads near zero: an erasure, not a wrong bit.
5. **Decode.** A component's edges vote for a lattice symmetry (12) and offset
   against every registered print's code. It decodes only when its best
   hypothesis is far beyond chance over the whole hypothesis space.
6. **Page frame.** Decoded components move into the page frame and the lattice
   grows over the page's own junctions. Components too weak to decode alone
   attach when the page model fixes their symmetry and bounds their offset; the
   page model also bridges washed-off gaps. Every junction must then be
   supported by its own bits against the page code.
7. **Model.** The page plane's pose through the junctions (six degrees of
   freedom, so one strip of the frame still extrapolates), bent locally by the
   junctions' residuals where the skin curves.

It refuses rather than guesses: no decodable component, a contested decode,
components that disagree, or page bits that do not support the decode return
`rejected` or `ambiguous`.
"""

from __future__ import annotations

import heapq
import math
import time

import cv2
import numpy as np
import stencil_coded as coded
from stencil_bench_trackers import ACCEPTED, AMBIGUOUS, Located

SQRT3 = math.sqrt(3)
DIRS = np.array(coded.DIRS, float)
STEPS = tuple((int(a), int(b)) for a, b in coded.DIRS)
TRANSFORMS = coded.transforms()
SIGMAS_PX = tuple(1.0*1.2**k for k in range(16))       # knot LoG scales, 1-15 px
MIN_ROUNDNESS = .35         # smaller over larger Hessian eigenvalue a knot candidate needs
CANDIDATE_SNR = 4.0         # a knot candidate's LoG response over the skin's noise
SEED_SNR = 8.0              # a seed's
SEED_POOL = 160             # strongest candidates considered as seeds, ranked by lattice support
SEED_TRIES = 80             # seeds grown at most
WORKING_SPACINGS_PX = (28., 23., 19.)   # lattice spacings a decode is tried at, in order
MAX_SKIN_PX = 1200          # longest skin extent decoded at full resolution
SEED_HYPOTHESES = 3         # (spacing, angle) guesses a seed grows before keeping the best
SNAP = .25                  # a prediction snaps to a candidate this many spacings away
MIN_DETECTED = 6            # detected junctions a component needs before it is decoded
MAX_VIRTUAL_DEPTH = 2       # predicted-only steps away from a detected junction
DECISIVE_BIT = .35          # |bit| and presence (in line contrasts) that make an edge count
DECISIVE_PRESENCE = .4

# log10 chance of the best agreement over the whole hypothesis space. Measured on 64 train
# scenes: every component decoded below 1e-2 was right, every wrong hypothesis sat above it.
SIGNIFICANCE = -4.0
RUNNER_UP = -2.0            # the second-best hypothesis must not be this significant
ATTACH_MIN = 4              # detected junctions a weak component needs to attach to a decoded one
ATTACH_REACH = 4            # lattice steps around the page model's prediction an attachment searches
ATTACH_SIGNIFICANCE = -3.0
CYLINDER_MIN = 12           # junctions before a cylinder is fitted
CYLINDER_GAIN = .7          # a cylinder replaces the plane when its residual is under this share
BRIDGE_ROUNDS = 4
BRIDGE_MIN = 15             # detected page junctions before the page model may bridge gaps
BRIDGE_MM = 15.0            # how far past detected junctions the page model predicts others
BRIDGE_AGREEMENT = .85      # bits on bridged junctions' edges that must match the page code
GROUP_MIN_AGREE = 6         # agreeing edge ends an attached or bridged group needs to be kept


# --- ink, skin and knots ------------------------------------------------------------------------

def ink_map(image):
    """Negative log of red and green: both inks absorb there, and shading and exposure become
    additive offsets that the LoG ignores."""
    logs = np.log(image.astype(np.float32)+4.)
    return -(logs[..., 2]+logs[..., 1])/2


def skin_mask(image, blur_px):
    """Where the skin is: warm (red over blue) after smoothing thin lines away, plus every region
    the skin encloses. A dense real transfer makes the whole frame band violet or blue rather
    than warm, and the skin always surrounds it. The table, mat and clutter are grey and reach
    the image border; a region larger than a quarter of the image is never filled."""
    logs = np.log(image.astype(np.float32)+4.)
    skin = (cv2.GaussianBlur(logs[..., 2]-logs[..., 0], (0, 0), blur_px) > .12).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1-skin, connectivity=4)
    h, w = skin.shape
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if x > 0 and y > 0 and x+bw < w and y+bh < h and area < .25*h*w:
            skin[labels == label] = 1
    return skin.astype(bool)


def substrate_mask(image, substrate):
    """Where knots may be: the skin (`skin`, the bench and phone photos), or the whole image
    (`any`: paper prints and grey frames, which carry no warm skin; the caller bounds the search
    to regions of the frame instead)."""
    if substrate == "any":
        return np.ones(image.shape[:2], bool)
    if substrate != "skin":
        raise ValueError(f"unknown coded substrate {substrate!r}")
    return skin_mask(image, 6.)


def _parabola(a, b, c):
    denominator = a-2*b+c
    return 0. if abs(denominator) < 1e-9 else float(np.clip(.5*(a-c)/denominator, -.5, .5))


def _sample(image, xy):
    xy = np.asarray(xy, np.float32).reshape(-1, 1, 2)
    return cv2.remap(image, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1)


class KnotStack:
    """Knot candidates: scale-normalised Laplacian-of-Gaussian maxima over space and scale, as
    signal over the skin's noise, whose whole support lies on skin.

    Work runs on the skin's bounding box (an overhead view's pad is a small part of its frame);
    candidate positions are in full-image pixels. A candidate at scale sigma implies a lattice
    spacing sigma*ratio, the print's spacing over its transferred knot's LoG scale; blur and
    spread make that good only to a factor of two, so it only bounds the search."""

    def __init__(self, image, ratio, substrate="skin"):
        self.shape = image.shape[:2]
        self.xy, self.score, self.spacing = np.empty((0, 2)), np.empty(0), np.empty(0)
        self._seeds = None
        skin = substrate_mask(image, substrate)
        ys, xs = np.nonzero(skin)
        if not len(xs):
            self.origin = np.zeros(2)
            self.smooth = np.zeros((1, 1), np.float32)
            return
        pad = int(4*SIGMAS_PX[-1])+2
        h, w = self.shape
        x0, y0 = max(0, xs.min()-pad), max(0, ys.min()-pad)
        x1, y1 = min(w, xs.max()+1+pad), min(h, ys.max()+1+pad)
        self.origin = np.array([x0, y0], float)
        self.ink = ink_map(image[y0:y1, x0:x1])
        self.smooth = cv2.GaussianBlur(self.ink, (0, 0), .7)
        self._candidates(self._levels(skin[y0:y1, x0:x1]), skin[y0:y1, x0:x1], ratio)

    def _levels(self, skin):
        """LoG signal-to-noise per scale, zeroed where the blob is elongated: a knot is round, a
        spread petal or a stroke is not."""
        levels = []
        for sigma in SIGMAS_PX:
            blurred = cv2.GaussianBlur(self.ink, (0, 0), sigma)
            dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
            dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
            dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
            response = -(dxx+dyy)*sigma*sigma
            values = response[skin]
            noise = 1.4826*float(np.median(np.abs(values-np.median(values))))+1e-6
            spread = np.sqrt((dxx-dyy)**2+4*dxy**2)
            roundness = (np.abs(dxx+dyy)-spread)/(np.abs(dxx+dyy)+spread+1e-9)   # small/large eigenvalue
            levels.append(np.where(roundness >= MIN_ROUNDNESS, response/noise, 0))
        return np.stack(levels)

    def _candidates(self, levels, skin, ratio):
        spatial = np.stack([level >= cv2.dilate(level, np.ones((3, 3), np.uint8)) for level in levels])
        across = np.ones_like(spatial)
        across[1:] &= levels[1:] >= levels[:-1]
        across[:-1] &= levels[:-1] >= levels[1:]
        # A candidate's whole LoG support must lie on skin: the skin's own silhouette is a strong edge.
        inside = np.stack([cv2.erode(skin.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(6*sigma) | 1,)*2), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            for sigma in SIGMAS_PX])
        k, y, x = np.nonzero(spatial & across & (levels > CANDIDATE_SNR) & inside)
        xy = np.c_[x, y].astype(float)
        h, w = levels.shape[1:]
        for i, (kk, yy, xx) in enumerate(zip(k, y, x, strict=True)):
            if 0 < yy < h-1 and 0 < xx < w-1:
                level = levels[kk]
                xy[i, 0] += _parabola(level[yy, xx-1], level[yy, xx], level[yy, xx+1])
                xy[i, 1] += _parabola(level[yy-1, xx], level[yy, xx], level[yy+1, xx])
        self.xy, self.score = xy+self.origin, levels[k, y, x]
        self.spacing = np.array(SIGMAS_PX)[k]*ratio
        self.level = k

    def sample(self, xy):
        """Smoothed ink at full-image pixels."""
        return _sample(self.smooth, np.asarray(xy, float)-self.origin)

    def near(self, xy, radius, spacing, exclude, level=None):
        """Strongest candidate within `radius` of `xy` whose scale fits: within one level of the
        component's knots when `level` is known (marks and ornaments sit at other scales), else
        within a factor of two of `spacing`. Index or None."""
        if not len(self.xy):
            return None
        d = np.hypot(*(self.xy-xy).T)
        scale = (np.abs(self.level-level) <= 1) if level is not None else \
            (np.abs(np.log(self.spacing/spacing)) < math.log(2.))
        ok = (d <= radius) & scale & ~exclude
        if not ok.any():
            return None
        index = np.flatnonzero(ok)
        return int(index[np.argmax(self.score[index]-2*d[index]/radius)])

    def seeds(self):
        """Candidates with a lattice around them, best-supported lattice first: (score, xy,
        [(spacing, angle)], index). Ranking by lattice support rather than blob strength keeps
        text, rulers and blemishes, which are strong blobs but no lattice, from using up the
        attempts."""
        if self._seeds is not None:
            return self._seeds
        out = []
        for i in np.argsort(-self.score)[:SEED_POOL]:
            if self.score[i] < SEED_SNR:
                break
            support, found = self._lattice_at(i)
            if found:
                out.append((support, (float(self.score[i]), self.xy[i].copy(), found, int(i))))
        out.sort(key=lambda item: -item[0])
        self._seeds = [seed for _, seed in out]
        return self._seeds

    def _lattice_at(self, i):
        """(spacing, angle) hypotheses around candidate i, best supported first: local maxima over
        spacing of the candidate score on the first, sqrt(3) and second rings of neighbours. A
        lattice of twice or sqrt(3) times the spacing is a sublattice and scores about the same;
        growth tells them apart. Only candidates at about the seed's own blob scale count, since
        knots share a size and petals, dots and ornaments crowd the ring at others; each kept
        hypothesis's angle is then refined to the best-supported one within 15 degrees."""
        guess = self.spacing[i]
        delta = self.xy-self.xy[i]
        d = np.hypot(delta[:, 0], delta[:, 1])
        near = (d > .3*guess) & (d < 4.2*guess) & (np.abs(np.log(self.spacing/guess)) < math.log(1.5))
        if near.sum() < 3:
            return 0., []
        delta, d, weight = delta[near], d[near], np.minimum(self.score[near], 3*CANDIDATE_SNR)
        theta = np.arctan2(delta[:, 1], delta[:, 0])
        results = []
        for spacing in guess*np.exp(np.linspace(math.log(.5), math.log(2.1), 30)):
            first = np.abs(d/spacing-1) < .12
            if first.sum() < 2:
                continue
            angle = float(np.angle((weight[first]*np.exp(6j*theta[first])).sum())/6)
            results.append((_ring_support(delta, weight, spacing, angle), float(spacing), angle))
        if not results or max(r[0] for r in results) < 4*CANDIDATE_SNR:
            return 0., []
        best = max(r[0] for r in results)
        peaks = [r for j, r in enumerate(results) if r[0] >= .4*best
                 and (j == 0 or r[0] >= results[j-1][0]) and (j == len(results)-1 or r[0] >= results[j+1][0])]
        peaks.sort(key=lambda r: -r[0])
        return best, [(spacing, _refine_angle(delta, weight, spacing, angle)) for _, spacing, angle in peaks[:SEED_HYPOTHESES]]


def _refine_angle(delta, weight, spacing, angle):
    turns = angle+np.radians(np.arange(-15, 15.1, 1.5))
    return float(turns[int(np.argmax(_ring_supports(delta, weight, spacing, turns)))])


RING = np.array([(radius, turn+k*math.pi/3) for radius, turn in ((1., 0.), (SQRT3, math.pi/6), (2., 0.))
                 for k in range(6)])       # (radius in spacings, angle offset) of the lattice's first three rings


def _ring_supports(delta, weight, spacing, angles):
    """Per lattice angle: the summed weight of the strongest candidate near each of the first
    three rings' 18 lattice positions."""
    angles = np.atleast_1d(np.asarray(angles, float))
    turn = angles[:, None]+RING[None, :, 1]
    expected = spacing*RING[None, :, 0, None]*np.stack([np.cos(turn), np.sin(turn)], -1)
    distance = np.hypot(*(expected[:, :, None, :]-delta[None, None]).transpose(3, 0, 1, 2))
    hit = np.where(distance < .15*spacing, weight[None, None], 0.)
    return hit.max(-1).sum(-1) if delta.size else np.zeros(len(angles))


def _ring_support(delta, weight, spacing, angle):
    return float(_ring_supports(delta, weight, spacing, [angle])[0])


# --- lattice ------------------------------------------------------------------------------------

def _hex(dq, dr):
    return (abs(dq)+abs(dr)+abs(dq+dr))//2


def _neighbours(key):
    return [(key[0]+da, key[1]+db) for da, db in STEPS]


def _spacing(linear):
    return float(np.sqrt(abs(np.linalg.det(linear))/(SQRT3/2)))


class Lattice(dict):
    """{(a, b): node} in one component's own lattice frame, with fast local affine fits. A node
    is a dict: xy, score, detected, depth (virtual steps from a detected node), candidate."""

    def __init__(self):
        super().__init__()
        self._keys, self._xy, self._levels = [], [], []

    def place(self, key, node):
        self[key] = node
        if node["detected"]:
            self._keys.append(key)
            self._xy.append(node["xy"])
            if node.get("level") is not None:
                self._levels.append(node["level"])

    def level(self):
        """The knots' typical LoG level, once three are known."""
        return int(np.median(self._levels)) if len(self._levels) >= 3 else None

    def detected(self):
        return list(self._keys)

    def affine(self, around, radius):
        """Weighted least-squares lattice->pixel affine (2x3) from detected nodes near `around`."""
        if len(self._keys) < 3:
            return None
        keys = np.asarray(self._keys, float)
        d = keys-np.asarray(around, float)
        hexd = (np.abs(d[:, 0])+np.abs(d[:, 1])+np.abs(d.sum(1)))/2
        near = hexd <= radius
        if near.sum() < 3:
            return None
        design = np.c_[keys[near], np.ones(int(near.sum()))]
        if np.linalg.matrix_rank(design) < 3:
            return None
        weights = 1./(1+hexd[near])
        solution, *_ = np.linalg.lstsq(design*weights[:, None], np.asarray(self._xy)[near]*weights[:, None],
                                       rcond=None)
        return solution.T

    def spacing(self):
        affine = self.affine(self._keys[0], 99) if self._keys else None
        return _spacing(affine[:, :2]) if affine is not None else 10.


def _predict(nodes, key, basis):
    """(pixel, local lattice->pixel linear map) for an unplaced key, or None."""
    for radius in (2, 3, 4):
        affine = nodes.affine(key, radius)
        if affine is not None:
            return affine[:, :2]@np.array(key, float)+affine[:, 2], affine[:, :2]
    if basis is None:
        return None
    anchor = min(nodes._keys, key=lambda k: _hex(k[0]-key[0], k[1]-key[1]))
    return nodes[anchor]["xy"]+basis@np.array([key[0]-anchor[0], key[1]-anchor[1]], float), basis


def _place(stack, nodes, key, prediction, claimed, group=None):
    """Snap a predicted key to a candidate, or make it virtual; returns whether it was placed."""
    predicted, linear = prediction
    spacing = _spacing(linear)
    found = stack.near(predicted, SNAP*spacing, spacing, claimed, nodes.level())
    if found is not None:
        claimed[found] = True
        nodes.place(key, {"xy": stack.xy[found].copy(), "score": float(stack.score[found]), "detected": True,
                          "depth": 0, "candidate": found, "level": int(stack.level[found]), "group": group})
        return True
    depth = min(nodes[n]["depth"] for n in _neighbours(key) if n in nodes)+1
    if depth > MAX_VIRTUAL_DEPTH:
        return False
    nodes.place(key, {"xy": predicted, "score": 0., "detected": False, "depth": depth})
    return True


def grow(stack, nodes, claimed, basis=None, allowed=None, group=None):
    """Grow a lattice component over the knot candidates, in place, most constrained position
    first. `basis` (2x2, lattice to pixels) predicts before three detected nodes exist;
    `allowed` limits the positions (the page's own junctions once the frame is known);
    `claimed` marks used candidates; new nodes carry `group`."""
    h, w = stack.shape
    heap, visited = [], set(nodes)

    def push(key):
        for neighbour in _neighbours(key):
            if neighbour not in visited and (allowed is None or neighbour in allowed):
                heapq.heappush(heap, (-sum(n in nodes for n in _neighbours(neighbour)), neighbour))

    for key in list(nodes):
        push(key)
    while heap:
        _, key = heapq.heappop(heap)
        if key in visited:
            continue
        visited.add(key)
        prediction = _predict(nodes, key, basis)
        if prediction is None or not (0 <= prediction[0][0] < w and 0 <= prediction[0][1] < h
                                      and 4 < _spacing(prediction[1]) < 80):
            continue
        if _place(stack, nodes, key, prediction, claimed, group) and nodes[key]["depth"] < MAX_VIRTUAL_DEPTH:
            push(key)
    return nodes


def seed_components(stack, seed, used):
    """The components grown from one seed, one per (spacing, angle) hypothesis, in the seed's own
    lattice frame. A lattice of twice or sqrt(3) times the spacing still snaps to candidates
    (they are dense along the band), so the caller keeps whichever decodes best."""
    score, xy, hypotheses, index = seed
    for spacing, angle in hypotheses:
        basis = spacing*np.array([[math.cos(angle), math.cos(angle+math.pi/3)],
                                  [math.sin(angle), math.sin(angle+math.pi/3)]])
        nodes = Lattice()
        nodes.place((0, 0), {"xy": xy, "score": score, "detected": True, "depth": 0, "candidate": index,
                             "level": int(stack.level[index])})
        claimed = used.copy()
        claimed[index] = True
        yield grow(stack, nodes, claimed, basis=basis)


# --- bits ---------------------------------------------------------------------------------------

def _arc_offsets():
    """Chord fractions and across offsets of the arc samples; across is the fraction of the way
    from the chord midpoint to the bulge vertex."""
    t = np.array(coded.SAMPLE_T)
    return t, (np.sqrt(1-(t-.5)**2)-SQRT3/2)/(SQRT3/2)


def edge_measure(stack, p, q, linear, k, mode="side"):
    """(ink difference between the two candidate arcs, ink presence) for the edge from pixel p
    to pixel q along lattice direction k, with `linear` the local lattice-to-pixel map. The
    difference is positive when the arc bulging toward DIRS[k+1] carries more ink; presence is
    the inkier arc over the two triangle centroids beside the edge. A `teardrop` edge compares
    the petal's centre line near p with near q instead: positive when the end at p is filled."""
    if isinstance(mode, tuple):
        return _mark_measure(stack, p, q, linear, k, mode)
    t, across = _arc_offsets()
    chord = p[None]*(1-t[:, None])+q[None]*t[:, None]
    values, centroids = [], []
    for side in (1, -1):
        apex = linear@(DIRS[(k+side) % 6]-DIRS[k]/2)     # chord midpoint to bulge vertex
        values.append(stack.sample(chord+across[:, None]*apex[None]))
        centroids.append((p+q)/2+apex/3)
    background = float(np.mean(stack.sample(np.array(centroids))))
    return float(np.median(values[0]-values[1])), max(float(np.median(v)) for v in values)-background


def _mark_measure(stack, p, q, linear, k, fractions):
    t = np.array(fractions)
    near_p = stack.sample(p[None]*(1-t[:, None])+q[None]*t[:, None])
    near_q = stack.sample(q[None]*(1-t[:, None])+p[None]*t[:, None])
    centroids = [(p+q)/2+linear@(DIRS[(k+side) % 6]-DIRS[k]/2)/3 for side in (1, -1)]
    background = float(np.mean(stack.sample(np.array(centroids))))
    return float(np.median(near_p-near_q)), max(float(np.median(near_p)), float(np.median(near_q)))-background


def read_bits(stack, nodes, mode="side"):
    """Soft bits for every edge with a detected end: [(a, b, k, bit, presence)] in the component
    frame, in units of the component's line contrast (its inkier quarter of edges are intact).
    bit > 0: more ink on the arc bulging toward (a, b)+DIRS[k+1]. A clear edge reads bit ~ +-1
    and presence ~ 1; a washed-off one presence ~ 0 whatever its bit."""
    edges = []
    for (a, b), node in nodes.items():
        affine = None
        for k in range(3):
            other = (a+STEPS[k][0], b+STEPS[k][1])
            if other not in nodes or not (node["detected"] or nodes[other]["detected"]):
                continue
            affine = nodes.affine((a, b), 2) if affine is None else affine
            if affine is not None:
                edges.append((a, b, k, *edge_measure(stack, node["xy"], nodes[other]["xy"], affine[:, :2], k, mode)))
    if not edges:
        return []
    contrast = max(float(np.percentile([e[4] for e in edges], 75)), 1e-3)
    return [(a, b, k, float(np.clip(d/contrast, -3, 3)), e/contrast) for a, b, k, d, e in edges]


def decisive(edges):
    return [(a, b, k, bit) for a, b, k, bit, presence in edges
            if abs(bit) >= DECISIVE_BIT and presence >= DECISIVE_PRESENCE]


# --- decode -------------------------------------------------------------------------------------

class Codebook:
    """One registered print: its page edges by direction, for offset voting."""

    def __init__(self, artwork, code):
        self.pattern_id = artwork.pattern_id
        self.print_id = code["print_id"]
        self.geometry = code["geometry"]
        self.layout = coded.Layout(**self.geometry, centre_mm=code["centre_mm"])
        self.page = np.array([self.geometry["width_mm"], self.geometry["height_mm"]], float)
        edges = np.array(code["edges"], np.int64)
        self.by_direction = {k: (edges[edges[:, 2] == k][:, :2], edges[edges[:, 2] == k][:, 3]) for k in range(3)}
        style = code.get("style", {})
        self.reflect_flips = style.get("bit", "side") == "side"
        # The reader's mode: "side", or the chord fractions (from each end) where a mark sits.
        if style.get("bit") == "teardrop":
            start, stop = style.get("teardrop_from", .3), style.get("teardrop_to", .5)
            self.mode = tuple(start+(stop-start)*f for f in (.25, .5, .75))
        elif style.get("bit") == "seed":
            self.mode = tuple(style.get("seed_at", .36)+d for d in (-.04, 0., .04))
        else:
            self.mode = "side"
        self.edge_bit = {(q, r, k): b for q, r, k, b in code["edges"]}
        self.nodes = {tuple(n) for n in code["nodes"]}
        self.spacing_mm = float(self.geometry["spacing_mm"])
        keys = sorted(self.nodes)
        self.key_by_uv = {tuple(np.round(uv, 9)): key for key, uv in zip(keys, self.uv(keys), strict=True)}

    def uv(self, lattice):
        return np.array([self.layout.position(q, r) for q, r in lattice]).reshape(-1, 2)/self.page

    def agreement(self, observed, sym, offset):
        """(agree, disagree) of decisive observed edges mapped by a symmetry and offset."""
        sigma, m, matrix = TRANSFORMS[sym]
        agree = disagree = 0
        for a, b, k, bit in observed:
            q, r, kk, mapped = coded.map_edge(a, b, k, 1 if bit > 0 else -1, sigma, m, matrix, offset,
                                              self.reflect_flips)
            expected = self.edge_bit.get((q, r, kk))
            if expected is not None:
                agree += expected == mapped
                disagree += expected != mapped
        return agree, disagree


def log10_tail(n, k):
    """log10 P(Binomial(n, 1/2) >= k)."""
    if k <= n/2:
        return 0.
    terms = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1) for i in range(k, n+1)]
    top = max(terms)
    return (top+math.log(sum(math.exp(v-top) for v in terms))-n*math.log(2))/math.log(10)


def vote(edges, codebooks):
    """Every (print, symmetry, offset) hypothesis's agreement with the decisive observed bits.

    Returns hypotheses sorted by agreement minus disagreement; the top three carry the decisive
    edge count and log10 chance of their agreement over the whole hypothesis space. An observed
    edge that a hypothesis maps where the page draws nothing counts as a non-agreement."""
    observed = decisive(edges)
    if not observed:
        return []
    results = []
    for book_index, book in enumerate(codebooks):
        for sym in range(len(TRANSFORMS)):
            results += _vote_symmetry(observed, book, book_index, sym)
    results.sort(key=lambda r: -r["score"])
    hypotheses = sum(len(b.nodes) for b in codebooks)*len(TRANSFORMS)
    for r in results[:3]:
        r["decisive"] = len(observed)
        r["log10_chance"] = log10_tail(len(observed), r["agree"])+math.log10(max(hypotheses, 1))
    return results


def _vote_symmetry(observed, book, book_index, sym, span=256):
    """Top three offsets for one print and symmetry, by bincounting every edge pairing."""
    sigma, m, matrix = TRANSFORMS[sym]
    mapped = np.array([coded.map_edge(a, b, k, 1 if bit > 0 else -1, sigma, m, matrix, (0, 0), book.reflect_flips)
                       for a, b, k, bit in observed], np.int64)
    agree = np.zeros(span*span, np.int64)
    disagree = np.zeros(span*span, np.int64)
    for kk in range(3):
        rows = mapped[mapped[:, 2] == kk]
        page_nodes, page_bits = book.by_direction[kk]
        if not len(rows) or not len(page_nodes):
            continue
        offsets = page_nodes[None, :, :]-rows[:, None, :2]+span//2
        # A component grown far from its seed can pair edges at offsets no page reaches.
        inside = ((offsets >= 0) & (offsets < span)).all(-1).ravel()
        keys = (offsets[..., 0]*span+offsets[..., 1]).ravel()[inside]
        same = (page_bits[None, :] == rows[:, 3:4]).ravel()[inside]
        agree += np.bincount(keys[same], minlength=span*span)
        disagree += np.bincount(keys[~same], minlength=span*span)
    score = agree-disagree
    return [{"book": book_index, "sym": sym, "offset": (int(key//span-span//2), int(key % span-span//2)),
             "agree": int(agree[key]), "disagree": int(disagree[key]), "score": int(score[key])}
            for key in np.argpartition(-score, 3)[:3] if agree[key] > 0]


# --- model --------------------------------------------------------------------------------------

class LocalModel:
    """uv -> pixel: the page's surface posed in the camera through all junctions (a homography
    when the camera is unknown), bent locally by the junctions' residuals.

    With intrinsics the surface is the page wrapped on a cylinder, or the flat page when a
    cylinder fits no better (`Surface`). Its pose has six degrees of freedom where a homography
    has eight, so a view that sees only one strip of the frame still extrapolates sensibly to
    the rest, and the cylinder follows an arm around its curve. The residual correction is a
    Gaussian-weighted mean of nearby junctions' residuals (page mm), damped by a prior weight:
    it follows wobble where junctions were seen and fades back to the surface elsewhere."""

    def __init__(self, uv, pixels, page_mm, intrinsics=None, bandwidth_mm=6.0, prior=.3, curved=False):
        self.page = np.asarray(page_mm, float)
        self.ctrl = uv*self.page
        self.surface = Surface.fit(self.ctrl, pixels, self.page, intrinsics, curved) if intrinsics else None
        if self.surface is None:
            self.h, _ = cv2.findHomography(uv.astype(np.float64), pixels.astype(np.float64), 0)
        self.residual = pixels-self._plane(uv)
        self.bandwidth, self.prior = bandwidth_mm, prior

    def _plane(self, uv):
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        if self.surface is not None:
            return self.surface.project(uv*self.page)
        return cv2.perspectiveTransform(uv.reshape(-1, 1, 2), self.h).reshape(-1, 2)

    def __call__(self, uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        weights = np.exp(-(((uv*self.page)[:, None]-self.ctrl[None])**2).sum(-1)/(2*self.bandwidth**2))
        return self._plane(uv)+weights@self.residual/(weights.sum(1, keepdims=True)+self.prior)


class Surface:
    """The page wrapped on a cylinder, posed in the camera. The cylinder's axis lies at `angle`
    in the page, its crest `offset` mm from the page centre across the axis, and `curvature` is
    1/radius, signed (zero is the flat page). Page point p lands at arc length along the curve,
    so the page is unrolled onto the cylinder without stretch."""

    STEPS = np.array([1e-4]*3+[1e-2]*3+[1e-4, 1e-2, 1e-6])

    def __init__(self, params, page, camera, dist):
        self.params, self.centre, self.camera, self.dist = np.asarray(params, float), page/2, camera, dist

    def points(self, mm, params=None):
        angle, offset, curvature = (self.params if params is None else params)[6:]
        across = np.array([math.cos(angle), math.sin(angle)])
        along = np.array([-across[1], across[0]])
        crest = self.centre+offset*across
        rel = np.asarray(mm, float).reshape(-1, 2)-crest
        u, v = rel@across, rel@along
        if abs(curvature) < 1e-7:
            x, z = u, .5*curvature*u*u
        else:
            x, z = np.sin(curvature*u)/curvature, (1-np.cos(curvature*u))/curvature
        return np.c_[crest+np.outer(x, across)+np.outer(v, along), z]

    def project(self, mm, params=None):
        params = self.params if params is None else params
        return cv2.projectPoints(self.points(mm, params), params[:3], params[3:6], self.camera,
                                 self.dist)[0].reshape(-1, 2)

    @classmethod
    def fit(cls, mm, pixels, page, intrinsics, curved):
        """The flat page's pose, or the cylinder when `curved` and it fits clearly better (a
        residual under CYLINDER_GAIN of the plane's). None when no pose is found."""
        camera = np.array([[intrinsics["fx"], 0, intrinsics["cx"]], [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1.]])
        dist = np.array(intrinsics.get("distortion") or [0.]*5, float)
        ok, rvec, tvec = cv2.solvePnP(np.c_[mm, np.zeros(len(mm))].astype(np.float64), pixels.astype(np.float64),
                                      camera, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or float(tvec[2, 0]) <= 0:
            return None
        plane = cls(np.r_[rvec.ravel(), tvec.ravel(), 0., 0., 0.], page, camera, dist)
        if not curved or len(mm) < CYLINDER_MIN:
            return plane
        best, best_cost = plane, plane._cost(mm, pixels)
        plane_cost = best_cost
        for angle in (0., math.pi/2, math.pi/4, -math.pi/4):
            for curvature in (1/45., -1/45.):
                trial = cls(np.r_[rvec.ravel(), tvec.ravel(), angle, 0., curvature], page, camera, dist)
                cost = trial._refine(mm, pixels)
                if cost < best_cost:
                    best, best_cost = trial, cost
        return best if best_cost < CYLINDER_GAIN*plane_cost else plane

    def _cost(self, mm, pixels, params=None):
        return float(((self.project(mm, params)-pixels)**2).sum())

    def _refine(self, mm, pixels, iterations=40):
        """Levenberg-Marquardt on the reprojection error, numeric Jacobian. Returns the cost."""
        params, damping = self.params.copy(), 1e-3
        cost = self._cost(mm, pixels, params)
        for _ in range(iterations):
            residual = (self.project(mm, params)-pixels).ravel()
            jacobian = np.stack([((self.project(mm, params+step*np.eye(9)[j])-pixels).ravel()-residual)/step
                                 for j, step in enumerate(self.STEPS)], 1)
            normal = jacobian.T@jacobian
            change = np.linalg.solve(normal+damping*np.diag(np.diag(normal)+1e-9), -jacobian.T@residual)
            trial = self._cost(mm, pixels, params+change)
            if trial < cost:
                params, cost, damping = params+change, trial, damping/3
                if abs(params[8]) > 1/15.:       # tighter than a 15 mm radius is not an arm
                    return math.inf
            else:
                damping *= 5
                if damping > 1e6:
                    break
        self.params = params
        return cost


# --- tracker ------------------------------------------------------------------------------------

class RefusedError(Exception):
    def __init__(self, reason, status="rejected"):
        super().__init__(reason)
        self.reason, self.status = reason, status


class CodedTracker:
    """Coded flower-of-life decoder over the registered prints' codes. `substrate` is where knots
    may lie (`substrate_mask`); the bench decodes skin."""

    name = "coded"

    def __init__(self, max_components=6, substrate="skin"):
        self.codebooks = []
        self.max_components = max_components
        self.substrate = substrate
        self.intrinsics = None
        self.deadline = None

    def prepare(self, artworks):
        for artwork in artworks:
            code = coded.load_code(artwork.reference.parent) if artwork.reference is not None else None
            if code is None:
                raise ValueError(f"the coded tracker needs a coded.json beside {artwork.reference}")
            self.codebooks.append(Codebook(artwork, code))
        if not self.codebooks:
            raise ValueError("the coded tracker needs at least one coded artwork")
        geometry = self.codebooks[0].geometry
        self.mode = self.codebooks[0].mode
        if not geometry.get("knot_mm"):
            raise ValueError("the coded tracker locates junction knots; this print has none (knot_mm=0)")
        # LoG scale of a transferred knot: its radius grows by about half the stroke's spread.
        knot_sigma = (geometry["knot_mm"]/2+.45*geometry["stroke_mm"])/math.sqrt(2)
        self.ratio = geometry["spacing_mm"]/knot_sigma

    def locate(self, image, intrinsics):
        # Coarse first: a close, sharp view (a phone photo) is brought to at most MAX_SKIN_PX of
        # skin, where the seeds see beads rather than ink speckle; then the seeds' spacing sets
        # the working scale. Wrist and overhead frames are left alone by both steps.
        ys, xs = np.nonzero(substrate_mask(image, self.substrate))
        extent = max(np.ptp(xs), np.ptp(ys))+1 if len(xs) else 0
        coarse = min(1., MAX_SKIN_PX/extent) if extent else 1.
        if coarse < 1:
            image = cv2.resize(image, None, fx=coarse, fy=coarse, interpolation=cv2.INTER_AREA)
            intrinsics = _scaled_intrinsics(intrinsics, coarse)
        stack = KnotStack(image, self.ratio, self.substrate)
        first = None
        for fine in _working_scales(stack):
            attempt = stack if fine == 1 else KnotStack(
                cv2.resize(image, None, fx=fine, fy=fine, interpolation=cv2.INTER_AREA), self.ratio, self.substrate)
            located = self._decode(attempt, _scaled_intrinsics(intrinsics, fine))
            located = _rescaled(located, coarse*fine) if coarse*fine < 1 else located
            if located.status == ACCEPTED:
                return located
            first = first or located
        return first

    def locate_all(self, image, intrinsics, deadline=None):
        """Every registered print in one image: ({pattern_id: Located}, the image's refusal when
        no component decodes). Components are grouped by the print they decode as and each
        print's group is decoded as `locate` decodes one, so registered prints side by side do
        not refuse each other; a component contested between prints refuses both, and one
        print's components that disagree refuse that print. `deadline` (a `time.perf_counter`
        value) ends the search for components early; what was found by then is decoded."""
        self.deadline = deadline
        try:
            ys, xs = np.nonzero(substrate_mask(image, self.substrate))
            extent = max(np.ptp(xs), np.ptp(ys))+1 if len(xs) else 0
            coarse = min(1., MAX_SKIN_PX/extent) if extent else 1.
            if coarse < 1:
                image = cv2.resize(image, None, fx=coarse, fy=coarse, interpolation=cv2.INTER_AREA)
                intrinsics = _scaled_intrinsics(intrinsics, coarse)
            stack = KnotStack(image, self.ratio, self.substrate)
            found, refusal = {}, None
            for fine in _working_scales(stack):
                if self._late():
                    refusal = refusal or Located.rejected("decode_time_budget")
                    break
                attempt = stack if fine == 1 else KnotStack(
                    cv2.resize(image, None, fx=fine, fy=fine, interpolation=cv2.INTER_AREA), self.ratio, self.substrate)
                results, failed = self._decode_all(attempt, _scaled_intrinsics(intrinsics, fine))
                for pattern, located in results.items():
                    if pattern not in found or located.status == ACCEPTED:
                        found[pattern] = _rescaled(located, coarse*fine) if coarse*fine < 1 else located
                refusal = refusal or failed
                if any(located.status == ACCEPTED for located in found.values()):
                    break
            return found, refusal
        finally:
            self.deadline = None

    def _late(self):
        return self.deadline is not None and time.perf_counter() > self.deadline

    def _decode_all(self, stack, intrinsics):
        """({pattern_id: Located} per print with a decoded component, refusal or None)."""
        self.intrinsics = intrinsics
        try:
            decoded, pending = self._components(stack, split=True)
        except RefusedError as refusal:
            return {}, Located.rejected(refusal.reason, refusal.status)
        groups, refused = {}, {}
        for d in decoded:
            groups.setdefault(d["best"]["book"], []).append(d)
            if d["contested"]:
                refused.update(dict.fromkeys({d["best"]["book"], d["runner_up"]}, "contested_decode"))
        for index, group in groups.items():
            if len({TRANSFORMS[d["best"]["sym"]][0] for d in group}) > 1:
                refused.setdefault(index, "components_disagree")
        results, claimed = {}, np.zeros(len(stack.xy), bool)
        for index in sorted(groups, key=lambda i: min(d["best"]["log10_chance"] for d in groups[i])):
            pattern = self.codebooks[index].pattern_id
            try:
                if index in refused:
                    raise RefusedError(refused[index], AMBIGUOUS)
                book, page, claimed, spacing = self._page_frame(stack, groups[index], claimed)
                self._extend(stack, book, page, claimed, pending, spacing)
                results[pattern] = self._result(stack, book, page, groups[index], spacing)
            except RefusedError as refusal:
                results[pattern] = Located(refusal.status, pattern, np.empty((0, 2)), np.empty((0, 2)),
                                           reason=refusal.reason)
        return results, None

    def _decode(self, stack, intrinsics):
        self.intrinsics = intrinsics
        try:
            decoded, pending = self._components(stack)
            book, page, claimed, spacing = self._page_frame(stack, decoded)
            self._extend(stack, book, page, claimed, pending, spacing)
            return self._result(stack, book, page, decoded, spacing)
        except RefusedError as refusal:
            return Located.rejected(refusal.reason, refusal.status)

    def _components(self, stack, split=False):
        """Grow components from the seeds; decode each alone. Returns (decoded, pending). With
        `split` the caller judges contested and disagreeing components per print."""
        used = np.zeros(len(stack.xy), bool)
        decoded, pending, reasons = [], [], []
        for tried, seed in enumerate(stack.seeds()):
            if (tried >= SEED_TRIES or len([r for r in reasons if r != "too_few_junctions"]) >= 4*self.max_components
                    or len(decoded) >= self.max_components):
                break
            if self._late():
                reasons.insert(0, "decode_time_budget")
                break
            if used[seed[3]]:
                continue
            best = self._best_component(stack, seed, used)
            if best is None:
                reasons.append("too_few_junctions")
                continue
            nodes, edges, hypotheses = best
            used[[nodes[k]["candidate"] for k in nodes.detected()]] = True
            if not hypotheses or hypotheses[0]["log10_chance"] > SIGNIFICANCE:
                # Too weak alone; it may still attach to a decoded component's page frame.
                pending.append({"nodes": nodes, "edges": edges})
                reasons.append("weak_decode")
                continue
            contested = len(hypotheses) > 1 and hypotheses[1]["log10_chance"] <= RUNNER_UP
            decoded.append({"nodes": nodes, "edges": edges, "best": hypotheses[0], "contested": contested,
                            "runner_up": hypotheses[1]["book"] if contested else None})
        _refuse_components(decoded, reasons, split)
        return decoded, pending

    def _best_component(self, stack, seed, used):
        """(nodes, edges, hypotheses) of the seed's most significant component, else its largest;
        None when none has ATTACH_MIN junctions."""
        best = None
        for nodes in seed_components(stack, seed, used):
            detected = len(nodes.detected())
            if detected < ATTACH_MIN:
                continue
            edges = read_bits(stack, nodes, self.mode)
            hypotheses = vote(edges, self.codebooks) if detected >= MIN_DETECTED else []
            rank = (hypotheses[0]["log10_chance"] if hypotheses else 99., -detected)
            if best is None or rank < best[0]:
                best = (rank, (nodes, edges, hypotheses))
        return best[1] if best else None

    def _page_frame(self, stack, decoded, claimed=None):
        """The decoded components' junctions in page lattice coordinates, slipped junctions
        dropped. Two components that share five or more junctions must put them in the same place
        on median, or the decode is ambiguous; a shared junction they place apart (a growth error
        in either) is kept by neither."""
        book = self.codebooks[decoded[0]["best"]["book"]]
        placed, owner, apart, conflicts = {}, {}, {}, set()
        for index, d in enumerate(sorted(decoded, key=lambda d: d["best"]["log10_chance"])):
            limit = .3*d["nodes"].spacing()
            for key, target in self._page_targets(book, d).items():
                node = d["nodes"][key]
                if target in placed:
                    distance = float(np.hypot(*(placed[target]["xy"]-node["xy"])))
                    apart.setdefault((owner[target], index), []).append(distance/limit)
                    if distance > limit:
                        conflicts.add(target)
                    continue
                placed[target], owner[target] = node, index
        if any(len(ratios) >= 5 and np.median(ratios) > 1 for ratios in apart.values()):
            raise RefusedError("components_disagree", AMBIGUOUS)
        page = Lattice()
        claimed = np.zeros(len(stack.xy), bool) if claimed is None else claimed.copy()
        for target, node in placed.items():
            if target not in conflicts:
                page.place(target, dict(node))
                claimed[node["candidate"]] = True
        return book, page, claimed, float(np.median([d["nodes"].spacing() for d in decoded]))

    @staticmethod
    def _page_targets(book, decoded):
        """{component key: page key} for a decoded component's detected junctions on the page,
        without the ones its own bits say slipped."""
        sigma, m, matrix = TRANSFORMS[decoded["best"]["sym"]]
        offset = decoded["best"]["offset"]
        targets = {key: (matrix[0][0]*key[0]+matrix[0][1]*key[1]+offset[0],
                         matrix[1][0]*key[0]+matrix[1][1]*key[1]+offset[1]) for key in decoded["nodes"].detected()}
        page_edges = [(*coded.map_edge(a, b, k, 1 if bit > 0 else -1, sigma, m, matrix, offset, book.reflect_flips),
                       presence) for a, b, k, bit, presence in decoded["edges"] if abs(bit) >= DECISIVE_BIT]
        slipped = _slipped(book, page_edges, targets.values())
        return {key: target for key, target in targets.items() if target in book.nodes and target not in slipped}

    def _extend(self, stack, book, page, claimed, pending, spacing):
        """Grow over the page's junctions, attach weak components and bridge washed-off gaps."""
        grow(stack, page, claimed, allowed=book.nodes)
        # What the page model reaches is judged later as a group, with whatever grew from it.
        for component in sorted(pending, key=lambda c: -len(c["nodes"]._keys)):
            group = f"attach-{id(component)}"
            if self._attach(stack, book, page, claimed, component, spacing, group):
                grow(stack, page, claimed, allowed=book.nodes, group=group)
        for _ in range(BRIDGE_ROUNDS):
            if not self._bridge(stack, book, page, claimed, spacing):
                break
            grow(stack, page, claimed, allowed=book.nodes, group="bridge")

    def _model(self, book, page):
        detected = page.detected()
        return LocalModel(book.uv(detected), np.array([page[k]["xy"] for k in detected]), book.page,
                          self.intrinsics)

    def _attach(self, stack, book, page, claimed, component, spacing, group):
        """Place a component too weak to decode alone into the page frame. Its orientation against
        the page model fixes the lattice symmetry and the model bounds its offset to a few steps,
        so a handful of agreeing bits is already far beyond chance. Returns whether it attached."""
        nodes = component["nodes"]
        keys = [k for k in nodes.detected() if not claimed[nodes[k]["candidate"]]]
        if len(page.detected()) < MIN_DETECTED or len(keys) < ATTACH_MIN:
            return False
        frame = self._attach_frame(book, page, nodes, keys, spacing)
        if frame is None:
            return False
        sym, base, model = frame
        offsets = self._plausible_offsets(book, nodes, keys, sym, base, model, spacing)
        observed = decisive(component["edges"])
        scores = sorted(((agree-disagree, agree, offset) for offset in offsets
                         for agree, disagree in [book.agreement(observed, sym, offset)]), key=lambda item: -item[0])
        if not scores:
            return False
        best, agree, offset = scores[0]
        # Chance over at least three offsets, though the geometry usually leaves one or two.
        if (log10_tail(len(observed), agree)+math.log10(max(3, len(scores))) > ATTACH_SIGNIFICANCE
                or (len(scores) > 1 and scores[1][0] > best-3)):
            return False
        matrix = np.array(TRANSFORMS[sym][2])
        placed = False
        for key in keys:
            target = (int(matrix[0]@key)+offset[0], int(matrix[1]@key)+offset[1])
            if target in book.nodes and not (target in page and page[target]["detected"]):
                page.place(target, dict(nodes[key], group=group))
                claimed[nodes[key]["candidate"]] = True
                placed = True
        return placed

    def _attach_frame(self, book, page, nodes, keys, spacing):
        """(symmetry, offset guess) that puts a component's centre junction where the page model
        predicts its nearest page junction, or None when no symmetry fits the local lattice."""
        model = self._model(book, page)
        page_keys = sorted(book.nodes)
        centroid = np.mean([nodes[k]["xy"] for k in keys], 0)
        centre = min(keys, key=lambda k: np.hypot(*(nodes[k]["xy"]-centroid)))
        p0 = page_keys[int(np.argmin(np.hypot(*(model(book.uv(page_keys))-nodes[centre]["xy"]).T)))]
        base = model(book.uv([p0]))[0]
        local_page = np.c_[model(book.uv([(p0[0]+1, p0[1])]))[0]-base, model(book.uv([(p0[0], p0[1]+1)]))[0]-base]
        affine = nodes.affine(centre, 99)
        if affine is None:
            return None
        fits = [np.linalg.norm(affine[:, :2]-local_page@np.array(matrix, float)) for _, _, matrix in TRANSFORMS]
        sym = int(np.argmin(fits))
        # The page model extrapolates into the component's region, so only a clear winner counts.
        if fits[sym] > .6*spacing or fits[sym] > .5*sorted(fits)[1]:
            return None
        matrix = np.array(TRANSFORMS[sym][2])
        return sym, (p0[0]-int(matrix[0]@centre), p0[1]-int(matrix[1]@centre)), model

    @staticmethod
    def _plausible_offsets(book, nodes, keys, sym, base, model, spacing):
        """Offsets within ATTACH_REACH of the guess that put the component's junctions within half
        a spacing (median) of where the page model predicts them."""
        matrix = np.array(TRANSFORMS[sym][2])
        mapped = np.array(keys)@matrix.T
        pixels = np.array([nodes[k]["xy"] for k in keys])
        out = []
        for dq in range(-ATTACH_REACH, ATTACH_REACH+1):
            for dr in range(-ATTACH_REACH, ATTACH_REACH+1):
                offset = (base[0]+dq, base[1]+dr)
                if _hex(dq, dr) > ATTACH_REACH:
                    continue
                residual = np.hypot(*(model(book.uv(mapped+offset))-pixels).T)
                if np.median(residual) < .5*spacing:
                    out.append(offset)
        return out

    def _bridge(self, stack, book, page, claimed, spacing):
        """Snap unplaced page junctions within BRIDGE_MM of a detected one to candidates where the
        page model predicts them. Returns whether any were placed."""
        detected = page.detected()
        missing = [k for k in book.nodes if k not in page or not page[k]["detected"]]
        if len(detected) < BRIDGE_MIN or not missing:
            return False
        model = self._model(book, page)
        seen_mm, missing_uv = book.uv(detected)*book.page, book.uv(missing)
        near = np.min(np.hypot(*((missing_uv*book.page)[:, None]-seen_mm[None]).transpose(2, 0, 1)), 1)
        placed = False
        for key, xy in zip(np.array(missing)[near <= BRIDGE_MM], model(missing_uv[near <= BRIDGE_MM]), strict=True):
            found = stack.near(xy, SNAP*spacing, spacing, claimed, page.level()) if np.isfinite(xy).all() else None
            if found is not None:
                claimed[found] = True
                page.place(tuple(int(v) for v in key), {"xy": stack.xy[found].copy(), "score": float(stack.score[found]),
                                                        "detected": True, "depth": 0, "candidate": found,
                                                        "level": int(stack.level[found]),
                                                        "group": "bridge", "bridged": True})
                placed = True
        return placed

    def _result(self, stack, book, page, decoded, spacing):
        """Keep the junctions their own bits support against the page code; fit the model."""
        edges = read_bits(stack, page, self.mode)
        support, agree, disagree = _support(book, edges)
        slipped = _slipped(book, edges, page.detected())
        # The decoded components and what grew from them must agree with the code as a whole.
        core = [support.get(k, (0, 0)) for k in page.detected() if not page[k].get("group")]
        if sum(a for a, _ in core) < 3*sum(d for _, d in core):
            raise RefusedError("page_bits_disagree", AMBIGUOUS)
        keys = [k for k in _trusted(page, support) if k not in slipped]
        if len(keys) < MIN_DETECTED:
            raise RefusedError("too_few_junctions")
        uv, pixels = book.uv(keys), np.array([page[k]["xy"] for k in keys])
        _, inliers = cv2.findHomography(uv, pixels, cv2.RANSAC, max(3., .3*spacing))
        keep = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(uv), bool)
        uv, pixels = uv[keep], pixels[keep]
        best = min(decoded, key=lambda d: d["best"]["log10_chance"])["best"]
        return Located(ACCEPTED, book.pattern_id, uv, pixels,
                       confidence=np.array([page[k]["score"] for k in keys])[keep], reason="decoded",
                       model=LocalModel(uv, pixels, book.page, self.intrinsics, curved=True)
                       if len(uv) >= MIN_DETECTED else None,
                       extra={"mirrored": bool(TRANSFORMS[best["sym"]][0] < 0), "print_id": book.print_id,
                              "components": len(decoded), "log10_chance": round(best["log10_chance"], 2),
                              "decisive_edges": best["decisive"], "page_bits_agree": agree,
                              "page_bits_disagree": disagree})


def _refuse_components(decoded, reasons, split):
    """Raise the image's refusal: no component decoded, or (unless `split`, where the caller
    judges each print) a contested component or components that disagree."""
    if not decoded:
        raise RefusedError("decode_time_budget" if reasons[:1] == ["decode_time_budget"] else
                           "weak_decode" if "weak_decode" in reasons else reasons[0] if reasons else "no_lattice")
    if split:
        return
    if any(d["contested"] for d in decoded):
        raise RefusedError("contested_decode", AMBIGUOUS)
    if len({(d["best"]["book"], TRANSFORMS[d["best"]["sym"]][0]) for d in decoded}) > 1:
        raise RefusedError("components_disagree", AMBIGUOUS)


def _working_scales(stack):
    """Downsampling factors to try, in order: the lattice at about each of WORKING_SPACINGS_PX,
    never enlarged. At high resolution a real bead's ink speckle breaks it into many small
    blobs that lattice growth snaps to, and on a real transfer photo no single spacing decoded
    at every resolution, so a failed decode is retried at the next."""
    seeds = stack.seeds()[:5]
    if not seeds:
        return [1.]
    spacing = float(np.median([seed[2][0][0] for seed in seeds]))
    factors = []
    for target in WORKING_SPACINGS_PX:
        factor = min(1., target/spacing)
        if not factors or factor < .92*factors[-1]:
            factors.append(factor)
    return factors


def _scaled_intrinsics(intrinsics, factor):
    if not intrinsics:
        return intrinsics
    scaled = dict(intrinsics)
    for key in ("fx", "fy", "cx", "cy"):
        scaled[key] = intrinsics[key]*factor
    for key in ("width", "height"):
        if key in scaled:
            scaled[key] = round(intrinsics[key]*factor)
    return scaled


def _rescaled(located, factor):
    """A result found on an image downsampled by `factor`, in the original image's pixels."""
    if located.status != ACCEPTED:
        return located
    model = located.model
    located.pixels = located.pixels/factor
    located.model = (lambda uv: model(uv)/factor) if model is not None else None
    located.extra["working_scale"] = round(factor, 3)
    return located


def _support(book, edges):
    """Per page junction [agree, disagree] of its decisive edges against the page code, and totals."""
    support, agree, disagree = {}, 0, 0
    for a, b, k, bit in decisive(edges):
        expected = book.edge_bit.get((a, b, k))
        if expected is None:
            continue
        ok = (bit > 0) == (expected > 0)
        agree, disagree = agree+ok, disagree+(not ok)
        for key in ((a, b), (a+STEPS[k][0], b+STEPS[k][1])):
            support.setdefault(key, [0, 0])[0 if ok else 1] += 1
    return support, agree, disagree


def _slipped(book, edges, keys, radius=2, margin=2):
    """Junctions whose neighbourhood a one-step lattice shift explains better than their own
    place: growth slipped there, and a slipped region is locally consistent in the image, so
    only its bits show it. Compares decisive edges within `radius` steps against the page code
    as placed and under each of the six unit shifts."""
    observed = decisive(edges)
    if not observed:
        return set()
    starts = np.array([(a, b) for a, b, _, _ in observed], float)
    slipped = set()
    for key in keys:
        d = starts-np.array(key, float)
        near = (np.abs(d[:, 0])+np.abs(d[:, 1])+np.abs(d.sum(1)))/2 <= radius
        local = [observed[i] for i in np.flatnonzero(near)]
        if len(local) < 4:
            continue
        placed = _agree_shifted(book, local, (0, 0))
        if max(_agree_shifted(book, local, step) for step in STEPS) >= placed+margin:
            slipped.add(key)
    return slipped


def _agree_shifted(book, local, step):
    score = 0
    for a, b, k, bit in local:
        expected = book.edge_bit.get((a+step[0], b+step[1], k))
        if expected is not None:
            score += 1 if (bit > 0) == (expected > 0) else -1
    return score


def _trusted(page, support):
    """Detected page junctions their own bits support. Junctions the page model reached (an
    attached component, the bridged set) are judged together first: a group with fewer than
    GROUP_MIN_AGREE agreeing edge ends, or whose edges agree with the page code less than
    BRIDGE_AGREEMENT of the time, is dropped whole; a bridged junction also needs two agreeing
    edges and none disagreeing."""
    groups = {}
    for key in page.detected():
        if page[key].get("group"):
            groups.setdefault(page[key]["group"], []).append(support.get(key, (0, 0)))
    trusted = {g for g, votes in groups.items() if sum(a for a, _ in votes) >= GROUP_MIN_AGREE
               and sum(a for a, _ in votes) >= BRIDGE_AGREEMENT*sum(a+d for a, d in votes)}
    keys = []
    for key in page.detected():
        agree, disagree = support.get(key, (0, 0))
        group = page[key].get("group")
        if group and group not in trusted:
            continue
        if disagree <= agree if not page[key].get("bridged") else disagree == 0 and agree >= 2:
            keys.append(key)
    return keys


# --- verification at tracked junctions -----------------------------------------------------------

def verify_bits(image, uv, pixels, book, mode, homography):
    """(agree, disagree) of the page code's decisive edges between tracked junctions: the print's
    own bits re-read where tracking put its junctions, with no search. `uv` are exact page-lattice
    UVs of the junctions (a decode's), `homography` the current page UV -> pixel fit, which gives
    each edge its local lattice-to-pixel map. A slipped lattice or another print reads at chance.
    Vectorised: the same measures as `edge_measure`, for every edge at once."""
    keys = [book.key_by_uv.get(tuple(np.round(value, 9))) for value in np.asarray(uv, float)]
    where = {key: np.asarray(xy, float) for key, xy in zip(keys, pixels, strict=True) if key is not None}
    edges = [(key, k) for key in where for k in range(3)
             if (key[0]+STEPS[k][0], key[1]+STEPS[k][1]) in where
             and (key[0], key[1], k) in book.edge_bit]
    if not edges:
        return 0, 0
    p = np.array([where[key] for key, _ in edges])
    q = np.array([where[(key[0]+STEPS[k][0], key[1]+STEPS[k][1])] for key, k in edges])
    k = np.array([k for _, k in edges])
    linear = _lattice_linear(homography, (p+q)/2, book)
    stack = _InkPatch(image, np.r_[p, q], 2*np.hypot(*(q-p).T).max())
    diff, presence = _measure_edges(stack, p, q, linear, k, mode)
    contrast = max(float(np.percentile(presence, 75)), 1e-3)
    bits, presence = np.clip(diff/contrast, -3, 3), presence/contrast
    agree = disagree = 0
    for (key, kk), bit, seen in zip(edges, bits, presence, strict=True):
        if abs(bit) >= DECISIVE_BIT and seen >= DECISIVE_PRESENCE:
            ok = (bit > 0) == (book.edge_bit[(key[0], key[1], kk)] > 0)
            agree, disagree = agree+ok, disagree+(not ok)
    return agree, disagree


class _InkPatch:
    """The smoothed ink map (as a KnotStack's) over the box around some pixels."""

    def __init__(self, image, points, pad):
        h, w = image.shape[:2]
        low = np.maximum(np.floor(points.min(0)-pad), 0).astype(int)
        high = np.minimum(np.ceil(points.max(0)+pad)+1, [w, h]).astype(int)
        self.origin = low.astype(float)
        crop = image[low[1]:high[1], low[0]:high[0]]
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        self.smooth = cv2.GaussianBlur(ink_map(crop), (0, 0), .7)

    def sample(self, xy):
        xy = np.asarray(xy, float)
        return _sample(self.smooth, xy.reshape(-1, 2)-self.origin).reshape(xy.shape[:-1])


def _lattice_linear(homography, pixels_hint, book):
    """Per point, the page lattice -> pixel linear map through the homography's Jacobian at the
    page point that maps to `pixels_hint`."""
    h = np.asarray(homography, float)
    inverse = np.linalg.inv(h)
    uv = cv2.perspectiveTransform(np.asarray(pixels_hint, float).reshape(-1, 1, 2), inverse).reshape(-1, 2)
    ones = np.c_[uv, np.ones(len(uv))]
    projected = ones@h.T
    w = projected[:, 2:3]
    xy = projected[:, :2]/w
    # d(pixel)/d(uv) of a homography: (H[:2, :2] - xy * H[2, :2]) / w
    jacobian = (h[None, :2, :2]-xy[:, :, None]*h[None, 2:3, :2])/w[:, :, None]
    lattice = book.spacing_mm*np.array([[1., .5], [0., SQRT3/2]])/book.page[:, None]
    return jacobian@lattice


def _measure_edges(stack, p, q, linear, k, mode):
    """`edge_measure` for arrays of edges: (ink difference, ink presence) per edge."""
    sides = [linear@(DIRS[(k+side) % 6]-DIRS[k]/2)[:, :, None] for side in (1, -1)]
    apex = [value[..., 0] for value in sides]
    centroids = np.stack([(p+q)/2+a/3 for a in apex], 1)
    background = stack.sample(centroids).mean(1)
    if isinstance(mode, tuple):
        t = np.array(mode)[None, :, None]
        near_p = stack.sample(p[:, None]*(1-t)+q[:, None]*t)
        near_q = stack.sample(q[:, None]*(1-t)+p[:, None]*t)
        values = (near_p, near_q)
    else:
        t, across = _arc_offsets()
        chord = p[:, None]*(1-t[None, :, None])+q[:, None]*t[None, :, None]
        values = tuple(stack.sample(chord+across[None, :, None]*a[:, None]) for a in apex)
    diff = np.median(values[0]-values[1], 1)
    presence = np.maximum(np.median(values[0], 1), np.median(values[1], 1))-background
    return diff, presence
