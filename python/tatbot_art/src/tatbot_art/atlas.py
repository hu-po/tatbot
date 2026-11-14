"""The map of the wall: every design placed by what it looks like and what made it.

Two designs are close when their genes agree (style, subject, ink load, fill) and when their Flow
images look alike (an OKLab thumbnail and a chroma-weighted hue histogram). Classical MDS of that
distance lays each neighbourhood (Ward-linkage clusters) out as an island, the islands placed by how
far apart their designs are, and each design links to its nearest few. The page relaxes the layout with a force simulation and
draws lineage from each document's own `parents` and `study_of`; this module only measures.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .color import from_8bit, oklab

THUMB = 12        # OKLab thumbnail side, cells
HUES = 12         # hue histogram bins, plus one for ink with no chroma
NEAR = 5          # neighbours each design links to
GENE_WEIGHT = 0.55
GENE_PARTS = {"style": 2.0, "subject": 1.5, "palette": 1.5, "fill": 0.5}


def descriptor(image: Path) -> np.ndarray:
    """What a design looks like: a coarse OKLab picture and where its colour sits on the hue wheel."""
    with Image.open(image) as im:
        rgb = im.convert("RGB")
        small = np.asarray(rgb.resize((THUMB, THUMB), Image.Resampling.BOX))
        mid = np.asarray(rgb.resize((64, 64), Image.Resampling.BOX))
    thumb = oklab(from_8bit(small)).reshape(-1)
    lab = oklab(from_8bit(mid)).reshape(-1, 3)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    ink = 1.0 - lab[:, 0]  # how far from white paper
    bins = ((np.arctan2(lab[:, 2], lab[:, 1]) + math.pi) / (2 * math.pi) * HUES).astype(int) % HUES
    hist = np.bincount(bins, weights=chroma, minlength=HUES)
    grey = np.array([np.sum(ink * (chroma < 0.04))])
    colour = np.concatenate([hist, grey])
    colour = colour / (colour.sum() or 1.0)
    return np.concatenate([thumb, 4.0 * colour])


def _inks(doc: dict) -> frozenset[str]:
    return frozenset(i["id"] if isinstance(i, dict) else i for i in doc.get("inks", []))


def gene_distance(a: dict, b: dict, family: dict[str, str]) -> float:
    """0 for the same choices, 1 for nothing shared; styles of one separation family are half apart."""
    ga, gb = a.get("genes", {}), b.get("genes", {})
    style = 0.0 if ga.get("style") == gb.get("style") else (
        0.5 if family.get(ga.get("style")) and family.get(ga.get("style")) == family.get(gb.get("style")) else 1.0)
    ia, ib = _inks(a), _inks(b)
    palette = 1.0 - len(ia & ib) / (len(ia | ib) or 1)
    parts = {"style": style, "subject": float(ga.get("subject") != gb.get("subject")), "palette": palette,
             "fill": float(ga.get("fill") != gb.get("fill"))}
    return sum(GENE_PARTS[k] * v for k, v in parts.items()) / sum(GENE_PARTS.values())


def distances(ids: list[str], docs: dict[str, dict], looks: dict[str, np.ndarray], family: dict[str, str]) -> np.ndarray:
    """Gene and visual distance blended; the visual part scaled so its 95th percentile is 1."""
    n = len(ids)
    genes = np.array([[gene_distance(docs[a], docs[b], family) for b in ids] for a in ids]) if n else np.zeros((0, 0))
    vecs = np.stack([looks[i] for i in ids]) if n else np.zeros((0, 1))
    seen = np.sqrt(((vecs[:, None, :] - vecs[None, :, :]) ** 2).sum(-1))
    off = seen[~np.eye(n, dtype=bool)]
    scale = float(np.percentile(off, 95)) if off.size else 1.0
    return GENE_WEIGHT * genes + (1 - GENE_WEIGHT) * np.clip(seen / (scale or 1.0), 0, 1)


def mds(d: np.ndarray) -> np.ndarray:
    """Classical MDS to two dimensions, scaled into [-1, 1]."""
    n = len(d)
    if n < 3:
        return np.zeros((n, 2))
    j = np.eye(n) - 1.0 / n
    b = -0.5 * j @ (d ** 2) @ j
    vals, vecs = np.linalg.eigh(b)
    top = np.argsort(vals)[::-1][:2]
    xy = vecs[:, top] * np.sqrt(np.maximum(vals[top], 1e-9))
    return xy / (np.abs(xy).max() or 1.0)


def islands(d: np.ndarray, cl: np.ndarray, spread: float = 0.055, rounds: int = 200) -> np.ndarray:
    """Two-level layout: each neighbourhood an island placed by MDS of the mean distance between
    neighbourhoods, its designs laid out inside it by their own MDS, islands pushed apart until none overlap."""
    k = int(cl.max()) + 1 if len(cl) else 0
    if k < 2:
        return mds(d)
    members = [np.flatnonzero(cl == c) for c in range(k)]
    between = np.array([[d[np.ix_(a, b)].mean() if i != j else 0.0 for j, b in enumerate(members)]
                        for i, a in enumerate(members)])
    centre = mds(between) if k > 2 else np.array([[-0.5, 0.0], [0.5, 0.0]])
    size = np.array([spread * math.sqrt(len(m)) + 0.04 for m in members])
    for _ in range(rounds):  # separate overlapping islands, larger ones moving less
        moved = False
        for i in range(k):
            for j in range(i + 1, k):
                gap = centre[j] - centre[i]
                dist = float(np.hypot(*gap)) or 1e-6
                need = size[i] + size[j] + 0.05
                if dist < need:
                    push = (need - dist) * gap / dist
                    wi, wj = size[j] / (size[i] + size[j]), size[i] / (size[i] + size[j])
                    centre[i] -= push * wi
                    centre[j] += push * wj
                    moved = True
        if not moved:
            break
    xy = np.zeros((len(d), 2))
    for c, m in enumerate(members):
        local = mds(d[np.ix_(m, m)]) if len(m) > 2 else np.array([[-0.5, 0.0], [0.5, 0.0]])[:len(m)]
        xy[m] = centre[c] + local * size[c]
    return xy / (np.abs(xy).max() or 1.0)


def clusters(d: np.ndarray, k: int | None = None) -> np.ndarray:
    """Ward-linkage neighbourhoods, about one per ten designs (4 to 20). Average linkage left one
    neighbourhood holding a third of the wall; Ward keeps them comparable in size."""
    n = len(d)
    if n < 3:
        return np.zeros(n, dtype=int)
    k = k or max(4, min(20, round(n / 10)))
    z = linkage(squareform(d, checks=False), method="ward")
    return fcluster(z, t=min(k, n), criterion="maxclust") - 1


def graph(docs: dict[str, dict], looks: dict[str, np.ndarray], family: dict[str, str], totals: dict[str, int],
          made: str) -> dict:
    """The page's map document: node positions and clusters, and each design's nearest neighbours.

    Only designs are measured and clustered. A fill-study variant shares its source's image and all its
    genes but one, so it takes its source's neighbourhood and sits in a small ring around it; it takes no
    neighbour slots and does not sway a neighbourhood's label.
    """
    base = sorted(i for i in docs if i in looks and not docs[i].get("study_of"))
    variants = sorted(i for i in docs if i in looks and docs[i].get("study_of") in base)
    d = distances(base, docs, looks, family)
    cl = clusters(d)
    xy = islands(d, cl)
    at = {i: k for k, i in enumerate(base)}
    ring: dict[str, list[str]] = {}
    for v in variants:
        ring.setdefault(docs[v]["study_of"], []).append(v)
    ids, pos, group = list(base), [list(p) for p in xy], [int(c) for c in cl]
    for src, vs in ring.items():
        for j, v in enumerate(vs):
            a = 2 * math.pi * j / len(vs)
            ids.append(v)
            pos.append([xy[at[src]][0] + 0.03 * math.cos(a), xy[at[src]][1] + 0.03 * math.sin(a)])
            group.append(int(cl[at[src]]))
    edges = set()
    for a in range(len(base)):
        order = [b for b in np.argsort(d[a]) if b != a][:NEAR]
        edges.update((min(a, b), max(a, b)) for b in order)
    groups = []
    for c in range(int(cl.max()) + 1 if len(cl) else 0):
        members = [base[k] for k in np.flatnonzero(cl == c)]
        styles = Counter(docs[m]["genes"]["style"] for m in members).most_common(2)
        label = styles[0][0] if styles[0][1] * 2 > len(members) or len(styles) == 1 else f"{styles[0][0]} / {styles[1][0]}"
        groups.append({"label": label, "size": len(members), "score": sum(totals.get(m, 0) for m in members)})
    # `cluster` and the pairs in `near` index into `ids`, which keeps the document small.
    return {"schema": "tatbot.art-map/1", "made": made, "ids": ids,
            "xy": [[round(float(x), 4), round(float(y), 4)] for x, y in pos], "cluster": group,
            "clusters": groups, "near": [[int(a), int(b), round(float(1 - d[a, b]), 3)] for a, b in sorted(edges)]}
