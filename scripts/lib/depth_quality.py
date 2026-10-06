"""Measured-depth admission and source-balanced fusion evidence. No motion API."""
from __future__ import annotations

import numpy as np


def filter_depth(depth, units, support=None, frames=None, temporal_mad=None, *, options=None):
    """Reject unreliable measurements; never fill or persist missing pixels.

    Defaults are conservative software filters, not physical accuracy claims.
    Legacy captures without temporal evidence retain an explicit unknown count.
    """
    opts = {} if options is None else options
    if not isinstance(opts, dict) or set(opts)-{
            "min_support_fraction", "max_temporal_mad_mm", "depth_edge_mm"}:
        raise ValueError("unknown depth quality settings")
    fraction = float(opts.get('min_support_fraction', .75))
    temporal_limit = float(opts.get('max_temporal_mad_mm', 1.0))*1e-3
    edge = float(opts.get('depth_edge_mm', 5.0))*1e-3
    if not (np.isfinite([units, fraction, temporal_limit, edge]).all()
            and units > 0 and 0 < fraction <= 1 and temporal_limit > 0 and edge > 0):
        raise ValueError('invalid depth quality settings')
    raw = np.asarray(depth)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError('depth quality requires a Z16 image')
    valid = (raw > 0) & (raw < 65535)
    initial = int(valid.sum())
    counts = {'invalid': int(raw.size-initial), 'low_support': 0, 'unstable': 0, 'edge': 0}
    if support is not None:
        support = np.asarray(support)
        if (support.shape != raw.shape or not np.issubdtype(support.dtype, np.integer)
                or frames is None or not 1 <= frames <= 255
                or np.any(support < 0) or np.any(support > frames)):
            raise ValueError('invalid temporal support evidence')
        keep = support >= int(np.ceil(frames*fraction))
        counts['low_support'] = int((valid & ~keep).sum())
        valid &= keep
    if temporal_mad is not None:
        mad = np.asarray(temporal_mad)
        if mad.shape != raw.shape or not np.isfinite(mad).all() or np.any(mad < 0):
            raise ValueError('invalid temporal depth deviation')
        keep = mad <= temporal_limit
        counts['unstable'] = int((valid & ~keep).sum())
        valid &= keep
    # Reject both sides of measured discontinuities, never wrap image borders.
    z = raw.astype(float)*units
    discontinuity = np.zeros_like(valid)
    for axis in (0, 1):
        lo, hi = [slice(None)]*2, [slice(None)]*2
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        jump = valid[lo] & valid[hi] & (np.abs(z[lo]-z[hi]) > edge)
        discontinuity[lo] |= jump
        discontinuity[hi] |= jump
    counts['edge'] = int((valid & discontinuity).sum())
    valid &= ~discontinuity
    neighbors = np.zeros_like(raw, dtype=np.uint8)
    padded_z = np.pad(z, 1, constant_values=np.nan)
    padded_valid = np.pad(valid, 1)
    radius = np.maximum(edge, z*.01)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        other = padded_z[1+dy:1+dy+raw.shape[0], 1+dx:1+dx+raw.shape[1]]
        seen = padded_valid[1+dy:1+dy+raw.shape[0], 1+dx:1+dx+raw.shape[1]]
        neighbors += seen & (np.abs(other-z) <= radius)
    counts['isolated'] = int((valid & (neighbors == 0)).sum())
    valid &= neighbors > 0
    return np.where(valid, raw, 0).astype(np.uint16), {
        'pixels': int(raw.size), 'accepted': int(valid.sum()), 'rejected': counts,
        'temporal_evidence': support is not None and temporal_mad is not None,
        'settings': {'min_support_fraction': fraction, 'max_temporal_mad_mm': temporal_limit*1e3,
                     'depth_edge_mm': edge*1e3},
    }


def balanced_cells(cell, heights, source, ncells, group_median):
    """One median per source/cell. Pixel density cannot increase source weight."""
    labels, ids = np.unique(np.asarray(source), return_inverse=True)
    combined = ids*ncells + cell
    med, count = group_median(combined, heights, ncells*len(labels))
    mad, _ = group_median(combined, np.abs(heights-med[combined]), ncells*len(labels))
    med, count, mad = (a.reshape(len(labels), ncells) for a in (med, count, mad))
    present = count > 0
    # Within-source scatter supplies a relative noise weight, bounded so a
    # quantized flat view never has infinite authority. This is not accuracy.
    weight = np.where(present, 1/np.maximum(np.nan_to_num(mad), .0005)**2, 0)
    weight /= np.maximum(weight.sum(axis=0), 1)
    value = np.sum(np.nan_to_num(med)*weight, axis=0)
    value[~present.any(axis=0)] = np.nan
    lower = np.min(np.where(present, med, np.inf), axis=0)
    upper = np.max(np.where(present, med, -np.inf), axis=0)
    disagreement = np.where(present.any(axis=0), upper-lower, 0)
    return value, present.sum(axis=0), disagreement
