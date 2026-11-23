"""Shared, backend-independent analysis for the EPOCH LWFA benchmark.

Both analyze_sdf.py (raw SDF via the `sdfr` module) and analyze_openpmd.py
(converted openPMD via `openpmd_api`) feed identically shaped numpy arrays
into these functions, so the emitted statistics are directly comparable.
The comparison itself lives in compare_analysis.py.

All statistics are deterministic float64 reductions of the stored data;
SDF arrays arrive Fortran-ordered and openPMD chunks C-ordered, so sums may
differ at the last ulp - the comparator applies a relative tolerance.
"""

import numpy as np


def array_stats(arr):
    """Deterministic statistics of one field block."""
    a = np.asarray(arr, dtype=np.float64)
    absa = np.abs(a)
    flat_idx = int(absa.argmax(axis=None))
    return {
        "shape": [int(s) for s in a.shape],
        "sum": float(a.sum(dtype=np.float64)),
        "absmax": float(absa.max()),
        "absmax_index": [int(i) for i in np.unravel_index(flat_idx, a.shape)],
        "sumsq": float(np.multiply(a, a, dtype=np.float64).sum(
            dtype=np.float64)),
        "nan_count": int(np.isnan(a).sum()),
    }


def grid_stats(grid):
    """grid: sequence of 1D node arrays (one per axis)."""
    axes = [np.asarray(g, dtype=np.float64) for g in grid]
    origin = [float(a[0]) for a in axes]
    spacing = [float(a[1] - a[0]) if a.size > 1 else 0.0 for a in axes]
    end = [float(a[-1]) for a in axes]
    shape = [int(a.size) for a in axes]
    # uniformity check: max deviation of consecutive differences
    nonuniform = [
        float(np.abs(np.diff(a) - (a[-1] - a[0]) / max(a.size - 1,
                                                       1)).max())
        if a.size > 2 else 0.0 for a in axes
    ]
    return {"origin": origin, "spacing": spacing, "end": end,
            "shape": shape, "nonuniformity": nonuniform}


def particle_stats(pos, mom, w, ids=None):
    """pos/mom: (x, y, z) arrays of the same length; w: weights;
    ids: optional integer ids. EPOCH momenta are normalized (p / m c),
    so gamma = sqrt(1 + px^2 + py^2 + pz^2)."""
    x, y, z = (np.asarray(v, dtype=np.float64) for v in pos)
    ww = np.asarray(w, dtype=np.float64)
    n = int(x.size)
    assert y.size == n and z.size == n, "length mismatch"
    if mom is not None:
        px, py, pz = (np.asarray(v, dtype=np.float64) for v in mom)
        assert px.size == n, "length mismatch"
        gamma = np.sqrt(1.0 + px * px + py * py + pz * pz)
    else:
        gamma = np.ones(n, dtype=np.float64)
    out = {
        "n": n,
        "sum_w": float(ww.sum(dtype=np.float64)),
        "mean_w": float(ww.sum(dtype=np.float64) / n) if n else 0.0,
        "centroid": [
            float((x * ww).sum(dtype=np.float64)) if n else 0.0,
            float((y * ww).sum(dtype=np.float64)) if n else 0.0,
            float((z * ww).sum(dtype=np.float64)) if n else 0.0,
        ],
        "mean_gamma": float(gamma.sum(dtype=np.float64) / n) if n else 0.0,
        "max_gamma": float(gamma.max()) if n else 0.0,
        "p95_gamma": float(np.quantile(gamma, 0.95)) if n else 0.0,
        "sum_gamma": float(gamma.sum(dtype=np.float64)),
        "extent": [
            [float(x.min()), float(x.max())] if n else [0.0, 0.0],
            [float(y.min()), float(y.max())] if n else [0.0, 0.0],
            [float(z.min()), float(z.max())] if n else [0.0, 0.0],
        ],
    }
    if ids is not None:
        ii = np.asarray(ids)
        out["id_sum"] = int(ii.astype(np.int64).sum())
        out["id_min"] = int(ii.min()) if n else 0
        out["id_max"] = int(ii.max()) if n else 0
        out["id_unique"] = int(np.unique(ii).size)
    return out
