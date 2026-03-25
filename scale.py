"""
xiangxiang nov 13 2025     
"""

import numpy as np
import umap


def _unique_sorted_ints(vals):
    """
    Take an array-like of numbers, round to int, remove duplicates, sort.
    Returns a Python list of ints.
    """
    vals = np.asarray(vals).astype(int)
    vals = np.unique(vals)
    vals = np.sort(vals)
    return vals.tolist()


def _auto_caps(n_samples, min_cap=None, max_cap=None):
    """
    Resolve min_cap / max_cap if 'auto' or None.

    n_samples: number of cells (rows of X)

    Returns:
        min_cap, max_cap (both ints)
    """
    # default min_cap
    if min_cap is None or (isinstance(min_cap, str) and min_cap.lower() == "auto"):
        # at least 5, but not more than n_samples-1
        min_cap = max(5, min(15, n_samples - 1))
    else:
        min_cap = int(min_cap)

    # default max_cap
    if max_cap is None or (isinstance(max_cap, str) and max_cap.lower() == "auto"):
        # something like up to 1/4 of n_samples, but at least 15
        max_cap = max(15, min(n_samples // 4, n_samples - 1))
    else:
        max_cap = int(max_cap)

    # ensure consistency
    if max_cap < min_cap:
        max_cap = min_cap

    # cannot exceed n_samples-1
    max_cap = min(max_cap, n_samples - 1)

    return min_cap, max_cap


def generate_scales(n_samples, n_scales, mode="lin", min_cap=None, max_cap=None):
    """
    Generate a monotonically increasing list of 'n_neighbors' values.
      mode ∈ {"lin","log","hybrid","sqrt","power{p}','inv"}:
        - lin:    evenly spaced in [min_cap, max_cap]
        - log:    logarithmically spaced (denser at small neighbors)
        - hybrid: mix of small local (log-like), medium linear, and a few large caps
        - sqrt:   k(t) = min_cap + (max_cap-min_cap) * t^(1/2)
        - power{p}: k(t) = min_cap + (max_cap-min_cap) * t^p, e.g. "power0.4"
        - inv:    inverse-shaped schedule via normalized 1/(t+eps)
    Caps "auto" are resolved based on n_samples via _auto_caps().
    """
    n_scales = int(n_scales)
    mode = str(mode).lower()
    min_cap, max_cap = _auto_caps(n_samples, min_cap, max_cap)

    if n_scales <= 0:
        raise ValueError("n_scales must be >= 1")

    def _quantize_and_unique(vals):
        return _unique_sorted_ints(np.round(vals))

    if mode == "lin":
        vals = np.linspace(min_cap, max_cap, num=n_scales)
        scales = _quantize_and_unique(vals)

    elif mode == "log":
        a, b = np.log(max(min_cap, 1)), np.log(max_cap)
        vals = np.exp(np.linspace(a, b, num=n_scales))
        scales = _quantize_and_unique(vals)

    elif mode == "sqrt":
        t = np.linspace(0.0, 1.0, num=n_scales)
        vals = min_cap + (max_cap - min_cap) * np.sqrt(t)
        scales = _quantize_and_unique(vals)

    elif mode.startswith("power"):
        import re
        m = re.search(r"(power|pow)\s*([0-9]*\.?[0-9]+)", mode)
        p = float(m.group(2)) if m else 0.5
        p = max(1e-3, min(5.0, p))
        t = np.linspace(0.0, 1.0, num=n_scales)
        vals = min_cap + (max_cap - min_cap) * (t ** p)
        scales = _quantize_and_unique(vals)

    elif mode == "inv":
        t = np.linspace(0.0, 1.0, num=n_scales)
        eps = 1e-6
        f0 = 1.0 / (0.0 + eps)
        f1 = 1.0 / (1.0 + eps)
        f = 1.0 / (t + eps)
        g = (f0 - f) / (f0 - f1)
        vals = min_cap + (max_cap - min_cap) * g
        scales = _quantize_and_unique(vals)

    elif mode == "hybrid":
        k_small = max(2, int(round(0.4 * n_scales)))
        k_mid   = max(2, int(round(0.4 * n_scales)))
        k_large = max(1, n_scales - k_small - k_mid)

        a, b = np.log(max(min_cap, 1)), np.log(max_cap)
        small_vals = np.exp(
            np.linspace(a, np.log(min_cap + (max_cap - min_cap) * 0.35), num=k_small)
        )

        mid_start = min_cap + (max_cap - min_cap) * 0.25
        mid_end   = min_cap + (max_cap - min_cap) * 0.75
        mid_vals  = np.linspace(mid_start, mid_end, num=k_mid)

        large_start = min_cap + (max_cap - min_cap) * 0.65
        large_vals  = np.linspace(large_start, max_cap, num=k_large)

        vals = np.concatenate([small_vals, mid_vals, large_vals])
        scales = _quantize_and_unique(vals)

        while len(scales) < n_scales:
            cand = int(np.clip(scales[-1] + 1, min_cap, max_cap))
            if cand not in scales:
                scales.append(cand)
            else:
                break
        scales = scales[:n_scales]

    else:
        raise ValueError("mode must be one of {'lin','log','hybrid','sqrt','power{p}','inv'}")

    upper_bound = max(2, min(max_cap, n_samples - 1))
    scales = [int(np.clip(s, 2, upper_bound)) for s in scales]
    scales = _unique_sorted_ints(scales)
    if len(scales) == 0:
        scales = [min(upper_bound, max(2, min_cap))]
    while len(scales) < n_scales and scales[-1] < upper_bound:
        nxt = scales[-1] + 1
        if nxt not in scales:
            scales.append(nxt)
        else:
            break
    return scales