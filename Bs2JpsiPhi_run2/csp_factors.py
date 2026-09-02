"""Csp factor loading and event-by-event lookup for Bs2JpsiPhi analysis.

The Csp factors are bin-by-bin correction factors for the phi(1020) -> KK
amplitude, computed from the ratio of the full (f0 + phi) integral over
the f0(980) component integral, weighted by the KK-pair efficiency map.
They are binned in m(K+K-) (a.k.a. X_M, in MeV) with the following bins:

    [990, 1008), [1008, 1016), [1016, 1020),
    [1020, 1024), [1024, 1032), [1032, 1050]

Pre-computed Csp factor values (one per bin) are read from per-year JSON
files located under ``Bs2JpsiPhi_run2/params/Csp/``.  Given the m(K+K-)
value of an event, the corresponding Csp factor can be looked up and then
multiplied with the per-event weight (``sw``).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

DEFAULT_MKK_BINS = np.array([990, 1008, 1016, 1020, 1024, 1032, 1050], dtype=np.float64)

# Absolute default path for the shipped JSON files.
_DEFAULT_CSP_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "params", "Csp"
)


def load_csp_factors(
    year: int,
    csp_dir: Optional[str] = None,
    json_filename: Optional[str] = None,
) -> List[Dict]:
    """Load the pre-computed CspFactors list for a given data-taking year.

    Parameters
    ----------
    year : int
        Data-taking year (e.g. 2015, 2016, 2017, 2018).
    csp_dir : str, optional
        Directory containing the ``CspFactors<year>.json`` files.
        Defaults to ``Bs2JpsiPhi_run2/params/Csp``.
    json_filename : str, optional
        Override the file name.  Defaults to ``CspFactors<year>.json``.

    Returns
    -------
    list of dict
        Each dictionary contains keys ``Name``, ``Value``, ``Error``,
        ``Bin_ll``, ``Bin_ul``.
    """
    if csp_dir is None:
        csp_dir = _DEFAULT_CSP_DIR
    if json_filename is None:
        json_filename = f"CspFactors{int(year)}.json"

    file_path = os.path.join(csp_dir, json_filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"CspFactors JSON file not found: {file_path}"
        )

    with open(file_path, "r") as fh:
        payload = json.load(fh)

    year_key = str(int(year))
    if year_key not in payload:
        raise KeyError(
            f"Year key '{year_key}' not found in {file_path}. "
            f"Available keys: {list(payload.keys())}"
        )

    factors = payload[year_key].get("CspFactors")
    if factors is None:
        raise KeyError(
            f"'CspFactors' key missing for year {year_key} in {file_path}"
        )

    return factors


def load_csp_factors_multi_year(
    years: Iterable[int],
    csp_dir: Optional[str] = None,
) -> Dict[int, List[Dict]]:
    """Load CspFactors for multiple years at once.

    Returns
    -------
    dict
        Mapping ``{year: csp_factors_list}``.
    """
    return {int(y): load_csp_factors(int(y), csp_dir=csp_dir) for y in years}


def load_csp_factors_baseline(
    csp_dir: Optional[str] = None,
    json_filename: str = "CspFactorsAll.json",
) -> List[Dict]:
    """Load the common (Baseline) CspFactors used for all data-taking years.

    Parameters
    ----------
    csp_dir : str, optional
        Directory containing the JSON file.  Defaults to
        ``Bs2JpsiPhi_run2/params/Csp``.
    json_filename : str
        File name of the Baseline CspFactors JSON.  Defaults to
        ``CspFactorsAll.json``.

    Returns
    -------
    list of dict
        Each dictionary contains keys ``Name``, ``Value``, ``Error``,
        ``Bin_ll``, ``Bin_ul``.
    """
    if csp_dir is None:
        csp_dir = _DEFAULT_CSP_DIR

    file_path = os.path.join(csp_dir, json_filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"CspFactors JSON file not found: {file_path}"
        )

    with open(file_path, "r") as fh:
        payload = json.load(fh)

    # The Baseline file uses the key "All".
    key = "All"
    if key not in payload:
        # Fallback: try the first available key.
        keys = list(payload.keys())
        if not keys:
            raise KeyError(f"No CspFactors entries found in {file_path}")
        key = keys[0]

    factors = payload[key].get("CspFactors")
    if factors is None:
        raise KeyError(
            f"'CspFactors' key missing in {file_path}"
        )

    return factors


def _build_lookup_table(csp_factors: List[Dict]):
    """Convert a CspFactors list into parallel arrays for vectorized lookup."""
    values = np.array([float(entry["Value"]) for entry in csp_factors], dtype=np.float64)
    bin_ll = np.array([float(entry["Bin_ll"]) for entry in csp_factors], dtype=np.float64)
    bin_ul = np.array([float(entry["Bin_ul"]) for entry in csp_factors], dtype=np.float64)
    # Sort by lower edge to guarantee a stable lookup.
    order = np.argsort(bin_ll)
    return bin_ll[order], bin_ul[order], values[order]


def get_csp_factor(mKK: float, csp_factors: List[Dict]) -> float:
    """Return the Csp factor for a single m(K+K-) value (MeV).

    Parameters
    ----------
    mKK : float
        Invariant mass of the K+K- system in MeV (same units as the
        ``X_M`` branch used in the Bs2JpsiPhi analysis).
    csp_factors : list of dict
        Pre-computed CspFactors for the target year.

    Returns
    -------
    float
        Csp factor.  Returns 1.0 if the event lies outside the covered
        mKK range (``[990, 1050] MeV``).
    """
    bin_ll, bin_ul, values = _build_lookup_table(csp_factors)
    return _lookup_scalar(mKK, bin_ll, bin_ul, values)


def _lookup_scalar(mKK, bin_ll, bin_ul, values):
    if mKK < bin_ll[0] or mKK > bin_ul[-1]:
        return 1.0
    for lo, hi, val in zip(bin_ll, bin_ul, values):
        if mKK < hi:
            return float(val)
    # Fallback (e.g. mKK exactly at the last bin upper edge).
    return float(values[-1])


def compute_csp_for_events(
    mKK_array: np.ndarray,
    year_array: Optional[np.ndarray] = None,
    csp_factors_by_year: Optional[Dict[int, List[Dict]]] = None,
    out_of_range_value: float = 1.0,
) -> np.ndarray:
    """Compute an event-by-event Csp factor array.

    If ``year_array`` and ``csp_factors_by_year`` are both ``None``, a
    single set of Baseline factors must be supplied via
    ``csp_factors_by_year={0: factors}`` — or use
    :func:`compute_csp_for_events_baseline` instead.

    Parameters
    ----------
    mKK_array : np.ndarray
        Per-event m(K+K-) values in MeV (shape ``(N,)``).
    year_array : np.ndarray, optional
        Per-event year identifiers (shape ``(N,)``).  Only integer years
        present in ``csp_factors_by_year`` are used; others fall back to
        ``out_of_range_value``.
    csp_factors_by_year : dict, optional
        Mapping ``{year: csp_factors_list}`` as returned by
        :func:`load_csp_factors_multi_year`.
    out_of_range_value : float
        Value assigned to events whose mKK falls outside the covered range
        of the loaded factors (default ``1.0``).

    Returns
    -------
    np.ndarray
        Per-event Csp factors with the same shape as ``mKK_array``.
    """
    mKK = np.asarray(mKK_array, dtype=np.float64).ravel()
    out = np.full(mKK.shape, float(out_of_range_value), dtype=np.float64)

    if year_array is None or csp_factors_by_year is None:
        return out

    year = np.asarray(year_array).ravel()

    for yr, factors in csp_factors_by_year.items():
        mask = (year == int(yr))
        if not np.any(mask):
            continue
        sub_mKK = mKK[mask]
        bin_ll, bin_ul, values = _build_lookup_table(factors)
        n_bins = len(values)
        # Vectorized lookup per sub-event.
        sub_out = np.full(sub_mKK.shape, float(out_of_range_value), dtype=np.float64)
        for i, (lo, hi, val) in enumerate(zip(bin_ll, bin_ul, values)):
            if i == n_bins - 1:
                in_bin = (sub_mKK >= lo) & (sub_mKK <= hi)
            else:
                in_bin = (sub_mKK >= lo) & (sub_mKK < hi)
            sub_out[in_bin] = val
        out[mask] = sub_out

    return out


def compute_csp_for_events_baseline(
    mKK_array: np.ndarray,
    csp_factors: List[Dict],
    out_of_range_value: float = 1.0,
) -> np.ndarray:
    """Compute event-by-event Csp factors using a single Baseline set.

    This applies the same Csp factors to all events regardless of year,
    as required by the analysis (common Csp factors for all data-taking
    years).

    Parameters
    ----------
    mKK_array : np.ndarray
        Per-event m(K+K-) values in MeV (shape ``(N,)``).
    csp_factors : list of dict
        Baseline CspFactors as returned by :func:`load_csp_factors_baseline`.
    out_of_range_value : float
        Value assigned to events whose mKK falls outside the covered range
        (default ``1.0``).

    Returns
    -------
    np.ndarray
        Per-event Csp factors with the same shape as ``mKK_array``.
    """
    mKK = np.asarray(mKK_array, dtype=np.float64).ravel()
    out = np.full(mKK.shape, float(out_of_range_value), dtype=np.float64)

    bin_ll, bin_ul, values = _build_lookup_table(csp_factors)
    n_bins = len(values)
    for i, (lo, hi, val) in enumerate(zip(bin_ll, bin_ul, values)):
        if i == n_bins - 1:
            # Last bin: inclusive on both ends [lo, hi].
            in_bin = (mKK >= lo) & (mKK <= hi)
        else:
            in_bin = (mKK >= lo) & (mKK < hi)
        out[in_bin] = val

    return out


def multiply_csp_into_weights(
    weights: np.ndarray,
    csp_factors: np.ndarray,
) -> np.ndarray:
    """Multiply the per-event weights by the per-event Csp factors.

    Both arrays must have the same shape.
    """
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(csp_factors, dtype=np.float64)
    if w.shape != c.shape:
        raise ValueError(
            f"Shape mismatch between weights ({w.shape}) and csp_factors ({c.shape})"
        )
    return w * c
