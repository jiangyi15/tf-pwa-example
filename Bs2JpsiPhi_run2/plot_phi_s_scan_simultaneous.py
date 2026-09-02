#!/usr/bin/env python3
"""Plot n_sigma and fitted-value histograms from phi_s_scan_results_simultaneous.txt
(SIMULTANEOUS fit toy results).

Results file is tab-separated; per parameter there are 4 columns:
    <param>_value, <param>_error, <param>_reference, <param>_n_sigma

Outputs:
    phi_s_scan_nsigma_hist_simultaneous.png  -- n_sigma histograms (one per parameter)
    phi_s_scan_value_hist_simultaneous.png   -- fitted-value histograms with a red
                                                dashed line at the reference value
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "phi_s_scan_results_simultaneous.txt"
OUT_NSIGMA = "phi_s_scan_nsigma_hist_simultaneous.png"
OUT_VALUE = "phi_s_scan_value_hist_simultaneous.png"
BINS_NSIGMA = np.arange(0, 6.5, 0.5)  # half-open bins in n_sigma

# Reference values for the red dashed line in the value histograms.
# Keep in sync with PARAMS/REFS + POLAR_PARAMS/POLAR_REFS in run_phi_s_scan_simultaneous.sh.
# NOTE: for Bs_delta_gamma the reference is for -Bs_delta_gamma (i.e.
#       DeltaGamma_s), since the value column already stores the negated
#       quantity.
REFS = {
    "Bs_poqi": -0.03,
    "Bs_delta_gamma": 0.08543,
    "Bs_gamma": 0.6614,
    "|A0|^2": 0.524176,
    "|A∥|^2": 0.225625,
    "|A⊥|^2": 0.250000,
    "δ∥ - δ0": 3.26,
    "δ⊥ - δ0": 3.08,
}

# Phase parameters whose fitted values are periodic in 2*pi: we wrap them
# into [ref - pi, ref + pi) so the histogram isn't split across the boundary.
PHASE_PARAMS = {"δ∥ - δ0", "δ⊥ - δ0"}

TWO_PI = 2.0 * math.pi


def wrap_near(value, centre):
    """Shift `value` by +-2*pi so it lies in [centre - pi, centre + pi)."""
    value = np.asarray(value, dtype=float)
    r = ((value - centre + math.pi) % TWO_PI) - math.pi
    return centre + r


if not os.path.exists(RESULTS):
    raise SystemExit(f"Result file not found: {RESULTS}")

with open(RESULTS) as f:
    header = f.readline().rstrip("\n").split("\t")

# find all n_sigma columns (== all parameters present in this run)
param_names = []
nsigma_cols = []
value_cols = []
for i, col in enumerate(header):
    if col.endswith("_n_sigma"):
        name = col[: -len("_n_sigma")]
        param_names.append(name)
        nsigma_cols.append(i)
        value_cols.append(header.index(f"{name}_value"))

if not param_names:
    raise SystemExit("No *_n_sigma columns found in header.")

matrix = np.genfromtxt(RESULTS, skip_header=1, delimiter="\t")
if matrix.ndim == 1:
    matrix = matrix.reshape(1, -1)
nrow = matrix.shape[0]

# ---- colours / grid layout (shared by both figures) -----------------
colors = plt.cm.tab10.colors
n_p = len(param_names)
ncol = min(3, n_p)
nrow_fig = math.ceil(n_p / ncol)


# =====================================================================
# Figure 1: n_sigma histograms (original behaviour)
# =====================================================================
fig, axes = plt.subplots(
    nrow_fig, ncol, figsize=(5 * ncol + 1, 4.5 * nrow_fig + 0.5), squeeze=False
)
axes = axes.flatten()

print(f"n_toy = {nrow}")
print(f"{'param':<20s}{'mean':>8s}{'median':>9s}{'  >2s%':>8s}{'  >3s%':>8s}")

bin_width = BINS_NSIGMA[1] - BINS_NSIGMA[0]

for idx, (name, ci) in enumerate(zip(param_names, nsigma_cols)):
    col = matrix[:, ci]
    nsigma = col[np.isfinite(col)]
    n_total = nsigma.size

    ax = axes[idx]
    ax.hist(
        nsigma,
        bins=BINS_NSIGMA,
        histtype="stepfilled",
        color=colors[idx % len(colors)],
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel(r"$n_\sigma$")
    ax.set_ylabel("Toys / {:.1f}".format(bin_width))
    ax.set_title(name)
    ax.grid(alpha=0.3)
    if n_total == 0:
        continue
    frac2 = np.mean(nsigma > 2) * 100
    frac3 = np.mean(nsigma > 3) * 100
    print(
        f"{name:<20s}{nsigma.mean():>8.3f}"
        f"{np.median(nsigma):>9.3f}{frac2:>7.1f}%{frac3:>7.1f}%"
    )

# hide unused axes
for j in range(idx + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    r"Distribution of $n_\sigma = |v - v_{\rm ref}| / \sigma$ over pseudo-experiments (simultaneous)",
    y=1.00,
)
fig.tight_layout()
fig.savefig(OUT_NSIGMA, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to: {OUT_NSIGMA}")


# =====================================================================
# Figure 2: fitted-value histograms with reference line
# =====================================================================
fig2, axes2 = plt.subplots(
    nrow_fig, ncol, figsize=(5 * ncol + 1, 4.5 * nrow_fig + 0.5), squeeze=False
)
axes2 = axes2.flatten()

print(f"\n{'param':<20s}{'mean':>10s}{'median':>10s}{'std':>9s}{'ref':>9s}")

for idx, (name, ci) in enumerate(zip(param_names, value_cols)):
    col = matrix[:, ci]
    values = col[np.isfinite(col)]
    n_total = values.size

    ax = axes2[idx]

    # reference (hardcoded; fall back to the *_reference column if missing)
    ref = REFS.get(name)
    if ref is None:
        ref_col = header.index(f"{name}_reference")
        ref = float(matrix[0, ref_col])

    # phase wrapping + fixed +/- pi range around the reference
    if name in PHASE_PARAMS:
        values = wrap_near(values, ref)
        lo = ref - math.pi
        hi = ref + math.pi
        bins = np.linspace(lo, hi, 31)
        ax.set_xlim(lo, hi)
        xlabel = f"{name}  [rad, wrapped to ref +/- pi]"
    else:
        # data-driven bins with a small padding that always includes the ref
        if n_total > 0:
            span = max(values.max() - values.min(), abs(ref) * 0.1, 1e-6)
            lo = min(values.min(), ref) - 0.1 * span
            hi = max(values.max(), ref) + 0.1 * span
            bins = np.linspace(lo, hi, 31)
            ax.set_xlim(lo, hi)
        else:
            bins = 20
        xlabel = name

    ax.hist(
        values,
        bins=bins,
        histtype="stepfilled",
        color=colors[idx % len(colors)],
        alpha=0.7,
        edgecolor="black",
    )
    # red dashed reference line
    ax.axvline(
        ref, color="red", linestyle="--", linewidth=1.8,
        label=f"ref = {ref:.4g}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("toys")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    if n_total > 1:
        print(
            f"{name:<20s}{values.mean():>10.4f}{np.median(values):>10.4f}"
            f"{values.std(ddof=1):>9.4f}{ref:>9.4f}"
        )
    elif n_total == 1:
        print(f"{name:<20s}{values[0]:>10.4f}{'--':>10s}{'--':>9s}{ref:>9.4f}")

# hide unused axes
for j in range(idx + 1, len(axes2)):
    axes2[j].axis("off")

fig2.suptitle("Fitted value distributions over pseudo-experiments (simultaneous)", y=1.00)
fig2.tight_layout()
fig2.savefig(OUT_VALUE, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to: {OUT_VALUE}")
