"""
Generate PHSP (Phase Space) samples for Bs->J/psi Phi analysis.

This script generates synthetic data for normalization calculation,
producing output files that correspond to the real data files from create_real_data.py.
"""
from tf_pwa.config_loader import ConfigLoader
import numpy as np
from tf_pwa import set_random_seed
import sys
import json
import argparse
import os

from angular_acceptance import AngularAcceptance
from time_acceptance import BSplineTimeAcceptance, load_time_acceptance


def parse_args():
    parser = argparse.ArgumentParser(description='Generate PHSP samples for Bs2JpsiPhi analysis')
    parser.add_argument('--seed', type=int, default=10, help='Random seed (default: 10)')
    parser.add_argument('--years', type=str, default='2015,2016,2017,2018',
                        help='Years to generate, comma-separated (default: 2015,2016,2017,2018)')
    parser.add_argument('--trigger', type=str, default='all',
                        choices=['all', 'unbiased', 'biased'],
                        help='Trigger type to generate (default: all)')
    parser.add_argument('--n-per-year', type=int, default=10000000,
                        help='Number of events per year (default: 10000000)')
    return parser.parse_args()

args = parse_args()

seed = args.seed
set_random_seed(seed+1000)

selected_years = [int(y.strip()) for y in args.years.split(',')]
selected_trigger = args.trigger
N_PER_YEAR = args.n_per_year

print(f"Selected years: {selected_years}")
print(f"Selected trigger: {selected_trigger}")
print(f"Events per year: {N_PER_YEAR}")
print(f"Random seed: {seed}")

config = ConfigLoader("config_data.yml")

years = [2015, 2016, 2017, 2018]

def load_angular_acceptance(year):
    json_path = f"params/angular_acceptance/{year}/angular_acceptance_omega_{year}_v4r1.json"
    return AngularAcceptance.from_json(json_path, year)

all_angles = []
all_t_exp = []
all_texp_weight = []
all_aacc_weight = []
all_tacc_weight = []
all_weight = []
all_trigger = []
all_year = []

all_tag = []
all_eta = []

t_min, t_max = 0.0, 15.0
gamma = 1.0

for year in years:
    if year not in selected_years:
        print(f"Skipping year {year} (not in selected years)")
        continue
    
    print(f"\nProcessing {year}...")
    
    try:
        knots, coeffs_unbiased, coeffs_biased = load_time_acceptance(year)
        assert len(coeffs_unbiased) == len(coeffs_biased)
        time_acc = BSplineTimeAcceptance(knots=knots, 
            coefficients_unbiased=coeffs_unbiased, coefficients_biased=coeffs_biased)
        angular_acc = load_angular_acceptance(year)
        print(f"Loaded omega-based angular acceptance for {year}")
        print(f"  c_unbiased: {angular_acc.c_unbiased}")
        print(f"  c_biased: {angular_acc.c_biased}")
        print(f"  norm_unbiased: {angular_acc.norm_unbiased}")
        print(f"  norm_biased: {angular_acc.norm_biased}")
        print(f"  Time acceptance knots: {knots}")
        print(f"  Time acceptance coeffs_unbiased: {coeffs_unbiased}")
        print(f"  Time acceptance coeffs_biased: {coeffs_biased}")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Acceptance params not found for {year}: {e}")
        print("Using default flat acceptance")
        time_acc = BSplineTimeAcceptance()
        angular_acc = AngularAcceptance()
    
    phi = np.random.random(N_PER_YEAR) * np.pi * 2 - np.pi
    c1 = np.random.random(N_PER_YEAR) * 2 - 1
    c2 = np.random.random(N_PER_YEAR) * 2 - 1
    
    angles = np.stack([np.arccos(c1), np.arccos(c2), phi], axis=-1)
    all_angles.append(angles)

    if selected_trigger == 'unbiased':
        trigger = np.zeros(N_PER_YEAR, dtype=np.int32)
    elif selected_trigger == 'biased':
        trigger = np.ones(N_PER_YEAR, dtype=np.int32)
    else:
        trigger = np.random.choice([0, 1], N_PER_YEAR, p=[0.8, 0.2])
    all_trigger.append(trigger)
    
    t_exp = -np.log(np.random.random(N_PER_YEAR) * (np.exp(-t_min*gamma) - np.exp(-t_max*gamma)) + np.exp(-t_max*gamma))/gamma
    texp_weight = np.exp(gamma * t_exp)
    
    cos_theta_K = c2
    cos_theta_mu = c1
    phi_K = phi
    
    batch_size = 250000
    n_events = len(cos_theta_K)
    ang_weight = np.zeros(n_events, dtype=np.float64)
    tim_weight = np.zeros(n_events, dtype=np.float64)
    
    for i in range(0, n_events, batch_size):
        end = min(i + batch_size, n_events)
        ang_weight[i:end] = angular_acc(
            cos_theta_K[i:end], cos_theta_mu[i:end], phi_K[i:end], 
            trigger[i:end] if hasattr(trigger, '__len__') else trigger
        )
        tim_weight[i:end] = time_acc(t_exp[i:end], 
            trigger[i:end] if hasattr(trigger, '__len__') else trigger
        )
        if (i // batch_size) % 10 == 0:
            print(f"    Processed {end}/{n_events} events")
    
    acc_weight = ang_weight * tim_weight
    
    all_t_exp.append(t_exp)
    all_texp_weight.append(texp_weight)
    all_aacc_weight.append(ang_weight)
    all_tacc_weight.append(tim_weight)
    all_weight.append(texp_weight * acc_weight)
    all_year.append(np.full(N_PER_YEAR, year, dtype=np.int32))
    
    tag = np.random.choice([-1, 0, 1], N_PER_YEAR, p=[0.4, 0.2, 0.4]).astype(np.int32)
    eta = np.random.random(N_PER_YEAR) * 0.5
    
    all_tag.append(tag)
    all_eta.append(eta)
    
    print(f"Generated {N_PER_YEAR} events for {year}")

print("\nMerging data...")

angles_merged = np.concatenate(all_angles, axis=0)
t_exp_merged = np.concatenate(all_t_exp, axis=0)
texp_weight_merged = np.concatenate(all_texp_weight, axis=0)
aacc_weight_merged = np.concatenate(all_aacc_weight, axis=0)
tacc_weight_merged = np.concatenate(all_tacc_weight, axis=0)
weight_merged = np.concatenate(all_weight, axis=0)
trigger_merged = np.concatenate(all_trigger, axis=0)
year_merged = np.concatenate(all_year, axis=0)

tag_merged = np.concatenate(all_tag, axis=0)
eta_merged = np.concatenate(all_eta, axis=0)

total_events = len(angles_merged)
print(f"Total events: {total_events}")

print("\n=== Weight validation ===")
print(f"Weight stats before filtering:")
print(f"  Min: {np.min(weight_merged):.6f}")
print(f"  Max: {np.max(weight_merged):.6f}")
print(f"  Mean: {np.mean(weight_merged):.6f}")
print(f"  Std: {np.std(weight_merged):.6f}")
print(f"  NaN count: {np.sum(np.isnan(weight_merged))}")
print(f"  Inf count: {np.sum(np.isinf(weight_merged))}")
print(f"  Negative count: {np.sum(weight_merged < 0)}")

valid_mask = ~np.isnan(weight_merged) & ~np.isinf(weight_merged)

invalid_count = total_events - np.sum(valid_mask)
print(f"\nOutlier detection results:")
print(f"  Total events: {total_events}")
print(f"  Valid events: {np.sum(valid_mask)}")
print(f"  Invalid/Outlier events: {invalid_count} ({invalid_count/total_events*100:.2f}%)")

if invalid_count > 0:
    outliers = weight_merged[~valid_mask]
    print(f"  Outlier weight range: [{np.min(outliers):.6f}, {np.max(outliers):.6f}]")

angles_merged = angles_merged[valid_mask]
t_exp_merged = t_exp_merged[valid_mask]
texp_weight_merged = texp_weight_merged[valid_mask]
aacc_weight_merged = aacc_weight_merged[valid_mask]
tacc_weight_merged = tacc_weight_merged[valid_mask]
weight_merged = weight_merged[valid_mask]
trigger_merged = trigger_merged[valid_mask]
year_merged = year_merged[valid_mask]

tag_merged = tag_merged[valid_mask]
eta_merged = eta_merged[valid_mask]

print(f"\nWeight stats after filtering:")
print(f"  Min: {np.min(weight_merged):.6f}")
print(f"  Max: {np.max(weight_merged):.6f}")
print(f"  Mean: {np.mean(weight_merged):.6f}")
print(f"  Std: {np.std(weight_merged):.6f}")
print(f"  Remaining events: {len(weight_merged)}")

years_suffix = '_'.join(str(y) for y in selected_years)
trigger_suffix = selected_trigger
file_suffix = f"{years_suffix}_{trigger_suffix}"

total_events = len(angles_merged)

print("\n" + "="*60)
print("Saving PHSP Output Files")
print("="*60)

print("\nSaving time data ...")
np.save(f"phsp_t_exp_{file_suffix}.npy", t_exp_merged)

print("Saving tagging data ...")
np.save(f"phsp_tag_{file_suffix}.npy", tag_merged)
np.save(f"phsp_eta_{file_suffix}.npy", eta_merged)

print("Saving other data...")
np.save(f"phsp_weight_{file_suffix}.npy", weight_merged)
np.save(f"phsp_angles_{file_suffix}.npy", angles_merged)
np.save(f"phsp_trigger_{file_suffix}.npy", trigger_merged)
np.save(f"phsp_year_{file_suffix}.npy", year_merged)

print(f"\nGenerated {total_events} PHSP events")
print(f"Time range: {np.min(t_exp_merged):.2f} - {np.max(t_exp_merged):.2f} ps")
print(f"Tag range: {np.min(tag_merged):.2f} - {np.max(tag_merged):.2f}")
print(f"Eta range: {np.min(eta_merged):.2f} - {np.max(eta_merged):.2f}")
print(f"Output files suffix: {file_suffix}")

print("\nGenerated output files:")
print(f"  Time: phsp_t_exp_{file_suffix}.npy")
print(f"  Tagging: phsp_tag_{file_suffix}.npy, phsp_eta_{file_suffix}.npy")
print(f"  Other: phsp_weight_{file_suffix}.npy, phsp_angles_{file_suffix}.npy, phsp_trigger_{file_suffix}.npy, phsp_year_{file_suffix}.npy")

for y in years:
    y_mask = year_merged == y
    print(f"Year {y}: {np.sum(y_mask)} events")


def plot_distributions():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    cos_theta_mu = np.cos(angles_merged[:, 0])
    cos_theta_K = np.cos(angles_merged[:, 1])
    phi_K = angles_merged[:, 2]

    t_exp = t_exp_merged

    unbiased_mask = trigger_merged == 0
    biased_mask = trigger_merged == 1

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)

    # Row 1: Basic variable distributions (time, angles)
    ax_time = fig.add_subplot(gs[0, 0])
    ax_cosK = fig.add_subplot(gs[0, 1])
    ax_cosMu = fig.add_subplot(gs[0, 2])

    # Row 2: phi, eta distributions, time acceptance vs t
    ax_phi = fig.add_subplot(gs[1, 0])
    ax_eta = fig.add_subplot(gs[1, 1])
    ax_tacc_vs_t = fig.add_subplot(gs[1, 2])

    # Row 3: Angular acceptance vs angles
    ax_aacc_cosK = fig.add_subplot(gs[2, 0])
    ax_aacc_cosMu = fig.add_subplot(gs[2, 1])
    ax_aacc_phi = fig.add_subplot(gs[2, 2])

    # ========== Row 1: Basic variable distributions ==========
    ax_time.hist(t_exp, bins=100, weights=weight_merged, density=True, alpha=0.7, color='blue', label='Weighted')
    ax_time.hist(t_exp, bins=100, density=True, alpha=0.3, color='gray', label='Unweighted')
    ax_time.set_xlabel('t (ps)')
    ax_time.set_ylabel('Density')
    ax_time.set_title('Time distribution (weighted vs unweighted)')
    ax_time.set_yscale('log')
    ax_time.legend(loc='upper right')

    ax_cosK.hist(cos_theta_K, bins=50, weights=weight_merged, density=True, alpha=0.7, color='red', label='Weighted')
    ax_cosK.hist(cos_theta_K, bins=50, density=True, alpha=0.3, color='gray', label='Unweighted')
    ax_cosK.set_xlabel('cos(theta_K)')
    ax_cosK.set_ylabel('Density')
    ax_cosK.set_title('cos(theta_K) distribution')
    ax_cosK.legend(loc='upper right')

    ax_cosMu.hist(cos_theta_mu, bins=50, weights=weight_merged, density=True, alpha=0.7, color='green', label='Weighted')
    ax_cosMu.hist(cos_theta_mu, bins=50, density=True, alpha=0.3, color='gray', label='Unweighted')
    ax_cosMu.set_xlabel('cos(theta_mu)')
    ax_cosMu.set_ylabel('Density')
    ax_cosMu.set_title('cos(theta_mu) distribution')
    ax_cosMu.legend(loc='upper right')

    # ========== Row 2: phi and eta distributions ==========
    ax_phi.hist(phi_K, bins=50, weights=weight_merged, density=True, alpha=0.7, color='orange', label='Weighted')
    ax_phi.hist(phi_K, bins=50, density=True, alpha=0.3, color='gray', label='Unweighted')
    ax_phi.set_xlabel('phi_K (rad)')
    ax_phi.set_ylabel('Density')
    ax_phi.set_title('phi_K distribution')
    ax_phi.legend(loc='upper right')
    
    # Eta distribution (weighted)
    ax_eta.hist(eta_merged, bins=50, density=True, alpha=0.3, color='brown', label='Unweighted')
    ax_eta.set_xlabel('Eta')
    ax_eta.set_ylabel('Density')
    ax_eta.set_title('Eta distribution (weighted vs unweighted)')
    ax_eta.legend(loc='upper right')

    # Time acceptance vs t (by trigger type)
    for mask, color, label in [(unbiased_mask, 'blue', 'Unbiased'), (biased_mask, 'red', 'Biased')]:
        if np.any(mask):
            ax_tacc_vs_t.scatter(t_exp[mask], tacc_weight_merged[mask], alpha=0.3, s=1, color=color, label=label)
    ax_tacc_vs_t.set_xlabel('t (ps)')
    ax_tacc_vs_t.set_ylabel('Time acceptance')
    ax_tacc_vs_t.set_title('Time acceptance vs t by trigger type')
    ax_tacc_vs_t.set_xscale('log')
    ax_tacc_vs_t.set_xlim(0.3, 11)
    ax_tacc_vs_t.set_ylim(0.5, 2.5)
    ax_tacc_vs_t.legend(loc='upper right')

    # ========== Row 3: Angular acceptance 1D projections (by trigger type) ==========
    n_bins = 50
    
    # A(cosθ_K) = ∫∫ A(cosθ_μ, cosθ_K, φ_K) d(cosθ_μ) dφ_K
    cosK_bins = np.linspace(-1, 1, n_bins + 1)
    cosK_centers = (cosK_bins[:-1] + cosK_bins[1:]) / 2
    for mask, color, label in [(unbiased_mask, 'blue', 'Unbiased'), (biased_mask, 'red', 'Biased')]:
        if np.any(mask):
            proj, _ = np.histogram(cos_theta_K[mask], bins=cosK_bins, weights=aacc_weight_merged[mask])
            count, _ = np.histogram(cos_theta_K[mask], bins=cosK_bins)
            proj = np.where(count > 0, proj / count, 0.0)
            ax_aacc_cosK.plot(cosK_centers, proj, color=color, linewidth=2, label=label)
    ax_aacc_cosK.set_xlabel('cos(theta_K)')
    ax_aacc_cosK.set_ylabel('Angular acceptance (projected)')
    ax_aacc_cosK.set_title('Angular acceptance vs cos(theta_K)')
    ax_aacc_cosK.legend(loc='upper right')
    ax_aacc_cosK.set_ylim(0.5, 1.5)
    
    # A(cosθ_μ) = ∫∫ A(cosθ_μ, cosθ_K, φ_K) d(cosθ_K) dφ_K
    cosMu_bins = np.linspace(-1, 1, n_bins + 1)
    cosMu_centers = (cosMu_bins[:-1] + cosMu_bins[1:]) / 2
    for mask, color, label in [(unbiased_mask, 'green', 'Unbiased'), (biased_mask, 'red', 'Biased')]:
        if np.any(mask):
            proj, _ = np.histogram(cos_theta_mu[mask], bins=cosMu_bins, weights=aacc_weight_merged[mask])
            count, _ = np.histogram(cos_theta_mu[mask], bins=cosMu_bins)
            proj = np.where(count > 0, proj / count, 0.0)
            ax_aacc_cosMu.plot(cosMu_centers, proj, color=color, linewidth=2, label=label)
    ax_aacc_cosMu.set_xlabel('cos(theta_mu)')
    ax_aacc_cosMu.set_ylabel('Angular acceptance (projected)')
    ax_aacc_cosMu.set_title('Angular acceptance vs cos(theta_mu)')
    ax_aacc_cosMu.legend(loc='upper right')
    ax_aacc_cosMu.set_ylim(0.5, 1.5)
    
    # A(phi_K) = ∫∫ A(cosθ_μ, cosθ_K, φ_K) d(cosθ_μ) d(cosθ_K)
    phi_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
    for mask, color, label in [(unbiased_mask, 'orange', 'Unbiased'), (biased_mask, 'red', 'Biased')]:
        if np.any(mask):
            proj, _ = np.histogram(phi_K[mask], bins=phi_bins, weights=aacc_weight_merged[mask])
            count, _ = np.histogram(phi_K[mask], bins=phi_bins)
            proj = np.where(count > 0, proj / count, 0.0)
            ax_aacc_phi.plot(phi_centers, proj, color=color, linewidth=2, label=label)
    ax_aacc_phi.set_xlabel('phi_K (rad)')
    ax_aacc_phi.set_ylabel('Angular acceptance (projected)')
    ax_aacc_phi.set_title('Angular acceptance vs phi_K')
    ax_aacc_phi.legend(loc='upper right')
    ax_aacc_phi.set_ylim(0.5, 1.5)

    plt.tight_layout()
    plt.savefig(f'phsp_distributions_{file_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGenerated plot: phsp_distributions_{file_suffix}.png")


plot_distributions()
