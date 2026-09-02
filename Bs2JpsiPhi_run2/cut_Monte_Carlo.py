import numpy as np
import sys
import argparse
import json

from time_acceptance import BSplineTimeAcceptance, load_time_acceptance


def parse_args():
    parser = argparse.ArgumentParser(description='Generate smeared time for MC with time acceptance via rejection sampling')
    parser.add_argument('--seed', type=int, default=10, help='Random seed (default: 10)')
    parser.add_argument('--years', type=str, default='2015,2016,2017,2018',
                        help='Years to process, comma-separated (default: 2015,2016,2017,2018)')
    parser.add_argument('--trigger', type=str, default='all', choices=['all', 'unbiased', 'biased'],
                        help='Trigger type (default: all)')
    return parser.parse_args()


args = parse_args()

seed = args.seed
selected_years = [int(y.strip()) for y in args.years.split(',')]
selected_trigger = args.trigger

np.random.seed(seed + 1000)

print(f"Selected years: {selected_years}")
print(f"Selected trigger: {selected_trigger}")
print(f"Random seed: {seed}")

years_suffix = '_'.join(str(y) for y in selected_years)
file_suffix = f"{years_suffix}_{selected_trigger}"

# Load MC trigger and year to determine target size
mc_trigger = np.load(f"MC_trigger_{file_suffix}.npy")
mc_year = np.load(f"MC_year_{file_suffix}.npy")
n_target = len(mc_trigger)

print(f"\nTarget events (from MC_trigger): {n_target}")

# Load real data resolution
resolution = np.load(f"data_t_resolution_{file_suffix}.npy")
print(f"Loaded data resolution: {len(resolution)} values")
print(f"  Resolution range: [{np.min(resolution):.6f}, {np.max(resolution):.6f}] ps")

# Time generation parameters (same as generate_phsp_fast.py)
t_min, t_max = 0.0, 15.0
gamma = 1.0


def generate_exponential_time(n):
    """Generate exponential decay time (same as generate_phsp_fast.py)"""
    u = np.random.random(n)
    return -np.log(u * (np.exp(-t_min * gamma) - np.exp(-t_max * gamma)) + np.exp(-t_max * gamma)) / gamma


# Results arrays
t_smear = np.zeros(n_target, dtype=np.float64)
t_res = np.zeros(n_target, dtype=np.float64)
texp_weight = np.zeros(n_target, dtype=np.float64)

print("\n" + "=" * 60)
print("Generating smeared time with time acceptance (rejection sampling)")
print("=" * 60)

for y in selected_years:
    year_mask = mc_year == y
    n_year = np.sum(year_mask)
    if n_year == 0:
        print(f"  Skipping year {y}: no events")
        continue

    print(f"\n  Year {y}: {n_year} events")

    # Load time acceptance for this year
    try:
        knots, coeffs_unbiased, coeffs_biased = load_time_acceptance(y)
        time_acc = BSplineTimeAcceptance(knots=knots, coefficients_unbiased=coeffs_unbiased,
                                         coefficients_biased=coeffs_biased)
        print(f"    Loaded time acceptance: knots={knots}")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"    Warning: time acceptance not found for {y}: {e}")
        print(f"    Using flat acceptance (accept all)")
        time_acc = None

    # Pre-compute max acceptance on a grid for rejection sampling normalization
    if time_acc is not None:
        t_grid = np.linspace(t_min, t_max, 10000)
        acc_grid_u = time_acc(t_grid, np.zeros(len(t_grid)))
        acc_grid_b = time_acc(t_grid, np.ones(len(t_grid)))
        max_acc = max(np.max(acc_grid_u), np.max(acc_grid_b))
        if max_acc <= 0:
            max_acc = 1.0
        print(f"    Max acceptance: {max_acc:.6f}")
    else:
        max_acc = 1.0

    # Process per trigger type
    for trig_val in [0, 1]:
        trig_mask = year_mask & (mc_trigger == trig_val)
        n_trig = np.sum(trig_mask)
        if n_trig == 0:
            continue

        trig_name = "unbiased" if trig_val == 0 else "biased"
        print(f"    {trig_name} trigger: {n_trig} events")

        accepted_t = []
        accepted_res = []
        accepted_texp = []
        n_generated = 0
        n_accepted_total = 0
        max_iter = 100

        for iteration in range(max_iter):
            if len(accepted_t) >= n_trig:
                break

            remaining = n_trig - len(accepted_t)
            batch_size = max(remaining * 3, 10000)

            # 1. Generate exponential time
            t_exp = generate_exponential_time(batch_size)

            # 2. Calculate time acceptance for t_exp 
            if time_acc is not None:
                acc = time_acc(t_exp, np.full(batch_size, trig_val))
                acc_prob = acc / max_acc
                acc_prob = np.clip(acc_prob, 0.0, 1.0)
            else:
                acc_prob = np.ones(batch_size)

            # 3. Rejection sampling
            u = np.random.random(batch_size)
            accept_mask = u < acc_prob

            # 4. Sample resolution and add noise after acceptance
            idx = np.random.randint(resolution.shape[0], size=batch_size)
            res_i = resolution[idx]
            time_rec = t_exp + np.random.normal(size=batch_size) * res_i

            accepted_t.extend(time_rec[accept_mask].tolist())
            accepted_res.extend(res_i[accept_mask].tolist())
            accepted_texp.extend(t_exp[accept_mask].tolist())

            n_generated += batch_size
            n_accepted_total += np.sum(accept_mask)

        if len(accepted_t) < n_trig:
            print(f"      Warning: only generated {len(accepted_t)}/{n_trig} events after {max_iter} iterations")

        accept_rate = n_accepted_total / n_generated if n_generated > 0 else 0
        print(f"      Generated: {n_generated}, Accepted: {n_accepted_total} ({accept_rate * 100:.1f}%)")

        t_smear[trig_mask] = accepted_t[:n_trig]
        t_res[trig_mask] = accepted_res[:n_trig]
        texp_weight[trig_mask] = np.exp(gamma * np.array(accepted_texp[:n_trig]))

print("\n" + "=" * 60)
print("Saving output files")
print("=" * 60)

np.save(f"MC_t_smear_{file_suffix}.npy", t_smear)
np.save(f"MC_t_resolution_{file_suffix}.npy", t_res)

print(f"  Saved: MC_t_smear_{file_suffix}.npy ({len(t_smear)} events)")
print(f"  Saved: MC_t_resolution_{file_suffix}.npy ({len(t_res)} events)")
print(f"  Total events: {len(t_smear)} ")

# Cut on time_rec > 0
cut = t_smear > 0

print(f"\nCut results (time_rec > 0):")
print(f"  Total events: {len(t_smear)}")
print(f"  Events with time_rec > 0: {np.sum(cut)} ({np.sum(cut) / len(t_smear) * 100:.2f}%)")
print(f"  Events removed: {np.sum(~cut)} ({np.sum(~cut) / len(t_smear) * 100:.2f}%)")

# Apply cut to texp_weight
texp_weight_cut = texp_weight * cut

# Final weight = sWeight * texp_weight_cut
w = np.load(f"MC_weight_{file_suffix}.npy")
new_w = w * texp_weight_cut
np.save(f"MC_weight_cut_{file_suffix}.npy", new_w)

# Output checks
print("\n" + "=" * 60)
print("Output Validation")
print("=" * 60)
print(f"  time_rec range: [{np.min(t_smear):.4f}, {np.max(t_smear):.4f}] ps")
print(f"  time_rec mean: {np.mean(t_smear):.4f} ps")
print(f"  resolution range: [{np.min(t_res):.6f}, {np.max(t_res):.6f}] ps")
print(f"  resolution mean: {np.mean(t_res):.6f} ps")
print(f"  texp_weight range: [{np.min(texp_weight):.4f}, {np.max(texp_weight):.4f}]")
print(f"  texp_weight mean: {np.mean(texp_weight):.4f}")
print(f"  sWeight sum (before cut): {np.sum(w):.4f}")
print(f"  Final weight sum (after cut): {np.sum(new_w):.4f}")

for y in selected_years:
    y_mask = mc_year == y
    if np.any(y_mask):
        print(f"  Year {y}: {np.sum(y_mask)} events, time_rec mean = {np.mean(t_smear[y_mask]):.4f} ps")
