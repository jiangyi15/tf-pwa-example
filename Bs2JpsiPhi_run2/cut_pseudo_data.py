import numpy as np
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Cut pseudo_data and pseudo_MC with time resolution')
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

# Load data resolution
resolution = np.load(f"data_t_resolution_{file_suffix}.npy")
print(f"\nLoaded data resolution: {len(resolution)} values")
print(f"  Resolution range: [{np.min(resolution):.6f}, {np.max(resolution):.6f}] ps")

# ============================================================
# Process pseudo_data
# ============================================================
print("\n" + "=" * 60)
print("Processing pseudo_data")
print("=" * 60)

pseudo_data_time = np.load(f"pseudo_data_t_smear_{file_suffix}.npy")
pseudo_data_weight = np.load(f"pseudo_data_weight_{file_suffix}.npy")
n_data = len(pseudo_data_time)
print(f"  Loaded pseudo_data: {n_data} events")

# Randomly sample resolution with same number of events
idx_data = np.random.randint(resolution.shape[0], size=n_data)
data_t_res = resolution[idx_data]
np.save(f"pseudo_data_t_resolution_{file_suffix}.npy", data_t_res)
print(f"  Saved: pseudo_data_t_resolution_{file_suffix}.npy ({len(data_t_res)} events)")

# Cut on time > 0
data_cut = pseudo_data_time > 0
print(f"\n  Cut results (time > 0):")
print(f"    Total events: {n_data}")
print(f"    Events with time > 0: {np.sum(data_cut)} ({np.sum(data_cut) / n_data * 100:.2f}%)")
print(f"    Events removed: {np.sum(~data_cut)} ({np.sum(~data_cut) / n_data * 100:.2f}%)")

# Apply cut to weight
data_weight_cut = pseudo_data_weight * data_cut
np.save(f"pseudo_data_weight_cut_{file_suffix}.npy", data_weight_cut)
print(f"  Saved: pseudo_data_weight_cut_{file_suffix}.npy")

print(f"\n  pseudo_data Validation:")
print(f"    time range: [{np.min(pseudo_data_time):.4f}, {np.max(pseudo_data_time):.4f}] ps")
print(f"    time mean: {np.mean(pseudo_data_time):.4f} ps")
print(f"    resolution range: [{np.min(data_t_res):.6f}, {np.max(data_t_res):.6f}] ps")
print(f"    resolution mean: {np.mean(data_t_res):.6f} ps")
print(f"    sWeight sum (before cut): {np.sum(pseudo_data_weight):.4f}")
print(f"    Final weight sum (after cut): {np.sum(data_weight_cut):.4f}")

# ============================================================
# Process pseudo_MC
# ============================================================
print("\n" + "=" * 60)
print("Processing pseudo_MC")
print("=" * 60)

pseudo_mc_time = np.load(f"pseudo_MC_t_smear_{file_suffix}.npy")
pseudo_mc_weight = np.load(f"pseudo_MC_weight_{file_suffix}.npy")
n_mc = len(pseudo_mc_time)
print(f"  Loaded pseudo_MC: {n_mc} events")

# Randomly sample resolution with same number of events
idx_mc = np.random.randint(resolution.shape[0], size=n_mc)
mc_t_res = resolution[idx_mc]
np.save(f"pseudo_MC_t_resolution_{file_suffix}.npy", mc_t_res)
print(f"  Saved: pseudo_MC_t_resolution_{file_suffix}.npy ({len(mc_t_res)} events)")

# Cut on time > 0
mc_cut = pseudo_mc_time > 0
print(f"\n  Cut results (time > 0):")
print(f"    Total events: {n_mc}")
print(f"    Events with time > 0: {np.sum(mc_cut)} ({np.sum(mc_cut) / n_mc * 100:.2f}%)")
print(f"    Events removed: {np.sum(~mc_cut)} ({np.sum(~mc_cut) / n_mc * 100:.2f}%)")

# Apply cut to weight
mc_weight_cut = pseudo_mc_weight * mc_cut
np.save(f"pseudo_MC_weight_cut_{file_suffix}.npy", mc_weight_cut)
print(f"  Saved: pseudo_MC_weight_cut_{file_suffix}.npy")

print(f"\n  pseudo_MC Validation:")
print(f"    time range: [{np.min(pseudo_mc_time):.4f}, {np.max(pseudo_mc_time):.4f}] ps")
print(f"    time mean: {np.mean(pseudo_mc_time):.4f} ps")
print(f"    resolution range: [{np.min(mc_t_res):.6f}, {np.max(mc_t_res):.6f}] ps")
print(f"    resolution mean: {np.mean(mc_t_res):.6f} ps")
print(f"    sWeight sum (before cut): {np.sum(pseudo_mc_weight):.4f}")
print(f"    Final weight sum (after cut): {np.sum(mc_weight_cut):.4f}")

print("\n" + "=" * 60)
print("All done!")
print("=" * 60)