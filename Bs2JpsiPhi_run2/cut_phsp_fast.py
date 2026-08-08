import numpy as np
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Apply time smearing and cut to PHSP samples using real data resolution')
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

time = np.load(f"phsp_t_exp_{file_suffix}.npy")
resolution = np.load(f"data_t_resolution_{file_suffix}.npy")

print(f"\nLoaded data:")
print(f"  PHSP events: {len(time)}")
print(f"  Calibrated resolution values from data: {len(resolution)}")
print(f"  resolution range: [{np.min(resolution):.6f}, {np.max(resolution):.6f}] ps")

idx = np.random.randint(resolution.shape[0], size=time.shape)

resolution_i = resolution[idx]

time_rec = time + np.random.normal(size=time.shape[0]) * resolution_i

np.save(f"phsp_t_smear_{file_suffix}.npy", time_rec)
np.save(f"phsp_t_resolution_{file_suffix}.npy", resolution_i)

cut = time_rec > 0

print(f"\nCut results:")
print(f"  Total events: {len(time_rec)}")
print(f"  Events with time_rec > 0: {np.sum(cut)} ({np.sum(cut)/len(time_rec)*100:.2f}%)")
print(f"  Events removed: {np.sum(~cut)} ({np.sum(~cut)/len(time_rec)*100:.2f}%)")

w = np.load(f"phsp_weight_{file_suffix}.npy")
new_w = w * cut
np.save(f"phsp_weight_cut_{file_suffix}.npy", new_w)
