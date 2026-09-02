import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Cut pseudo_data and pseudo_MC with time resolution (simultaneous fit)')
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

# simultaneous fit: each (year, trigger) is an independent sub-sample
trigger_list = ['unbiased', 'biased'] if selected_trigger == 'all' else [selected_trigger]

np.random.seed(seed + 1000)

print(f"Selected years: {selected_years}")
print(f"Selected trigger: {selected_trigger} -> sub-samples: {trigger_list}")
print(f"Random seed: {seed}")


def process_subsample(year, trig):
    """Process one (year, trigger) sub-sample: cut time > 0."""
    sub_suffix = f"{year}_{trig}"
    print("\n" + "=" * 60)
    print(f"Processing sub-sample {sub_suffix}")
    print("=" * 60)

    # ============================================================
    # Process pseudo_data
    # ============================================================
    print("  [pseudo_data]")
    try:
        pseudo_data_time = np.load(f"pseudo_data_t_smear_{sub_suffix}.npy")
        pseudo_data_weight = np.load(f"pseudo_data_weight_{sub_suffix}.npy")
    except FileNotFoundError as e:
        print(f"  SKIP (missing file): {e}")
        return False
    n_data = len(pseudo_data_time)
    print(f"    Loaded pseudo_data: {n_data} events")

    # Cut on time > 0
    data_cut = pseudo_data_time > 0
    print(f"    Cut (time > 0): kept {np.sum(data_cut)} / {n_data} ({np.sum(data_cut) / n_data * 100:.2f}%)")

    data_weight_cut = pseudo_data_weight * data_cut
    np.save(f"pseudo_data_weight_cut_{sub_suffix}.npy", data_weight_cut)
    print(f"    Saved: pseudo_data_weight_cut_{sub_suffix}.npy")

    print(f"    time range: [{np.min(pseudo_data_time):.4f}, {np.max(pseudo_data_time):.4f}] ps")
    print(f"    weight sum (before/after): {np.sum(pseudo_data_weight):.4f} / {np.sum(data_weight_cut):.4f}")

    # ============================================================
    # Process pseudo_MC
    # ============================================================
    print("  [pseudo_MC]")
    try:
        pseudo_mc_time = np.load(f"pseudo_MC_t_smear_{sub_suffix}.npy")
        pseudo_mc_weight = np.load(f"pseudo_MC_weight_{sub_suffix}.npy")
    except FileNotFoundError as e:
        print(f"    SKIP pseudo_MC (missing file): {e}")
        return True
    n_mc = len(pseudo_mc_time)
    print(f"    Loaded pseudo_MC: {n_mc} events")

    mc_cut = pseudo_mc_time > 0
    print(f"    Cut (time > 0): kept {np.sum(mc_cut)} / {n_mc} ({np.sum(mc_cut) / n_mc * 100:.2f}%)")

    mc_weight_cut = pseudo_mc_weight * mc_cut
    np.save(f"pseudo_MC_weight_cut_{sub_suffix}.npy", mc_weight_cut)
    print(f"    Saved: pseudo_MC_weight_cut_{sub_suffix}.npy")

    print(f"    time range: [{np.min(pseudo_mc_time):.4f}, {np.max(pseudo_mc_time):.4f}] ps")
    print(f"    weight sum (before/after): {np.sum(pseudo_mc_weight):.4f} / {np.sum(mc_weight_cut):.4f}")
    return True


n_ok = 0
n_fail = 0
for year in selected_years:
    for trig in trigger_list:
        if process_subsample(year, trig):
            n_ok += 1
        else:
            n_fail += 1

print("\n" + "=" * 60)
print(f"All done! sub-samples processed: {n_ok} ok, {n_fail} skipped")
print("=" * 60)
