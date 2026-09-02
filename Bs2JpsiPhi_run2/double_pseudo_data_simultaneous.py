import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Double pseudo_data and pseudo_MC samples for CP symmetry analysis (simultaneous fit)')
    parser.add_argument('--years', type=str, default='2015,2016,2017,2018',
                        help='Years to process, comma-separated (default: 2015,2016,2017,2018)')
    parser.add_argument('--trigger', type=str, default='all', choices=['all', 'unbiased', 'biased'],
                        help='Trigger type (default: all)')
    return parser.parse_args()


args = parse_args()

selected_years = [int(y.strip()) for y in args.years.split(',')]
selected_trigger = args.trigger

# simultaneous fit: each (year, trigger) is an independent sub-sample
trigger_list = ['unbiased', 'biased'] if selected_trigger == 'all' else [selected_trigger]

print(f"Selected years: {selected_years}")
print(f"Selected trigger: {selected_trigger} -> sub-samples: {trigger_list}")


def double_sample(prefix, sub_suffix):
    """Double a sample for CP symmetry analysis (per year×trigger sub-sample)."""
    print(f"\nDoubling {prefix} ({sub_suffix})...")

    # weight_cut
    p4 = np.load(f"{prefix}_weight_cut_{sub_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_weight_cut_double_{sub_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_weight_cut_double_{sub_suffix}.npy")

    # t_smear
    p4 = np.load(f"{prefix}_t_smear_{sub_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_t_smear_double_{sub_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_t_smear_double_{sub_suffix}.npy")

    # t_resolution
    p4 = np.load(f"{prefix}_t_resolution_{sub_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_t_resolution_double_{sub_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_t_resolution_double_{sub_suffix}.npy")

    # tag (second copy with sign flipped for CP symmetry)
    p4 = np.load(f"{prefix}_tag_{sub_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, -1]).reshape((2,))
    np.save(f"{prefix}_tag_double_{sub_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_tag_double_{sub_suffix}.npy")

    # angles
    p4 = np.load(f"{prefix}_angles_{sub_suffix}.npy").reshape((-1, 1, 3))
    p4_new = p4 * np.array([1, 1]).reshape((2, 1))
    np.save(f"{prefix}_angles_double_{sub_suffix}.npy", p4_new.reshape((-1, 3)))
    print(f"  Saved: {prefix}_angles_double_{sub_suffix}.npy")

    # eta
    p4 = np.load(f"{prefix}_eta_{sub_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_eta_double_{sub_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_eta_double_{sub_suffix}.npy")

    # trigger
    p4 = np.load(f"{prefix}_trigger_{sub_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_trigger_double_{sub_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_trigger_double_{sub_suffix}.npy")

    # year
    p4 = np.load(f"{prefix}_year_{sub_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_year_double_{sub_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_year_double_{sub_suffix}.npy")


n_ok = 0
for year in selected_years:
    for trig in trigger_list:
        sub_suffix = f"{year}_{trig}"
        try:
            # Only double pseudo_MC (PHSP): CP doubling is needed for the
            # normalization integral to cover both B and Bbar hypotheses.
            # pseudo_data is NOT doubled: each event has its own tag and
            # the model selects P/Pbar via where(tag>0, P, Pbar).
            double_sample("pseudo_MC", sub_suffix)
            n_ok += 1
        except FileNotFoundError as e:
            print(f"SKIP {sub_suffix} (missing file): {e}")

print(f"\nCP doubling complete! sub-samples: {n_ok}")
