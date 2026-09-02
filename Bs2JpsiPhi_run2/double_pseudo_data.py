import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Double pseudo_data and pseudo_MC samples for CP symmetry analysis')
    parser.add_argument('--years', type=str, default='2015,2016,2017,2018',
                        help='Years to process, comma-separated (default: 2015,2016,2017,2018)')
    parser.add_argument('--trigger', type=str, default='all', choices=['all', 'unbiased', 'biased'],
                        help='Trigger type (default: all)')
    return parser.parse_args()


args = parse_args()

selected_years = [int(y.strip()) for y in args.years.split(',')]
selected_trigger = args.trigger

print(f"Selected years: {selected_years}")
print(f"Selected trigger: {selected_trigger}")

years_suffix = '_'.join(str(y) for y in selected_years)
file_suffix = f"{years_suffix}_{selected_trigger}"


def double_sample(prefix, file_suffix):
    """Double a sample for CP symmetry analysis"""
    print(f"\nDoubling {prefix} samples...")
    
    # weight_cut
    p4 = np.load(f"{prefix}_weight_cut_{file_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_weight_cut_double_{file_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_weight_cut_double_{file_suffix}.npy")
    
    # t_smear
    p4 = np.load(f"{prefix}_t_smear_{file_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_t_smear_double_{file_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_t_smear_double_{file_suffix}.npy")
    
    # t_resolution
    p4 = np.load(f"{prefix}_t_resolution_{file_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_t_resolution_double_{file_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_t_resolution_double_{file_suffix}.npy")
    
    # tag (second copy with sign flipped for CP symmetry)
    p4 = np.load(f"{prefix}_tag_{file_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, -1]).reshape((2,))
    np.save(f"{prefix}_tag_double_{file_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_tag_double_{file_suffix}.npy")
    
    # angles
    p4 = np.load(f"{prefix}_angles_{file_suffix}.npy").reshape((-1, 1, 3))
    p4_new = p4 * np.array([1, 1]).reshape((2, 1))
    np.save(f"{prefix}_angles_double_{file_suffix}.npy", p4_new.reshape((-1, 3)))
    print(f"  Saved: {prefix}_angles_double_{file_suffix}.npy")
    
    # eta
    p4 = np.load(f"{prefix}_eta_{file_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_eta_double_{file_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_eta_double_{file_suffix}.npy")
    
    # trigger
    p4 = np.load(f"{prefix}_trigger_{file_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_trigger_double_{file_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_trigger_double_{file_suffix}.npy")
    
    # year
    p4 = np.load(f"{prefix}_year_{file_suffix}.npy").reshape((-1, 1))
    p4_new = p4 * np.array([1, 1]).reshape((2,))
    np.save(f"{prefix}_year_double_{file_suffix}.npy", p4_new.reshape((-1,)))
    print(f"  Saved: {prefix}_year_double_{file_suffix}.npy")


# Process pseudo_data
double_sample("pseudo_data", file_suffix)

# Process pseudo_MC
double_sample("pseudo_MC", file_suffix)

print("\nCP doubling complete for both pseudo_data and pseudo_MC!")