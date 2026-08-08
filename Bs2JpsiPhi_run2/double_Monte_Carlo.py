import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Double MC samples for CP symmetry analysis')
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


p4 = np.load(f"MC_weight_cut_{file_suffix}.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save(f"MC_weight_cut_double_{file_suffix}.npy", p4_new.reshape((-1,)))

p4 = np.load(f"MC_t_smear_{file_suffix}.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save(f"MC_t_smear_double_{file_suffix}.npy", p4_new.reshape((-1,)))

p4 = np.load(f"MC_t_resolution_{file_suffix}.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save(f"MC_t_resolution_double_{file_suffix}.npy", p4_new.reshape((-1,)))


p4 = np.load(f"MC_tag_{file_suffix}.npy").reshape((-1,1))
p4_new = p4 * np.array([1,-1]).reshape((2,))
np.save(f"MC_tag_double_{file_suffix}.npy", p4_new.reshape((-1,)))

p4 = np.load(f"MC_angles_{file_suffix}.npy").reshape((-1,1,3))
p4_new = p4 * np.array([1,1]).reshape((2,1))
np.save(f"MC_angles_double_{file_suffix}.npy", p4_new.reshape((-1,3)))

p4 = np.load(f"MC_eta_{file_suffix}.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save(f"MC_eta_double_{file_suffix}.npy", p4_new.reshape((-1,)))

p4 = np.load(f"MC_trigger_{file_suffix}.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save(f"MC_trigger_double_{file_suffix}.npy", p4_new.reshape((-1,)))

p4 = np.load(f"MC_year_{file_suffix}.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save(f"MC_year_double_{file_suffix}.npy", p4_new.reshape((-1,)))

print("\nCP doubling complete!")