import uproot
import numpy as np
import sys
import argparse
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.amp import time_dep
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Create pseudo-experiment data and pseudo MC from MC ROOT files')
    parser.add_argument('--years', type=str, default='2015,2016,2017,2018',
                        help='Years to process, comma-separated (default: 2015,2016,2017,2018)')
    parser.add_argument('--trigger', type=str, default='all',
                        choices=['all', 'unbiased', 'biased'],
                        help='Trigger type to select (default: all)')
    return parser.parse_args()

args = parse_args()

selected_years = [int(y.strip()) for y in args.years.split(',')]
selected_trigger = args.trigger

print(f"Selected years: {selected_years}")
print(f"Selected trigger: {selected_trigger}")

all_vars = [
    "helcosthetaK",
    "helcosthetaL",
    "helphi",
    "time",
    "sw_p2vv",
    "B_ConstJpsi_M_1",
    "B_ID_GenLvl",
    "B_TRUETAU_GenLvl"
]

def load_year_data(year):
    """Load data for a single year"""
    file_path = f"~/myeos/Bs2JpsiPhi_run2/v4r1_MC_Bs2JpsiPhi_{year}_sw.root"
    print(f"Loading {year} data from {file_path}")
    
    try:
        with uproot.open(file_path) as f:
            t = f.get("DecayTree")
            data = t.arrays(all_vars)
            data = {k: np.array(data[k]) for k in all_vars}
        
        data["year"] = np.full(len(data["B_ConstJpsi_M_1"]), year, dtype=np.int32)
        return data
        
    except Exception as e:
        print(f"Error loading {year} data: {e}")
        return None

years = [2015, 2016, 2017, 2018]
all_data_list = []

for year in years:
    if year not in selected_years:
        print(f"Skipping year {year} (not in selected years)")
        continue
    year_data = load_year_data(year)
    if year_data is not None:
        all_data_list.append(year_data)
        print(f"Loaded {len(year_data['B_ConstJpsi_M_1'])} events for {year}")

if not all_data_list:
    print("No data loaded!")
    sys.exit(1)

print(f"\nMerging {len(all_data_list)} years of data...")

merged_data = {}
for key in all_vars + ["year"]:
    merged_data[key] = np.concatenate([d[key] for d in all_data_list])

n_total = len(merged_data['B_ConstJpsi_M_1'])
print(f"Total events after merge: {n_total}")

print("\n" + "="*60)
print("Randomly splitting into pseudo_data (50%) and pseudo_MC (50%)")
print("="*60)

indices = np.arange(n_total)
np.random.shuffle(indices)
n_half = n_total // 2

data_indices = indices[:n_half]
mc_indices = indices[n_half:]

print(f"pseudo_data: {n_half} events")
print(f"pseudo_MC: {n_total - n_half} events")

data_dict = {}
mc_dict = {}
for key in all_vars + ["year"]:
    data_dict[key] = merged_data[key][data_indices]
    mc_dict[key] = merged_data[key][mc_indices]

# Extract variables for pseudo_data
data_sw = data_dict["sw_p2vv"]
data_helcosthetaK = data_dict["helcosthetaK"]
data_helcosthetaL = data_dict["helcosthetaL"]
data_helphi = data_dict["helphi"]
data_time = data_dict["time"]
data_year = data_dict["year"]
data_b_id_genlvl = data_dict["B_ID_GenLvl"]
data_b_truetau_ns = data_dict["B_TRUETAU_GenLvl"]
data_b_truetau = data_b_truetau_ns * 1000.
data_b_constjpsi_mass = data_dict["B_ConstJpsi_M_1"]

# Extract variables for pseudo_MC
mc_sw = mc_dict["sw_p2vv"]
mc_helcosthetaK = mc_dict["helcosthetaK"]
mc_helcosthetaL = mc_dict["helcosthetaL"]
mc_helphi = mc_dict["helphi"]
mc_time = mc_dict["time"]
mc_year = mc_dict["year"]
mc_b_id_genlvl = mc_dict["B_ID_GenLvl"]
mc_b_truetau_ns = mc_dict["B_TRUETAU_GenLvl"]
mc_b_truetau = mc_b_truetau_ns * 1000.
mc_b_constjpsi_mass = mc_dict["B_ConstJpsi_M_1"]

n_data = len(data_b_constjpsi_mass)
n_mc = len(mc_b_constjpsi_mass)

print(f"\npseudo_data events: {n_data}")
print(f"pseudo_MC events: {n_mc}")

print("\n" + "="*60)
print("Computing |A_sim|^2 for pseudo_MC amplitude correction")
print("="*60)

config = ConfigLoader("config_gen.yml")
config.set_params("final_params_decay.json")

f = config.get_particle_function("phi10")
ha = f.ha

batch_size = 250000
amp_sq_mc = np.zeros(n_mc, dtype=np.float64)
mc_c1 = mc_helcosthetaL
mc_c2 = mc_helcosthetaK
mc_phi = mc_helphi

for i in range(0, n_mc, batch_size):
    end = min(i + batch_size, n_mc)
    p4 = ha.build_data(
        {},
        [np.array([0.]), mc_c1[i:end], mc_c2[i:end]],
        [np.array([0.]), np.array([0.]), mc_phi[i:end]]
    )
    data = config.data.cal_angle(p4)
    data["time"] = tf.constant(mc_b_truetau[i:end], dtype=tf.float64)
    event_tag = np.where(mc_b_id_genlvl[i:end] > 0, 1.0, -1.0).astype(np.float64)
    data["tag"] = tf.constant(event_tag, dtype=tf.float64)
    data["eta"] = tf.constant(np.zeros(end - i), dtype=tf.float64)
    data["del_eta"] = tf.constant(np.zeros(end - i), dtype=tf.float64)
    amp_sq_mc[i:end] = config.get_amplitude()(data).numpy()

    if (i // batch_size) % 4 == 0:
        print(f"  Processed {end}/{n_mc} events, |A|^2 range: [{amp_sq_mc[i:end].min():.6f}, {amp_sq_mc[i:end].max():.6f}]")

n_low = np.sum(amp_sq_mc < 1e-10)
if n_low > 0:
    print(f"  WARNING: {n_low} events have |A|^2 < 1e-10, clipping to minimum")
amp_sq_mc = np.clip(amp_sq_mc, 1e-10, None)

print(f"\n|A_sim|^2 statistics for pseudo_MC:")
print(f"  Mean: {np.mean(amp_sq_mc):.6f}")
print(f"  Median: {np.median(amp_sq_mc):.6f}")
print(f"  Min: {np.min(amp_sq_mc):.6f}")
print(f"  Max: {np.max(amp_sq_mc):.6f}")
print(f"  Std: {np.std(amp_sq_mc):.6f}")

mc_sw_corrected = mc_sw / amp_sq_mc

print(f"\npseudo_MC Raw weight (sw_p2vv) statistics:")
print(f"  Mean: {np.mean(mc_sw):.6f}")
print(f"  Std: {np.std(mc_sw):.6f}")
print(f"\npseudo_MC Corrected weight (sw_p2vv / |A|^2) statistics:")
print(f"  Mean: {np.mean(mc_sw_corrected):.6f}")
print(f"  Std: {np.std(mc_sw_corrected):.6f}")

print("\n" + "="*60)
print("Generating trigger, tag, eta for both datasets")
print("="*60)

years_suffix = '_'.join(str(y) for y in selected_years)
trigger_suffix = selected_trigger
file_suffix = f"{years_suffix}_{trigger_suffix}"

real_data_tag = np.load(f"data_tag_{file_suffix}.npy")
real_data_eta = np.load(f"data_eta_{file_suffix}.npy")
real_data_trigger = np.load(f"data_trigger_{file_suffix}.npy")

idx_pseudo_data_tag = np.random.randint(real_data_tag.shape[0], size=n_data)
idx_pseudo_data_eta = np.random.randint(real_data_eta.shape[0], size=n_data)
idx_pseudo_data_trigger = np.random.randint(real_data_trigger.shape[0], size=n_data)

data_tag = real_data_tag[idx_pseudo_data_tag]
data_eta = real_data_eta[idx_pseudo_data_eta]
data_trigger = real_data_trigger[idx_pseudo_data_trigger]

idx_pseudo_mc_tag = np.random.randint(real_data_tag.shape[0], size=n_mc)
idx_pseudo_mc_eta = np.random.randint(real_data_eta.shape[0], size=n_mc)
idx_pseudo_mc_trigger = np.random.randint(real_data_trigger.shape[0], size=n_mc)

mc_tag = real_data_tag[idx_pseudo_mc_tag]
mc_eta = real_data_eta[idx_pseudo_mc_eta]
mc_trigger = real_data_trigger[idx_pseudo_mc_trigger]

print(f"pseudo_data - Tag range: {np.min(data_tag):.2f} - {np.max(data_tag):.2f}")
print(f"pseudo_data - Eta range: {np.min(data_eta):.2f} - {np.max(data_eta):.2f}")
print(f"pseudo_data - Trigger range: {np.min(data_trigger):.0f} - {np.max(data_trigger):.0f}")
print(f"pseudo_MC   - Tag range: {np.min(mc_tag):.2f} - {np.max(mc_tag):.2f}")
print(f"pseudo_MC   - Eta range: {np.min(mc_eta):.2f} - {np.max(mc_eta):.2f}")
print(f"pseudo_MC   - Trigger range: {np.min(mc_trigger):.0f} - {np.max(mc_trigger):.0f}")

print("\n" + "="*60)
print("Saving pseudo_data Output Files")
print("="*60)

print("Saving pseudo_data tagging data...")
np.save(f"pseudo_data_tag_{file_suffix}.npy", data_tag)
np.save(f"pseudo_data_eta_{file_suffix}.npy", data_eta)

print("Saving pseudo_data other data...")
np.save(f"pseudo_data_weight_{file_suffix}.npy", data_sw)
np.save(f"pseudo_data_angles_{file_suffix}.npy", np.stack([np.arccos(data_helcosthetaL), np.arccos(data_helcosthetaK), data_helphi], axis=-1))
np.save(f"pseudo_data_t_smear_{file_suffix}.npy", data_time)
np.save(f"pseudo_data_trigger_{file_suffix}.npy", data_trigger)
np.save(f"pseudo_data_year_{file_suffix}.npy", data_year)

print(f"\npseudo_data: {n_data} events")
print(f"Output files suffix: {file_suffix}")

print("\n" + "="*60)
print("Saving pseudo_MC Output Files")
print("="*60)

print("Saving pseudo_MC tagging data...")
np.save(f"pseudo_MC_tag_{file_suffix}.npy", mc_tag)
np.save(f"pseudo_MC_eta_{file_suffix}.npy", mc_eta)

print("Saving pseudo_MC other data...")
np.save(f"pseudo_MC_weight_{file_suffix}.npy", mc_sw_corrected)
np.save(f"pseudo_MC_angles_{file_suffix}.npy", np.stack([np.arccos(mc_helcosthetaL), np.arccos(mc_helcosthetaK), mc_helphi], axis=-1))
np.save(f"pseudo_MC_t_smear_{file_suffix}.npy", mc_time)
np.save(f"pseudo_MC_trigger_{file_suffix}.npy", mc_trigger)
np.save(f"pseudo_MC_year_{file_suffix}.npy", mc_year)

print(f"\npseudo_MC: {n_mc} events")
print(f"Output files suffix: {file_suffix}")

for y in years:
    y_mask_data = data_year == y
    y_mask_mc = mc_year == y
    if np.any(y_mask_data):
        print(f"pseudo_data Year {y}: {np.sum(y_mask_data)} events")
    if np.any(y_mask_mc):
        print(f"pseudo_MC Year {y}: {np.sum(y_mask_mc)} events")


def plot_distributions(dataset_name, sw_raw, sw_corrected, helcosthetaK, helcosthetaL, helphi, 
                       tag, eta, trigger, b_constjpsi_mass, suffix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    unbiased_mask = trigger == 0
    biased_mask = trigger == 1

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    ax1.hist(tag, bins=50, density=True, alpha=0.7, color='red')
    ax1.set_xlabel('Tag')
    ax1.set_ylabel('Density')
    ax1.set_title('Tag distribution')

    ax2.hist(eta, bins=50, density=True, alpha=0.7, color='orange')
    ax2.set_xlabel('Eta')
    ax2.set_ylabel('Density')
    ax2.set_title('Mistag rate distribution')

    ax3.scatter(b_constjpsi_mass, sw_raw, alpha=0.3, s=1, color='purple')
    ax3.set_xlabel('B_ConstJpsi_Mass')
    ax3.set_ylabel('sWeight')
    ax3.set_title('sWeight vs B_ConstJpsi_Mass')

    ax4.hist(helcosthetaK, bins=50, weights=sw_raw, density=True, alpha=0.7, color='grey', label='raw')
    ax4.hist(helcosthetaK, bins=50, weights=sw_corrected, density=True, alpha=0.7, color='cyan', label='corrected')
    ax4.set_xlabel('cos(theta_K)')
    ax4.set_ylabel('Density')
    ax4.set_title('Helicity cos(theta_K) distribution (weighted)')
    ax4.legend()

    ax5.hist(helcosthetaL, bins=50, weights=sw_raw, density=True, alpha=0.5, color='grey', label='raw')
    ax5.hist(helcosthetaL, bins=50, weights=sw_corrected, density=True, alpha=0.5, color='magenta', label='corrected')
    ax5.set_xlabel('cos(theta_L)')
    ax5.set_ylabel('Density')
    ax5.set_title('Helicity cos(theta_L) distribution (weighted)')
    ax5.legend()

    ax6.hist(helphi, bins=50, weights=sw_raw, density=True, alpha=0.5, color='grey', label='raw')
    ax6.hist(helphi, bins=50, weights=sw_corrected, density=True, alpha=0.5, color='brown', label='corrected')
    ax6.set_xlabel('phi (rad)')
    ax6.set_ylabel('Density')
    ax6.set_title('Helicity phi distribution (weighted)')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_distributions_{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGenerated plot: {dataset_name}_distributions_{suffix}.png")


print("\n" + "="*60)
print("Generating Plots")
print("="*60)

plot_distributions("pseudo_data", data_sw, data_sw, data_helcosthetaK, data_helcosthetaL, 
                   data_helphi, data_tag, data_eta, data_trigger, data_b_constjpsi_mass, file_suffix)

plot_distributions("pseudo_MC", mc_sw, mc_sw_corrected, mc_helcosthetaK, mc_helcosthetaL, 
                   mc_helphi, mc_tag, mc_eta, mc_trigger, mc_b_constjpsi_mass, file_suffix)

print("\n" + "="*60)
print("All done!")
print("="*60)