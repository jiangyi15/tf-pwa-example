import uproot
import numpy as np
import sys
import math
import argparse
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.amp import time_dep
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Create pseudo-experiment data and pseudo MC from MC ROOT files')
    parser.add_argument('--seed', type=int, default=10, help='Random seed (default: 10)')
    parser.add_argument('--years', type=str, default='2015,2016,2017,2018',
                        help='Years to process, comma-separated (default: 2015,2016,2017,2018)')
    parser.add_argument('--trigger', type=str, default='all',
                        choices=['all', 'unbiased', 'biased'],
                        help='Trigger type to select (default: all)')
    return parser.parse_args()

args = parse_args()

seed = args.seed
np.random.seed(seed)

selected_years = [int(y.strip()) for y in args.years.split(',')]
selected_trigger = args.trigger

print(f"Selected years: {selected_years}")
print(f"Selected trigger: {selected_trigger}")
print(f"Random seed: {seed}")

all_vars = [
    "helcosthetaK",
    "helcosthetaL",
    "helphi",
    "time",
    "sigmat",
    "sw_p2vv",
    "B_ConstJpsi_M_1",
    #"B_ID",
    "B_ID_GenLvl",
    "B_TRUETAU"
    #"B_TRUETAU_GenLvl"
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

# Filter out events with invalid B_TRUETAU (sentinel value -1 causes amplitude overflow)
valid_mask = merged_data['B_TRUETAU'] >= 0
n_invalid = int(np.sum(~valid_mask))
if n_invalid > 0:
    print(f"Filtering out {n_invalid} events with B_TRUETAU < 0 (sentinel values)")
    for key in all_vars + ["year"]:
        merged_data[key] = merged_data[key][valid_mask]
n_total = len(merged_data['B_ConstJpsi_M_1'])
print(f"Events after filtering: {n_total}")

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
data_sigmat = data_dict["sigmat"]
data_year = data_dict["year"]
#data_b_id = data_dict["B_ID"]
data_b_id = data_dict["B_ID_GenLvl"]
data_b_truetau_ns = data_dict["B_TRUETAU"]
#data_b_truetau_ns = data_dict["B_TRUETAU_GenLvl"]
data_b_truetau = data_b_truetau_ns * 1000.
data_b_constjpsi_mass = data_dict["B_ConstJpsi_M_1"]

# Extract variables for pseudo_MC
mc_sw = mc_dict["sw_p2vv"]
mc_helcosthetaK = mc_dict["helcosthetaK"]
mc_helcosthetaL = mc_dict["helcosthetaL"]
mc_helphi = mc_dict["helphi"]
mc_time = mc_dict["time"]
mc_sigmat = mc_dict["sigmat"]
mc_year = mc_dict["year"]
#mc_b_id = mc_dict["B_ID"]
mc_b_id = mc_dict["B_ID_GenLvl"]
mc_b_truetau_ns = mc_dict["B_TRUETAU"]
#mc_b_truetau_ns = mc_dict["B_TRUETAU_GenLvl"]
mc_b_truetau = mc_b_truetau_ns * 1000.
mc_b_constjpsi_mass = mc_dict["B_ConstJpsi_M_1"]

n_data = len(data_b_constjpsi_mass)
n_mc = len(mc_b_constjpsi_mass)

print(f"\npseudo_data events: {n_data}")
print(f"pseudo_MC events: {n_mc}")

print("\n" + "="*60)
print("Computing |A_sim|^2 for pseudo_MC amplitude correction")
print("="*60)

config = ConfigLoader("config_gen_pseudo_data.yml")
config.set_params("final_params_decay.json")

# ============================================================
# Check parameters against decay file
# ============================================================
print("\n" + "=" * 80)
print("Check parameters against physical results")
print("=" * 80)

# Physics results (LHCb results)
expected_params = {
    "delta_m": 17.8,      # Δm_s [ps⁻¹]
    "delta_gamma": 0.08543,  # ΔΓ_s [ps⁻¹]
    "gamma": 0.6614,      # Γ_s [ps⁻¹]
    "phi_s": -0.03,       # φ_s [rad]
    "A0_sq": 0.524176,      # |A₀(0)|² (from decay card: 0.724²)
    "Aparallel_sq": 0.225625,  # |A_∥(0)|² (from decay card: 0.475²)
    "Aperp_sq": 0.250000,   # |A_⊥(0)|² (from decay card: 0.500²)
    "delta_parallel": 3.26,  # δ_∥ - δ₀ [rad]
    "delta_perp": 3.08,      # δ_⊥ - δ₀ [rad]
}

# Convert tf-pwa parameters to physical results
with config.params_trans() as pt:
    # B_s mixing parameters
    bs_delta_m = float(pt["Bs_delta_m"].numpy())
    bs_delta_gamma = -float(pt["Bs_delta_gamma"].numpy())  # Attention: ΔΓ_s has negative sign convention
    bs_gamma = float(pt["Bs_gamma"].numpy())
    bs_phi_s = float(pt["Bs_poqi"].numpy())  # φ_s is Bs_poqi
    
    print("\n1. B_s mixing parameters comparison:")
    print("-" * 80)
    print(f"  {'Parameter':<20} {'tf-pwa':<15} {'Expected':<15} {'Difference':<15} {'Status':<10}")
    print(f"  {'-'*75}")
    
    for name, actual, expected in [
        ("Δm_s", bs_delta_m, expected_params["delta_m"]),
        ("ΔΓ_s", bs_delta_gamma, expected_params["delta_gamma"]),
        ("Γ_s", bs_gamma, expected_params["gamma"]),
        ("φ_s", bs_phi_s, expected_params["phi_s"]),
    ]:
        diff = actual - expected
        status = "✅" if abs(diff) < 0.01 else "❌"
        print(f"  {name:<20} {actual:<15.5f} {expected:<15.5f} {diff:+.5f} {status:<10}")
    
    # Polarization amplitude parameters
    rho0 = float(pt["Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0r"].numpy())
    phi0 = float(pt["Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0i"].numpy())
    rho1 = float(pt["Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0r"].numpy())
    phi1 = float(pt["Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0i"].numpy())
    rho2 = float(pt["Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0r"].numpy())
    phi2 = float(pt["Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0i"].numpy())
    
    # Convert to complex numbers
    g0 = tf.complex(rho0 * tf.cos(phi0), rho0 * tf.sin(phi0))
    g1 = tf.complex(rho1 * tf.cos(phi1), rho1 * tf.sin(phi1))
    g2 = tf.complex(rho2 * tf.cos(phi2), rho2 * tf.sin(phi2))
    
    # Convert to physical polarization amplitudes
    A0 = - g0 * math.sqrt(1/3) + g2 * math.sqrt(2/3)
    Aperp = -g1
    Aparallel = -g0 * math.sqrt(2/3) - g2 * math.sqrt(1/3)
    
    # Compute physical quantities
    A0_sq = np.abs(A0)**2
    Aperp_sq = np.abs(Aperp)**2
    Aparallel_sq = np.abs(Aparallel)**2
    
    # Phase differences
    phi_range = lambda x: (x - 0)% (2*math.pi) + 0
    delta_perp_minus_0 = phi_range(-tf.math.angle(Aperp/A0))
    delta_parallel_minus_0 = phi_range(-tf.math.angle(Aparallel/A0))
    
    print("\n2. Amplitude parameters comparison:")
    print("-" * 80)
    print(f"  {'Parameter':<20} {'tf-pwa':<15} {'Expected':<15} {'Difference':<15} {'Status':<10}")
    print(f"  {'-'*75}")
    
    for name, actual, expected in [
        ("|A₀(0)|²", A0_sq, expected_params["A0_sq"]),
        ("|A_⊥(0)|²", Aperp_sq, expected_params["Aperp_sq"]),
        ("|A_∥(0)|²", Aparallel_sq, expected_params["Aparallel_sq"]),
    ]:
        diff = actual - expected
        status = "✅" if abs(diff) < 0.01 else "❌"
        print(f"  {name:<20} {actual:<15.6f} {expected:<15.4f} {diff:+.6f} {status:<10}")
    
    print("\n3. Phase difference parameters comparison:")
    print("-" * 80)
    print(f"  {'Parameter':<20} {'tf-pwa':<15} {'Expected':<15} {'Difference':<15} {'Status':<10}")
    print(f"  {'-'*75}")
    
    for name, actual, expected in [
        ("δ_⊥ - δ₀", delta_perp_minus_0, expected_params["delta_perp"]),
        ("δ_∥ - δ₀", delta_parallel_minus_0, expected_params["delta_parallel"]),
    ]:
        diff = actual - expected
        status = "✅" if abs(diff) < 0.05 else "❌"
        print(f"  {name:<20} {actual:<15.4f} {expected:<15.2f} {diff:+.4f} {status:<10}")

print("=" * 80)
print("Parameters check completed")
print("=" * 80 + "\n")

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
    event_tag = np.where(mc_b_id[i:end] > 0, 1.0, -1.0).astype(np.float64)
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

#if selected_trigger == 'unbiased':
#    data_trigger = np.zeros(n_data, dtype=np.int32)
#    mc_trigger = np.zeros(n_mc, dtype=np.int32)
#elif selected_trigger == 'biased':
#    data_trigger = np.ones(n_data, dtype=np.int32)
#    mc_trigger = np.ones(n_mc, dtype=np.int32)
#else:
#    data_trigger = np.random.choice([0, 1], n_data, p=[0.8, 0.2])
#    mc_trigger = np.random.choice([0, 1], n_mc, p=[0.8, 0.2])

#data_tag = np.random.choice([-1, 0, 1], n_data, p=[0.4, 0.2, 0.4]).astype(np.int32)
#data_eta = np.random.random(n_data) * 0.5
#
#mc_tag = np.random.choice([-1, 0, 1], n_mc, p=[0.4, 0.2, 0.4]).astype(np.int32)
#mc_eta = np.random.random(n_mc) * 0.5

#data_tag = np.where(data_b_id > 0, 1, -1).astype(np.int32)
#data_eta = np.zeros(n_data, dtype=np.float64)  
#
#mc_tag = np.where(mc_b_id > 0, 1, -1).astype(np.int32)
#mc_eta = np.zeros(n_mc, dtype=np.float64)  

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
np.save(f"pseudo_data_t_resolution_{file_suffix}.npy", data_sigmat)
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
np.save(f"pseudo_MC_t_resolution_{file_suffix}.npy", mc_sigmat)
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