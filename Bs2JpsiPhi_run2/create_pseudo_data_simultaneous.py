import uproot
import numpy as np
import sys
import math
import argparse
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.amp import time_dep
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Create pseudo-experiment data and pseudo MC (simultaneous fit: per year×trigger)')
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

# simultaneous fit: each (year, trigger) is an independent sub-sample
trigger_list = ['unbiased', 'biased'] if selected_trigger == 'all' else [selected_trigger]

print(f"Selected years: {selected_years}")
print(f"Selected trigger: {selected_trigger} -> sub-samples: {trigger_list}")
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
    """Load MC data for a single year"""
    file_path = f"~/myeos/Bs2JpsiPhi_run2/v4r1_MC_Bs2JpsiPhi_{year}_sw.root"
    print(f"Loading {year} MC from {file_path}")

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


# ============================================================
# Load tf-pwa amplitude model (shared across all sub-samples)
# ============================================================
config = ConfigLoader("config_gen_pseudo_data.yml")
config.set_params("final_params_decay.json")

# ============================================================
# Check parameters against decay file (once, shared model)
# ============================================================
print("\n" + "=" * 80)
print("Check parameters against physical results")
print("=" * 80)

expected_params = {
    "delta_m": 17.8,
    "delta_gamma": 0.08543,
    "gamma": 0.6614,
    "phi_s": -0.03,
    "A0_sq": 0.524176,
    "Aparallel_sq": 0.225625,
    "Aperp_sq": 0.250000,
    "delta_parallel": 3.26,
    "delta_perp": 3.08,
}

with config.params_trans() as pt:
    bs_delta_m = float(pt["Bs_delta_m"].numpy())
    bs_delta_gamma = -float(pt["Bs_delta_gamma"].numpy())
    bs_gamma = float(pt["Bs_gamma"].numpy())
    bs_phi_s = float(pt["Bs_poqi"].numpy())

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

    rho0 = float(pt["Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0r"].numpy())
    phi0 = float(pt["Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0i"].numpy())
    rho1 = float(pt["Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0r"].numpy())
    phi1 = float(pt["Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0i"].numpy())
    rho2 = float(pt["Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0r"].numpy())
    phi2 = float(pt["Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0i"].numpy())

    g0 = tf.complex(rho0 * tf.cos(phi0), rho0 * tf.sin(phi0))
    g1 = tf.complex(rho1 * tf.cos(phi1), rho1 * tf.sin(phi1))
    g2 = tf.complex(rho2 * tf.cos(phi2), rho2 * tf.sin(phi2))

    A0 = - g0 * math.sqrt(1/3) + g2 * math.sqrt(2/3)
    Aperp = -g1
    Aparallel = -g0 * math.sqrt(2/3) - g2 * math.sqrt(1/3)

    A0_sq = np.abs(A0)**2
    Aperp_sq = np.abs(Aperp)**2
    Aparallel_sq = np.abs(Aparallel)**2

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


def compute_amp_sq(c1, c2, phi, b_truetau, b_id):
    """Compute |A_sim|^2 for a batch of events."""
    n = len(c1)
    amp_sq = np.zeros(n, dtype=np.float64)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        p4 = ha.build_data(
            {},
            [np.array([0.]), c1[i:end], c2[i:end]],
            [np.array([0.]), np.array([0.]), phi[i:end]]
        )
        data = config.data.cal_angle(p4)
        data["time"] = tf.constant(b_truetau[i:end], dtype=tf.float64)
        event_tag = np.where(b_id[i:end] > 0, 1.0, -1.0).astype(np.float64)
        data["tag"] = tf.constant(event_tag, dtype=tf.float64)
        data["eta"] = tf.constant(np.zeros(end - i), dtype=tf.float64)
        data["del_eta"] = tf.constant(np.zeros(end - i), dtype=tf.float64)
        amp_sq[i:end] = config.get_amplitude()(data).numpy()

        if (i // batch_size) % 4 == 0:
            print(f"    Processed {end}/{n} events, |A|^2 range: [{amp_sq[i:end].min():.6f}, {amp_sq[i:end].max():.6f}]")

    n_low = np.sum(amp_sq < 1e-10)
    if n_low > 0:
        print(f"    WARNING: {n_low} events have |A|^2 < 1e-10, clipping to minimum")
    amp_sq = np.clip(amp_sq, 1e-10, None)
    return amp_sq


def process_one_year(year, year_data):
    """Process one year: 50/50 split, amplitude-correct MC, then split by trigger."""
    # Filter out events with invalid B_TRUETAU
    valid_mask = year_data['B_TRUETAU'] >= 0
    n_invalid = int(np.sum(~valid_mask))
    if n_invalid > 0:
        print(f"  Filtering {n_invalid} events with B_TRUETAU < 0")
        for key in all_vars + ["year"]:
            year_data[key] = year_data[key][valid_mask]
    n_total = len(year_data['B_ConstJpsi_M_1'])
    print(f"  Year {year}: {n_total} valid events")

    # 50/50 split (per-year, independent)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    n_half = n_total // 2
    data_indices = indices[:n_half]
    mc_indices = indices[n_half:]
    print(f"  Split: pseudo_data={n_half}, pseudo_MC={n_total - n_half}")

    # Extract pseudo_data variables
    d = year_data
    data_sw = d["sw_p2vv"][data_indices]
    data_c1 = d["helcosthetaL"][data_indices]
    data_c2 = d["helcosthetaK"][data_indices]
    data_phi = d["helphi"][data_indices]
    data_time = d["time"][data_indices]
    data_sigmat = d["sigmat"][data_indices]
    data_year = d["year"][data_indices]
    data_b_id = d["B_ID_GenLvl"][data_indices]
    data_b_truetau = d["B_TRUETAU"][data_indices] * 1000.

    # Extract pseudo_MC variables
    mc_sw = d["sw_p2vv"][mc_indices]
    mc_c1 = d["helcosthetaL"][mc_indices]
    mc_c2 = d["helcosthetaK"][mc_indices]
    mc_phi = d["helphi"][mc_indices]
    mc_time = d["time"][mc_indices]
    mc_sigmat = d["sigmat"][mc_indices]
    mc_year = d["year"][mc_indices]
    mc_b_id = d["B_ID_GenLvl"][mc_indices]
    mc_b_truetau = d["B_TRUETAU"][mc_indices] * 1000.

    data_b_mass = d["B_ConstJpsi_M_1"][data_indices]
    mc_b_mass = d["B_ConstJpsi_M_1"][mc_indices]

    n_data = len(data_sw)
    n_mc = len(mc_sw)

    # Compute |A_sim|^2 for pseudo_MC
    print(f"  Computing |A_sim|^2 for pseudo_MC (year {year})...")
    amp_sq_mc = compute_amp_sq(mc_c1, mc_c2, mc_phi, mc_b_truetau, mc_b_id)
    mc_sw_corrected = mc_sw / amp_sq_mc

    print(f"  |A_sim|^2 stats: mean={np.mean(amp_sq_mc):.6f}, median={np.median(amp_sq_mc):.6f}")
    print(f"  Corrected weight: mean={np.mean(mc_sw_corrected):.6f}, std={np.std(mc_sw_corrected):.6f}")

    # Sample tag/eta/trigger from real data for this year
    # real data files are per-(year,trigger); we sample from all available triggers
    # by concatenating, then split pseudo events by the sampled trigger value.
    print(f"  Sampling tag/eta/trigger from real data (year {year})...")
    real_tags = []
    real_etas = []
    real_trigs = []
    for trig in trigger_list:
        sub = f"{year}_{trig}"
        try:
            real_tags.append(np.load(f"data_tag_{sub}.npy"))
            real_etas.append(np.load(f"data_eta_{sub}.npy"))
            real_trigs.append(np.load(f"data_trigger_{sub}.npy"))
        except FileNotFoundError:
            print(f"    Warning: data_*_{sub}.npy not found, skipping this trigger source")
    if not real_tags:
        print(f"  ERROR: no real data tag/eta files for year {year}, skipping")
        return False
    real_data_tag = np.concatenate(real_tags)
    real_data_eta = np.concatenate(real_etas)
    real_data_trigger = np.concatenate(real_trigs)

    # Sample for pseudo_data
    idx_data = np.random.randint(real_data_tag.shape[0], size=n_data)
    #data_tag = real_data_tag[idx_data]
    #data_eta = real_data_eta[idx_data]
    data_trigger = real_data_trigger[idx_data]

    # Sample for pseudo_MC
    idx_mc = np.random.randint(real_data_tag.shape[0], size=n_mc)
    #mc_tag = real_data_tag[idx_mc]
    #mc_eta = real_data_eta[idx_mc]
    mc_trigger = real_data_trigger[idx_mc]

    data_tag = np.where(data_b_id > 0, 1, -1).astype(np.int32)
    data_eta = np.zeros(n_data, dtype=np.float64)  
    
    mc_tag = np.where(mc_b_id > 0, 1, -1).astype(np.int32)
    mc_eta = np.zeros(n_mc, dtype=np.float64)  

    # ============================================================
    # Split by trigger value and save each (year, trigger) sub-sample
    # ============================================================
    print(f"\n  Saving per-(year,trigger) sub-sample files (year {year})")
    for trig in trigger_list:
        trig_val = 0 if trig == 'unbiased' else 1
        sub_suffix = f"{year}_{trig}"

        # pseudo_data
        d_trig = data_trigger == trig_val
        n_dt = int(np.sum(d_trig))
        if n_dt == 0:
            print(f"    SKIP {sub_suffix} (pseudo_data): 0 events")
            continue
        np.save(f"pseudo_data_tag_{sub_suffix}.npy", data_tag[d_trig])
        np.save(f"pseudo_data_eta_{sub_suffix}.npy", data_eta[d_trig])
        np.save(f"pseudo_data_weight_{sub_suffix}.npy", data_sw[d_trig])
        np.save(f"pseudo_data_angles_{sub_suffix}.npy",
                np.stack([np.arccos(data_c1[d_trig]),
                          np.arccos(data_c2[d_trig]),
                          data_phi[d_trig]], axis=-1))
        np.save(f"pseudo_data_t_smear_{sub_suffix}.npy", data_time[d_trig])
        np.save(f"pseudo_data_t_resolution_{sub_suffix}.npy", data_sigmat[d_trig])
        np.save(f"pseudo_data_trigger_{sub_suffix}.npy", data_trigger[d_trig])
        np.save(f"pseudo_data_year_{sub_suffix}.npy", data_year[d_trig])
        print(f"    pseudo_data_{sub_suffix}: {n_dt} events")

        # pseudo_MC
        m_trig = mc_trigger == trig_val
        n_mt = int(np.sum(m_trig))
        if n_mt == 0:
            print(f"    SKIP {sub_suffix} (pseudo_MC): 0 events")
            continue
        np.save(f"pseudo_MC_tag_{sub_suffix}.npy", mc_tag[m_trig])
        np.save(f"pseudo_MC_eta_{sub_suffix}.npy", mc_eta[m_trig])
        np.save(f"pseudo_MC_weight_{sub_suffix}.npy", mc_sw_corrected[m_trig])
        np.save(f"pseudo_MC_angles_{sub_suffix}.npy",
                np.stack([np.arccos(mc_c1[m_trig]),
                          np.arccos(mc_c2[m_trig]),
                          mc_phi[m_trig]], axis=-1))
        np.save(f"pseudo_MC_t_smear_{sub_suffix}.npy", mc_time[m_trig])
        np.save(f"pseudo_MC_t_resolution_{sub_suffix}.npy", mc_sigmat[m_trig])
        np.save(f"pseudo_MC_trigger_{sub_suffix}.npy", mc_trigger[m_trig])
        np.save(f"pseudo_MC_year_{sub_suffix}.npy", mc_year[m_trig])
        print(f"    pseudo_MC_{sub_suffix}: {n_mt} events")

        # Generate per-sub-sample distribution plots
        plot_distributions("pseudo_data",
                           data_sw[d_trig], data_sw[d_trig],
                           data_c2[d_trig], data_c1[d_trig], data_phi[d_trig],
                           data_tag[d_trig], data_eta[d_trig], data_trigger[d_trig],
                           data_b_mass[d_trig], sub_suffix)
        plot_distributions("pseudo_MC",
                           mc_sw[m_trig], mc_sw_corrected[m_trig],
                           mc_c2[m_trig], mc_c1[m_trig], mc_phi[m_trig],
                           mc_tag[m_trig], mc_eta[m_trig], mc_trigger[m_trig],
                           mc_b_mass[m_trig], sub_suffix)

    return True


def plot_distributions(dataset_name, sw_raw, sw_corrected, helcosthetaK, helcosthetaL, helphi,
                       tag, eta, trigger, b_constjpsi_mass, sub_suffix):
    """Plot distributions for a single (year, trigger) sub-sample."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

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
    plt.savefig(f'{dataset_name}_distributions_{sub_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Generated plot: {dataset_name}_distributions_{sub_suffix}.png")


years = [2015, 2016, 2017, 2018]
n_ok = 0

for year in years:
    if year not in selected_years:
        print(f"Skipping year {year} (not in selected years)")
        continue
    year_data = load_year_data(year)
    if year_data is None:
        continue
    print(f"\n{'='*60}\nProcessing year {year}\n{'='*60}")
    if process_one_year(year, year_data):
        n_ok += 1

if n_ok == 0:
    print("No data processed!")
    sys.exit(1)

print("\n" + "=" * 60)
print(f"All done! Years processed: {n_ok}")
print("=" * 60)
