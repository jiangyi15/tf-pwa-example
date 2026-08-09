import uproot
import numpy as np
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process Monte Carlo for Bs2JpsiPhi analysis')
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
    "sw_p2vv",
    "B_ConstJpsi_M_1"
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

print(f"Total events after merge: {len(merged_data['B_ConstJpsi_M_1'])}")

sw = merged_data["sw_p2vv"]
helcosthetaK = merged_data["helcosthetaK"]
helcosthetaL = merged_data["helcosthetaL"]
helphi = merged_data["helphi"]
year = merged_data["year"]

b_constjpsi_mass = merged_data["B_ConstJpsi_M_1"]

n = len(b_constjpsi_mass)
print(f"\nTotal events: {n}")

print("\n" + "="*60)
print("Generating trigger, tag, eta (random)")
print("="*60)

if selected_trigger == 'unbiased':
    trigger = np.zeros(n, dtype=np.int32)
elif selected_trigger == 'biased':
    trigger = np.ones(n, dtype=np.int32)
else:
    trigger = np.random.choice([0, 1], n)

tag = np.random.choice([-1, 0, 1], n, p=[0.4, 0.2, 0.4]).astype(np.int32)
eta = np.random.random(n) * 0.5

print(f"  Tag range: {np.min(tag):.2f} - {np.max(tag):.2f}")
print(f"  Eta range: {np.min(eta):.2f} - {np.max(eta):.2f}")
print(f"  Trigger range: {np.min(trigger):.0f} - {np.max(trigger):.0f}")

years_suffix = '_'.join(str(y) for y in selected_years)
trigger_suffix = selected_trigger
file_suffix = f"{years_suffix}_{trigger_suffix}"

print("\n" + "="*60)
print("Saving Output Files")
print("="*60)

print("Saving tagging data...")
np.save(f"MC_tag_{file_suffix}.npy", tag)
np.save(f"MC_eta_{file_suffix}.npy", eta)

print("Saving other data...")
np.save(f"MC_weight_{file_suffix}.npy", sw)
np.save(f"MC_angles_{file_suffix}.npy", np.stack([np.arccos(helcosthetaL), np.arccos(helcosthetaK), helphi], axis=-1))
np.save(f"MC_trigger_{file_suffix}.npy", trigger)
np.save(f"MC_year_{file_suffix}.npy", year)

print(f"\nProcessed {n} events total")
print(f"Tag range: {np.min(tag):.2f} - {np.max(tag):.2f}")
print(f"Eta range: {np.min(eta):.2f} - {np.max(eta):.2f}")
print(f"Output files suffix: {file_suffix}")

print(f"\nGenerated output files:")
print(f"  Tagging: MC_tag_{file_suffix}.npy, MC_eta_{file_suffix}.npy")
print(f"  Other: MC_weight_{file_suffix}.npy, MC_angles_{file_suffix}.npy, MC_trigger_{file_suffix}.npy, MC_year_{file_suffix}.npy")

for y in years:
    y_mask = year == y
    if np.any(y_mask):
        print(f"Year {y}: {np.sum(y_mask)} events")


def plot_distributions():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 14))
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

    ax3.scatter(b_constjpsi_mass, sw, alpha=0.3, s=1, color='purple')
    ax3.set_xlabel('B_ConstJpsi_Mass')
    ax3.set_ylabel('sWeight')
    ax3.set_title('sWeight vs B_ConstJpsi_Mass')

    ax4.hist(helcosthetaK, bins=50, weights=sw, density=True, alpha=0.7, color='cyan')
    ax4.set_xlabel('cos(theta_K)')
    ax4.set_ylabel('Density')
    ax4.set_title('Helicity cos(theta_K) distribution (weighted)')

    ax5.hist(helcosthetaL, bins=50, weights=sw, density=True, alpha=0.7, color='magenta')
    ax5.set_xlabel('cos(theta_L)')
    ax5.set_ylabel('Density')
    ax5.set_title('Helicity cos(theta_L) distribution (weighted)')

    ax6.hist(helphi, bins=50, weights=sw, density=True, alpha=0.7, color='brown')
    ax6.set_xlabel('phi (rad)')
    ax6.set_ylabel('Density')
    ax6.set_title('Helicity phi distribution (weighted)')

    plt.tight_layout()
    plt.savefig(f'MC_distributions_{file_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGenerated plot: MC_distributions_{file_suffix}.png")


plot_distributions()
