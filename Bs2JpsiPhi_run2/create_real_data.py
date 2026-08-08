import uproot
from tf_pwa.config_loader import ConfigLoader
import numpy as np
from tf_pwa.angle import LorentzVector as lv
import sys
import argparse
import os
import json

from flavour_tagging import apply_tagging_calibration, load_tagging_params
from time_resolution import calibrate_time_resolution, load_time_resolution_params

def parse_args():
    parser = argparse.ArgumentParser(description='Process real data for Bs2JpsiPhi analysis')
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
    "B_SSKaonLatest_TAGDEC", 
    "OS_Combination_DEC", 
    "B_SSKaonLatest_TAGETA", 
    "OS_Combination_ETA", 
    "sigmat", 
    "sw",
    "B_ConstJpsi_M_1"
]

trigger_vars = [
    "B_Hlt1TrackMuonDecision_TOS",
    "B_Hlt1TwoTrackMVADecision_TOS",
    "Jpsi_Hlt1DiMuonHighMassDecision_TOS"
]

def load_year_data(year):
    """Load data for a single year"""
    file_path = f"~/myeos/Bs2JpsiPhi_run2/v4r1_Bs2JpsiPhi_{year}_sw.root"
    print(f"Loading {year} data from {file_path}")
    
    try:
        with uproot.open(file_path) as f:
            t = f.get("DecayTree")
            data = t.arrays(all_vars)
            data = {k: np.array(data[k]) for k in all_vars}
        
        for var in trigger_vars:
            try:
                with uproot.open(file_path) as f:
                    t = f.get("DecayTree")
                    data[var] = np.array(t.arrays([var])[var])
            except Exception as e:
                print(f"Warning: {var} not found for {year}, using default False")
                data[var] = np.zeros(len(data["time"]), dtype=np.bool)
        
        data["year"] = np.full(len(data["time"]), year, dtype=np.int32)
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
        print(f"Loaded {len(year_data['time'])} events for {year}")

if not all_data_list:
    print("No data loaded!")
    sys.exit(1)

print(f"\nMerging {len(all_data_list)} years of data...")

merged_data = {}
for key in all_vars + trigger_vars + ["year"]:
    merged_data[key] = np.concatenate([d[key] for d in all_data_list])

print(f"Total events after merge: {len(merged_data['time'])}")

# Compute valid data mask
os_tag_raw = merged_data["OS_Combination_DEC"]
ss_tag_raw = merged_data["B_SSKaonLatest_TAGDEC"]
os_eta_raw = merged_data["OS_Combination_ETA"]
ss_eta_raw = merged_data["B_SSKaonLatest_TAGETA"]

cut_mask = ~np.isnan(os_tag_raw) & ~np.isinf(os_tag_raw) & \
           ~np.isnan(ss_tag_raw) & ~np.isinf(ss_tag_raw) & \
           ~np.isnan(os_eta_raw) & ~np.isinf(os_eta_raw) & \
           ~np.isnan(ss_eta_raw) & ~np.isinf(ss_eta_raw)

os_tag = os_tag_raw[cut_mask].astype(np.float64)
ss_tag = ss_tag_raw[cut_mask].astype(np.float64)
os_eta = os_eta_raw[cut_mask].astype(np.float64)
ss_eta = ss_eta_raw[cut_mask].astype(np.float64)

os_eta = np.clip(os_eta, 0.0, 0.5)
ss_eta = np.clip(ss_eta, 0.0, 0.5)

time = merged_data["time"][cut_mask]
sigmat = merged_data["sigmat"][cut_mask]
sw = merged_data["sw"][cut_mask]
helcosthetaK = merged_data["helcosthetaK"][cut_mask]
helcosthetaL = merged_data["helcosthetaL"][cut_mask]
helphi = merged_data["helphi"][cut_mask]
year = merged_data["year"][cut_mask]

b_constjpsi_mass = merged_data["B_ConstJpsi_M_1"][cut_mask]

print("\n" + "="*60)
print("Flavour Tagging Calibration and Combination")
print("="*60)

print(f"\nLoading tagging parameters from params/fit_inputs_{selected_years}.json")
tagging_params = load_tagging_params(selected_years)

print("\nApplying OS/SS tagging calibration and combination...")
tag, eta, del_eta = apply_tagging_calibration(
    os_tag, ss_tag, os_eta, ss_eta, year, selected_years
)

print(f"Tagging combination complete.")
print(f"  Tag range: {np.min(tag):.2f} - {np.max(tag):.2f}")
print(f"  Eta range: {np.min(eta):.4f} - {np.max(eta):.4f}")

print("\n" + "="*60)
print("Time Resolution Calibration")
print("="*60)

print(f"\nLoading time resolution parameters from params/fit_inputs_{selected_years}.json")
resolution_params = load_time_resolution_params(selected_years)

print("\nApplying sigma resolution calibration only ...")
sigmat_calibrated = np.zeros(len(sigmat), dtype=np.float64)
for y in selected_years:
    year_mask = year == y
    if not np.any(year_mask):
        continue
    params = resolution_params[y]
    sigmat_calibrated[year_mask] = calibrate_time_resolution(
        sigmat[year_mask],
        res_p0=params["res_p0"],
        res_p1=params["res_p1"]
    )
    print(f"  Year {y}: p0={params['res_p0']:.6f}, p1={params['res_p1']:.6f}")

print(f"Sigma calibration complete.")
print(f"  Original sigma range: {np.min(sigmat):.4f} - {np.max(sigmat):.4f} ps")
print(f"  Calibrated sigma range: {np.min(sigmat_calibrated):.4f} - {np.max(sigmat_calibrated):.4f} ps")

# Apply the same mask to trigger data
trk_tos = merged_data["B_Hlt1TrackMuonDecision_TOS"][cut_mask].astype(np.bool)
twk_tos = merged_data["B_Hlt1TwoTrackMVADecision_TOS"][cut_mask].astype(np.bool)
jpsi_tos = merged_data["Jpsi_Hlt1DiMuonHighMassDecision_TOS"][cut_mask].astype(np.bool)

biased = (trk_tos | twk_tos ) & (~ jpsi_tos)
unbiased = jpsi_tos

trigger = np.where(unbiased, 0, np.where(biased, 1, -1))

valid_trigger_mask = trigger != -1

if selected_trigger == 'unbiased':
    valid_trigger_mask &= (trigger == 0)
    print(f"\nSelecting only unbiased trigger (trigger==0)")
elif selected_trigger == 'biased':
    valid_trigger_mask &= (trigger == 1)
    print(f"\nSelecting only biased trigger (trigger==1)")
else:
    print(f"\nSelecting all trigger types")

print(f"Filtering out {np.sum(~valid_trigger_mask)} events with invalid or unwanted trigger")

time = time[valid_trigger_mask]
sigmat = sigmat[valid_trigger_mask]
sigmat_calibrated = sigmat_calibrated[valid_trigger_mask]
sw = sw[valid_trigger_mask]
helcosthetaK = helcosthetaK[valid_trigger_mask]
helcosthetaL = helcosthetaL[valid_trigger_mask]
helphi = helphi[valid_trigger_mask]
tag = tag[valid_trigger_mask]
eta = eta[valid_trigger_mask]
del_eta = del_eta[valid_trigger_mask]
trigger = trigger[valid_trigger_mask]
year = year[valid_trigger_mask]

b_constjpsi_mass = b_constjpsi_mass[valid_trigger_mask]

os_tag = os_tag[valid_trigger_mask]
ss_tag = ss_tag[valid_trigger_mask]
os_eta = os_eta[valid_trigger_mask]
ss_eta = ss_eta[valid_trigger_mask]

n = len(time)
print(f"\nRemaining events after trigger filter: {n}")

years_suffix = '_'.join(str(y) for y in selected_years)
trigger_suffix = selected_trigger
file_suffix = f"{years_suffix}_{trigger_suffix}"

print("\n" + "="*60)
print("Saving Output Files")
print("="*60)

print("\nSaving time data...")
np.save(f"data_t_smear_{file_suffix}.npy", time)
np.save(f"data_t_resolution_{file_suffix}.npy", sigmat_calibrated)

print("Saving tagging data...")
np.save(f"data_tag_{file_suffix}.npy", tag)
np.save(f"data_eta_{file_suffix}.npy", eta)

print("Saving other data...")
np.save(f"data_weight_{file_suffix}.npy", sw)
np.save(f"data_angles_{file_suffix}.npy", np.stack([np.arccos(helcosthetaL), np.arccos(helcosthetaK), helphi], axis=-1))
np.save(f"data_trigger_{file_suffix}.npy", trigger)
np.save(f"data_year_{file_suffix}.npy", year)

print(f"\nProcessed {n} events total")
print(f"Time range: {np.min(time):.2f} - {np.max(time):.2f} ps")
print(f"Resolution range: {np.min(sigmat_calibrated):.4f} - {np.max(sigmat_calibrated):.4f} ps")
print(f"Tag range: {np.min(tag):.2f} - {np.max(tag):.2f}")
print(f"Eta range: {np.min(eta):.2f} - {np.max(eta):.2f}")
print(f"Output files suffix: {file_suffix}")

print(f"\nGenerated output files:")
print(f"  Time: data_t_smear_{file_suffix}.npy")
print(f"  Time resolution: data_t_resolution_{file_suffix}.npy")
print(f"  Tagging: data_tag_{file_suffix}.npy, data_eta_{file_suffix}.npy")
print(f"  Other: data_weight_{file_suffix}.npy, data_angles_{file_suffix}.npy, data_trigger_{file_suffix}.npy, data_year_{file_suffix}.npy")

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
    gs = GridSpec(3, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    
    ax1.hist(time, bins=100, weights=sw, density=True, alpha=0.7, color='blue')
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Density')
    ax1.set_title('Time distribution (weighted)')
    
    ax2.hist(sigmat_calibrated, bins=50, weights=sw, density=True, alpha=0.7, color='darkgreen')
    ax2.set_xlabel('Time resolution (ps)')
    ax2.set_ylabel('Density')
    ax2.set_title('Calibrated time resolution (weighted)')
    
    ax3.hist(tag, bins=50, density=True, alpha=0.7, color='red')
    ax3.set_xlabel('Tag')
    ax3.set_ylabel('Density')
    ax3.set_title('Combined tagging decision')
    
    ax4.hist(eta, bins=50, density=True, alpha=0.7, color='orange')
    ax4.set_xlabel('Eta')
    ax4.set_ylabel('Density')
    ax4.set_title('Calibrated mistag rate distribution')
    
    ax5.scatter(b_constjpsi_mass, sw, alpha=0.3, s=1, color='purple')
    ax5.set_xlabel(f'm(J/psi K+ K-)')
    ax5.set_ylabel('sWeight')
    ax5.set_title('sWeight vs m(J/psi K+ K-)')
    
    ax6.hist(helcosthetaK, bins=50, weights=sw, density=True, alpha=0.7, color='cyan')
    ax6.set_xlabel('cos(theta_K)')
    ax6.set_ylabel('Density')
    ax6.set_title('Helicity cos(theta_K) distribution (weighted)')
    
    ax7.hist(helcosthetaL, bins=50, weights=sw, density=True, alpha=0.7, color='magenta')
    ax7.set_xlabel('cos(theta_L)')
    ax7.set_ylabel('Density')
    ax7.set_title('Helicity cos(theta_L) distribution (weighted)')
    
    ax8.hist(helphi, bins=50, weights=sw, density=True, alpha=0.7, color='brown')
    ax8.set_xlabel('phi (rad)')
    ax8.set_ylabel('Density')
    ax8.set_title('Helicity phi distribution (weighted)')
   
    ax9.hist(os_eta, bins=50, density=True, alpha=0.7, color='blue', label='OS')
    ax9.hist(ss_eta, bins=50, density=True, alpha=0.7, color='red', label='SS')
    ax9.set_xlabel('eta')
    ax9.set_ylabel('Density')
    ax9.set_title('Raw mistag rate distribution')
    ax9.legend()
    
    plt.tight_layout()
    plt.savefig(f'data_distributions_{file_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGenerated plot: data_distributions_{file_suffix}.png")


plot_distributions()
