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
from csp_factors import (
    load_csp_factors_baseline,
    compute_csp_for_events_baseline,
    multiply_csp_into_weights,
)

def parse_args():
    parser = argparse.ArgumentParser(description='Process real data for Bs2JpsiPhi analysis (simultaneous fit)')
    parser.add_argument('--years', type=str, default='2015,2016,2017,2018',
                        help='Years to process, comma-separated (default: 2015,2016,2017,2018)')
    parser.add_argument('--trigger', type=str, default='all',
                        choices=['all', 'unbiased', 'biased'],
                        help='Trigger type to select (default: all)')
    return parser.parse_args()

args = parse_args()

selected_years = [int(y.strip()) for y in args.years.split(',')]
selected_trigger = args.trigger

# simultaneous fit: each (year, trigger) is an independent sub-sample
trigger_list = ['unbiased', 'biased'] if selected_trigger == 'all' else [selected_trigger]

print(f"Selected years: {selected_years}")
print(f"Selected trigger: {selected_trigger} -> sub-samples: {trigger_list}")

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
    "B_ConstJpsi_M_1",
    "X_M",
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


def process_one_year(year, data):
    """Calibrate and save one year as per-(year,trigger) sub-samples.

    No cross-year merge: each year is calibrated independently and split by
    trigger value into separate files (simultaneous fit sub-samples).
    """
    # Compute valid data mask
    os_tag_raw = data["OS_Combination_DEC"]
    ss_tag_raw = data["B_SSKaonLatest_TAGDEC"]
    os_eta_raw = data["OS_Combination_ETA"]
    ss_eta_raw = data["B_SSKaonLatest_TAGETA"]

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

    time = data["time"][cut_mask]
    sigmat = data["sigmat"][cut_mask]
    sw = data["sw"][cut_mask]
    helcosthetaK = data["helcosthetaK"][cut_mask]
    helcosthetaL = data["helcosthetaL"][cut_mask]
    helphi = data["helphi"][cut_mask]
    year_arr = data["year"][cut_mask]
    b_constjpsi_mass = data["B_ConstJpsi_M_1"][cut_mask]
    x_m = data["X_M"][cut_mask]

    print("\n" + "=" * 60)
    print(f"Flavour Tagging Calibration (year {year})")
    print("=" * 60)

    print(f"\nLoading tagging parameters from params/fit_inputs_{selected_years}.json")
    tagging_params = load_tagging_params(selected_years)

    print("\nApplying OS/SS tagging calibration and combination...")
    tag, eta, del_eta = apply_tagging_calibration(
        os_tag, ss_tag, os_eta, ss_eta, year_arr, selected_years
    )

    print(f"Tagging combination complete (year {year}).")
    print(f"  Tag range: {np.min(tag):.2f} - {np.max(tag):.2f}")
    print(f"  Eta range: {np.min(eta):.4f} - {np.max(eta):.4f}")

    print("\n" + "=" * 60)
    print(f"Time Resolution Calibration (year {year})")
    print("=" * 60)

    print(f"\nLoading time resolution parameters from params/fit_inputs_{selected_years}.json")
    resolution_params = load_time_resolution_params(selected_years)

    print("\nApplying sigma resolution calibration only ...")
    params = resolution_params[year]
    sigmat_calibrated = calibrate_time_resolution(
        sigmat,
        res_p0=params["res_p0"],
        res_p1=params["res_p1"]
    )
    print(f"  Year {year}: p0={params['res_p0']:.6f}, p1={params['res_p1']:.6f}")
    print(f"  Calibrated sigma range: {np.min(sigmat_calibrated):.4f} - {np.max(sigmat_calibrated):.4f} ps")

    # Compute trigger type per event
    trk_tos = data["B_Hlt1TrackMuonDecision_TOS"][cut_mask].astype(np.bool)
    twk_tos = data["B_Hlt1TwoTrackMVADecision_TOS"][cut_mask].astype(np.bool)
    jpsi_tos = data["Jpsi_Hlt1DiMuonHighMassDecision_TOS"][cut_mask].astype(np.bool)

    biased = (trk_tos | twk_tos) & (~jpsi_tos)
    unbiased = jpsi_tos

    trigger = np.where(unbiased, 0, np.where(biased, 1, -1))

    print("\n" + "=" * 60)
    print(f"Csp Factor Computation (year {year})")
    print("=" * 60)

    csp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params", "Csp")
    csp_factors = np.ones_like(sw, dtype=np.float64)
    if not os.path.isdir(csp_dir):
        print(f"Warning: Csp directory not found at {csp_dir}, skipping Csp correction")
    else:
        print(f"\nLoading Baseline Csp factors from {csp_dir}/CspFactorsAll.json")
        csp_factors_baseline = load_csp_factors_baseline(csp_dir=csp_dir)
        bin_summary = ", ".join(
            f"[{f['Bin_ll']}-{f['Bin_ul']}]={f['Value']:.4f}" for f in csp_factors_baseline
        )
        print(f"  Baseline: {bin_summary}")

        csp_factors = compute_csp_for_events_baseline(x_m, csp_factors_baseline)
        print(f"  Csp factor range: {np.min(csp_factors):.4f} - {np.max(csp_factors):.4f}")
        sw = multiply_csp_into_weights(sw, csp_factors)
        print(f"  Weight range after Csp: {np.min(sw):.6f} - {np.max(sw):.6f}")

    # ============================================================
    # Split by trigger and save each (year, trigger) sub-sample
    # ============================================================
    print("\n" + "=" * 60)
    print(f"Saving per-(year,trigger) sub-sample files (year {year})")
    print("=" * 60)

    for trig in trigger_list:
        trig_val = 0 if trig == 'unbiased' else 1
        trig_mask = (trigger == trig_val)

        n_trig = int(np.sum(trig_mask))
        if n_trig == 0:
            print(f"  SKIP {year}_{trig}: 0 events")
            continue

        sub_suffix = f"{year}_{trig}"
        print(f"\n  [{sub_suffix}] {n_trig} events")

        np.save(f"data_t_smear_{sub_suffix}.npy", time[trig_mask])
        np.save(f"data_t_resolution_{sub_suffix}.npy", sigmat_calibrated[trig_mask])
        np.save(f"data_tag_{sub_suffix}.npy", tag[trig_mask])
        np.save(f"data_eta_{sub_suffix}.npy", eta[trig_mask])
        np.save(f"data_weight_{sub_suffix}.npy", sw[trig_mask])
        np.save(f"data_csp_{sub_suffix}.npy", csp_factors[trig_mask])
        np.save(f"data_angles_{sub_suffix}.npy",
                np.stack([np.arccos(helcosthetaL[trig_mask]),
                          np.arccos(helcosthetaK[trig_mask]),
                          helphi[trig_mask]], axis=-1))
        np.save(f"data_trigger_{sub_suffix}.npy", trigger[trig_mask])
        np.save(f"data_year_{sub_suffix}.npy", year_arr[trig_mask])
        print(f"    Saved data_*_{sub_suffix}.npy ({n_trig} events)")

    print(f"\nYear {year} done.")


years = [2015, 2016, 2017, 2018]
n_ok = 0

for year in years:
    if year not in selected_years:
        print(f"Skipping year {year} (not in selected years)")
        continue
    year_data = load_year_data(year)
    if year_data is None:
        print(f"Failed to load year {year}, skipping")
        continue
    print(f"Loaded {len(year_data['time'])} events for {year}")
    process_one_year(year, year_data)
    n_ok += 1

if n_ok == 0:
    print("No data loaded!")
    sys.exit(1)

print("\n" + "=" * 60)
print(f"All done! Years processed: {n_ok}")
print(f"Sub-samples: {[(y, t) for y in selected_years for t in trigger_list]}")
print("=" * 60)
