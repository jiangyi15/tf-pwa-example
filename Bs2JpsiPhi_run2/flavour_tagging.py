"""
Flavour tagging calibration and combination module.

This module implements flavour tagging calibration following LHCb Run2 Bs->J/psi Phi style.
Calibration formula (following RooDalitzTimeCPCBTAG.cxx):
    When tag = +1 (Bs):  eta_cal = (p0 + dp0/2) + (p1 + dp1/2) * (eta - eta_mean)
    When tag = -1 (Bsbar): eta_cal = (p0 - dp0/2) + (p1 - dp1/2) * (eta - eta_mean)

For combined taggers (OS + SS):
    Following C++ CombineTag() logic:
    - If only one tagger fires: use that tagger's calibration
    - If both taggers fire with same decision: combine using product rule
    - If both taggers fire with different decisions: choose lower mistag
"""

import os
import json
import numpy as np


def calibrate_eta(eta, eta_mean, p0, p1, dp0=0.0, dp1=0.0, tag=1):
    """
    Apply calibration with tag-dependent parameters.

    Following RooDalitzTimeCPCBTAG.cxx logic:
        When tag = +1 (Bs):   p0_eff = p0 + dp0/2,  p1_eff = p1 + dp1/2
        When tag = -1 (Bsbar): p0_eff = p0 - dp0/2,  p1_eff = p1 - dp1/2

    Args:
        eta: Raw mistag probability (numpy array)
        eta_mean: Mean eta value for the tagger
        p0, p1: Base calibration parameters
        dp0, dp1: Delta parameters (asymmetry corrections)
        tag: True tag value (+1 for Bs, -1 for Bsbar)

    Returns:
        Calibrated mistag probability
    """
    p0_eff = np.where(tag > 0, p0 + dp0 / 2.0, p0 - dp0 / 2.0)
    p1_eff = np.where(tag > 0, p1 + dp1 / 2.0, p1 - dp1 / 2.0)
    return p0_eff + p1_eff * (eta - eta_mean)


def combine_cali_tag_vectorized(tagos_dec, tagss_dec, tagos_eta, tagss_eta, os_params, ss_params):
    """
    Combine OS and SS taggers with calibration using vectorized operations.

    Following C++ CombineCaliTag() logic:
    - If both taggers return 0: tag=0, eta=0.5
    - If only OS fires: tag=tag_os, eta=calibrated_eta_os
    - If only SS fires: tag=tag_ss, eta=calibrated_eta_ss
    - If both fire with same decision: combine using product rule
    - If both fire with different decisions: choose lower mistag

    Args:
        tagos_dec: OS tag decision array (+1, -1, or 0)
        tagss_dec: SS tag decision array (+1, -1, or 0)
        tagos_eta: OS raw mistag probability array
        tagss_eta: SS raw mistag probability array
        os_params: OS calibration parameters [eta_mean, p0, p1, dp0, dp1]
        ss_params: SS calibration parameters [eta_mean, p0, p1, dp0, dp1]

    Returns:
        tuple: (combined_tag, combined_eta, combined_del_eta)
    """
    # Flatten inputs to ensure 1-dimensional arrays
    tagos_dec = np.asarray(tagos_dec).ravel()
    tagss_dec = np.asarray(tagss_dec).ravel()
    tagos_eta = np.asarray(tagos_eta).ravel()
    tagss_eta = np.asarray(tagss_eta).ravel()

    n = len(tagos_dec)
    tag_dec = np.zeros(n, dtype=np.float64)
    avg_mistag = np.full(n, 0.5, dtype=np.float64)
    del_mistag = np.zeros(n, dtype=np.float64)

    pos = os_params
    pss = ss_params

    mask_both_zero = (tagos_dec == 0) & (tagss_dec == 0)
    tag_dec[mask_both_zero] = 0
    avg_mistag[mask_both_zero] = 0.5
    del_mistag[mask_both_zero] = 0.0

    mask_os_only = (tagos_dec != 0) & (tagss_dec == 0)
    if np.any(mask_os_only):
        os_tag_sub = tagos_dec[mask_os_only]
        eta_cal = calibrate_eta(tagos_eta[mask_os_only], pos[0], pos[1], pos[2], pos[3], pos[4], os_tag_sub)
        dw_cal = calibrate_eta(tagos_eta[mask_os_only], pos[0], pos[3], pos[4], 0.0, 0.0, os_tag_sub)
        avg_mistag[mask_os_only] = np.clip(eta_cal, 0.0, 0.5)
        del_mistag[mask_os_only] = dw_cal
        tag_dec[mask_os_only] = os_tag_sub

    mask_ss_only = (tagos_dec == 0) & (tagss_dec != 0)
    if np.any(mask_ss_only):
        ss_tag_sub = tagss_dec[mask_ss_only]
        eta_cal = calibrate_eta(tagss_eta[mask_ss_only], pss[0], pss[1], pss[2], pss[3], pss[4], ss_tag_sub)
        dw_cal = calibrate_eta(tagss_eta[mask_ss_only], pss[0], pss[3], pss[4], 0.0, 0.0, ss_tag_sub)
        avg_mistag[mask_ss_only] = np.clip(eta_cal, 0.0, 0.5)
        del_mistag[mask_ss_only] = dw_cal
        tag_dec[mask_ss_only] = ss_tag_sub

    mask_both_nonzero = (tagos_dec != 0) & (tagss_dec != 0)
    if np.any(mask_both_nonzero):
        os_eta_sub = tagos_eta[mask_both_nonzero]
        ss_eta_sub = tagss_eta[mask_both_nonzero]
        os_dec_sub = tagos_dec[mask_both_nonzero]
        ss_dec_sub = tagss_dec[mask_both_nonzero]

        mistag1 = calibrate_eta(os_eta_sub, pos[0], pos[1], pos[2], pos[3], pos[4], os_dec_sub)
        mistag2 = calibrate_eta(ss_eta_sub, pss[0], pss[1], pss[2], pss[3], pss[4], ss_dec_sub)
        dw1 = calibrate_eta(os_eta_sub, pos[0], pos[3], pos[4], 0.0, 0.0, os_dec_sub)
        dw2 = calibrate_eta(ss_eta_sub, pss[0], pss[3], pss[4], 0.0, 0.0, ss_dec_sub)

        wp1 = mistag1 + dw1 / 2.
        wp2 = mistag2 + dw2 / 2.
        wm1 = mistag1 - dw1 / 2.
        wm2 = mistag2 - dw2 / 2.

        cbwp = np.zeros(len(os_eta_sub), dtype=np.float64)
        cbwm = np.zeros(len(os_eta_sub), dtype=np.float64)
        tag_sub = np.zeros(len(os_eta_sub), dtype=np.float64)

        mask_same_sign = (os_dec_sub == ss_dec_sub)
        if np.any(mask_same_sign):
            tag_sub[mask_same_sign] = os_dec_sub[mask_same_sign]
            denom = wp1[mask_same_sign] * wp2[mask_same_sign] + \
                    (1. - wp1[mask_same_sign]) * (1. - wp2[mask_same_sign])
            cbwp[mask_same_sign] = (wp1[mask_same_sign] * wp2[mask_same_sign]) / denom
            denom_wm = wm1[mask_same_sign] * wm2[mask_same_sign] + \
                       (1. - wm1[mask_same_sign]) * (1. - wm2[mask_same_sign])
            cbwm[mask_same_sign] = (wm1[mask_same_sign] * wm2[mask_same_sign]) / denom_wm

        mask_diff_sign = ~mask_same_sign
        if np.any(mask_diff_sign):
            mask_os_better = (mistag1[mask_diff_sign] < mistag2[mask_diff_sign])
            idx_os = np.where(mask_diff_sign)[0][mask_os_better]
            idx_ss = np.where(mask_diff_sign)[0][~mask_os_better]

            if len(idx_os) > 0:
                tag_sub[idx_os] = os_dec_sub[idx_os]
                denom = wp1[idx_os] * (1. - wp2[idx_os]) + (1. - wp1[idx_os]) * wp2[idx_os]
                cbwp[idx_os] = (wp1[idx_os] * (1. - wp2[idx_os])) / denom
                denom_wm = wm1[idx_os] * (1. - wm2[idx_os]) + (1. - wm1[idx_os]) * wm2[idx_os]
                cbwm[idx_os] = (wm1[idx_os] * (1. - wm2[idx_os])) / denom_wm

            if len(idx_ss) > 0:
                tag_sub[idx_ss] = ss_dec_sub[idx_ss]
                denom = wp2[idx_ss] * (1. - wp1[idx_ss]) + (1. - wp2[idx_ss]) * wp1[idx_ss]
                cbwp[idx_ss] = (wp2[idx_ss] * (1. - wp1[idx_ss])) / denom
                denom_wm = wm2[idx_ss] * (1. - wm1[idx_ss]) + (1. - wm2[idx_ss]) * wm1[idx_ss]
                cbwm[idx_ss] = (wm2[idx_ss] * (1. - wm1[idx_ss])) / denom_wm

        avg_mistag_sub = (cbwp + cbwm) / 2.
        del_mistag_sub = cbwp - cbwm

        mask_swap = avg_mistag_sub > 0.5
        avg_mistag_sub[mask_swap] = 1. - avg_mistag_sub[mask_swap]
        tag_sub[mask_swap] = -1 * tag_sub[mask_swap]

        tag_dec[mask_both_nonzero] = tag_sub
        avg_mistag[mask_both_nonzero] = np.clip(avg_mistag_sub, 0.0, 0.5)
        del_mistag[mask_both_nonzero] = del_mistag_sub

    return tag_dec, avg_mistag, del_mistag


def load_tagging_params(years, params_dir="params"):
    """
    Load tagging calibration parameters from fit_inputs_{year}.json files.

    Expected format (LHCb Run2 style):
        {
            "TaggingParameters": {
                "TaggingOS": {
                    "Parameter_ETA": {"Name": "eta_OS_Run2", "Value": 0.3546},
                    "Parameter": [
                        {"Name": "p0_OS_2016", "Value": 0.3831},
                        {"Name": "p1_OS_2016", "Value": 0.8518},
                        {"Name": "dp0_OS_2016", "Value": 0.0092},
                        {"Name": "dp1_OS_2016", "Value": 0.0141}
                    ]
                },
                "TaggingSSK": {...}
            }
        }

    For 2015 data, 2016 calibration parameters are used as per convention.

    Args:
        years: List of years to load parameters for
        params_dir: Directory containing fit_inputs JSON files

    Returns:
        dict: {year: {"OS": [eta_mean, p0, p1, dp0, dp1], "SS": [...]}}
    """
    params_by_year = {}

    for year in years:
        json_path = os.path.join(params_dir, f"fit_inputs_{year}.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"fit_inputs_{year}.json not found in {params_dir}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        tagging_params = data.get("TaggingParameters", {})
        os_tagger = tagging_params.get("TaggingOS", {})
        ss_tagger = tagging_params.get("TaggingSSK", {})

        os_aveta = os_tagger.get("Parameter_ETA", {}).get("Value", 0.3546)
        ss_aveta = ss_tagger.get("Parameter_ETA", {}).get("Value", 0.3546)

        os_params = [os_aveta, 0.0, 0.0, 0.0, 0.0]
        ss_params = [ss_aveta, 0.0, 0.0, 0.0, 0.0]

        target_year = year
        if year == 2015:
            target_year = 2016

        os_param_list = os_tagger.get("Parameter", [])
        for param in os_param_list:
            param_name = param["Name"]
            param_value = param["Value"]
            if f"p0_OS_{target_year}" == param_name:
                os_params[1] = param_value
            elif f"p1_OS_{target_year}" == param_name:
                os_params[2] = param_value
            elif f"dp0_OS_{target_year}" == param_name:
                os_params[3] = param_value
            elif f"dp1_OS_{target_year}" == param_name:
                os_params[4] = param_value

        ss_param_list = ss_tagger.get("Parameter", [])
        for param in ss_param_list:
            param_name = param["Name"]
            param_value = param["Value"]
            if f"p0_SSK_{target_year}" == param_name:
                ss_params[1] = param_value
            elif f"p1_SSK_{target_year}" == param_name:
                ss_params[2] = param_value
            elif f"dp0_SSK_{target_year}" == param_name:
                ss_params[3] = param_value
            elif f"dp1_SSK_{target_year}" == param_name:
                ss_params[4] = param_value

        params_by_year[year] = {"OS": os_params, "SS": ss_params}
        print(f"Loaded tagging params for {year} (using {target_year} calibration): OS=[{os_params}], SS=[{ss_params}]")

    return params_by_year


def apply_tagging_calibration(os_tag, ss_tag, os_eta, ss_eta, year, selected_years, params_dir="params"):
    """
    Apply OS/SS tagging calibration and combination for all events.

    Args:
        os_tag: OS tag decision array
        ss_tag: SS tag decision array
        os_eta: OS raw mistag probability array
        ss_eta: SS raw mistag probability array
        year: Year array for each event
        selected_years: List of years to process
        params_dir: Directory containing fit_inputs JSON files

    Returns:
        tuple: (combined_tag, combined_eta, combined_del_eta)
    """
    # Flatten inputs to ensure 1-dimensional arrays
    os_tag = np.asarray(os_tag).ravel()
    ss_tag = np.asarray(ss_tag).ravel()
    os_eta = np.asarray(os_eta).ravel()
    ss_eta = np.asarray(ss_eta).ravel()
    year = np.asarray(year).ravel()

    tagging_params = load_tagging_params(selected_years, params_dir)

    n = len(os_tag)
    tag = np.zeros(n, dtype=np.float64)
    eta = np.full(n, 0.5, dtype=np.float64)
    del_eta = np.zeros(n, dtype=np.float64)

    for y in selected_years:
        year_mask = year == y
        if not np.any(year_mask):
            continue

        os_params = tagging_params[y]["OS"]
        ss_params = tagging_params[y]["SS"]

        tag_sub, eta_sub, del_eta_sub = combine_cali_tag_vectorized(
            os_tag[year_mask], ss_tag[year_mask],
            os_eta[year_mask], ss_eta[year_mask],
            os_params, ss_params
        )

        tag[year_mask] = tag_sub
        eta[year_mask] = eta_sub
        del_eta[year_mask] = del_eta_sub

    return tag, eta, del_eta
