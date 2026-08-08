"""
Time resolution calibration module.

This module implements time resolution calibration following LHCb Run2 Bs->J/psi Phi style.
Calibration formula:
    sigma_eff = res_p0 + res_p1 * sigma
    t_eff = t - time_bias

The effective sigma is used in the convolution with the decay amplitude.
"""

import os
import json
import numpy as np


def calibrate_time_resolution(sigma, res_p0=1.0, res_p1=0.0, time_bias=0.0):
    """
    Apply time resolution calibration.

    The effective sigma is: sigma_eff = res_p0 + res_p1 * sigma
    The effective time is: t_eff = t - time_bias

    Args:
        sigma: Per-event time uncertainty array (numpy array)
        res_p0: Resolution scale parameter 0 (default 1.0)
        res_p1: Resolution scale parameter 1 (default 0.0)
        time_bias: Time bias (default 0.0, included for completeness)

    Returns:
        Calibrated sigma array
    """
    sigma_eff = res_p0 + res_p1 * sigma
    return sigma_eff


def apply_time_bias(t, time_bias=0.0):
    """
    Apply time bias correction.

    Args:
        t: Measured time array (numpy array)
        time_bias: Time bias to subtract (default 0.0)

    Returns:
        Time array with bias subtracted
    """
    return t - time_bias


def load_time_resolution_params(years, params_dir="params"):
    """
    Load time resolution parameters from fit_inputs_{year}.json files.

    Expected format:
        {
            "TimeResParameters": [
                {"Name": "p0", "Value": 1.0, "Error": 0.01},
                {"Name": "p1", "Value": 0.0, "Error": 0.01},
                {"Name": "rho_p0_p1_time_res", "Value": 0.0, "Error": 0.0}
            ],
            "TimeBias": {
                "name": "meanshift",
                "Value": 0.0,
                "Error": 0.001
            }
        }

    Args:
        years: List of years to load parameters for
        params_dir: Directory containing fit_inputs JSON files

    Returns:
        dict: {year: {"res_p0": float, "res_p1": float, "time_bias": float, ...}}
    """
    params_by_year = {}

    for year in years:
        json_path = os.path.join(params_dir, f"fit_inputs_{year}.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"fit_inputs_{year}.json not found in {params_dir}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        params = {
            "res_p0": 1.0,
            "res_p1": 0.0,
            "time_bias": 0.0,
            "res_p0_err": 0.0,
            "res_p1_err": 0.0,
            "time_bias_err": 0.0,
            "res_corr": 0.0
        }

        if "TimeResParameters" in data:
            for param in data["TimeResParameters"]:
                if param["Name"] == "p0":
                    params["res_p0"] = param["Value"]
                    params["res_p0_err"] = param["Error"]
                elif param["Name"] == "p1":
                    params["res_p1"] = param["Value"]
                    params["res_p1_err"] = param["Error"]
                elif param["Name"] == "rho_p0_p1_time_res":
                    params["res_corr"] = param["Value"]

        if "TimeBias" in data:
            params["time_bias"] = data["TimeBias"]["Value"]
            params["time_bias_err"] = data["TimeBias"]["Error"]

        params_by_year[year] = params
        print(f"Loaded time resolution params for {year}: res_p0={params['res_p0']:.6f}, "
              f"res_p1={params['res_p1']:.6f}, time_bias={params['time_bias']:.6f}")

    return params_by_year


def apply_time_resolution_calibration(sigma, t, year, selected_years, params_dir="params"):
    """
    Apply time resolution calibration for all events.

    Args:
        sigma: Raw per-event time uncertainty array
        t: Measured time array
        year: Year array for each event
        selected_years: List of years to process
        params_dir: Directory containing fit_inputs JSON files

    Returns:
        tuple: (calibrated_sigma, calibrated_t)
    """
    # Flatten inputs to ensure 1-dimensional arrays
    sigma = np.asarray(sigma).ravel()
    t = np.asarray(t).ravel()
    year = np.asarray(year).ravel()

    resolution_params = load_time_resolution_params(selected_years, params_dir)

    n = len(sigma)
    sigma_calibrated = np.zeros(n, dtype=np.float64)
    t_calibrated = np.zeros(n, dtype=np.float64)

    for y in selected_years:
        year_mask = year == y
        if not np.any(year_mask):
            continue

        params = resolution_params[y]
        sigma_calibrated[year_mask] = calibrate_time_resolution(
            sigma[year_mask],
            res_p0=params["res_p0"],
            res_p1=params["res_p1"]
        )
        t_calibrated[year_mask] = apply_time_bias(
            t[year_mask],
            time_bias=params["time_bias"]
        )

    return sigma_calibrated, t_calibrated
