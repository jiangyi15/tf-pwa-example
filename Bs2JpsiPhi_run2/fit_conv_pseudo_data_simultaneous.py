"""
Simultaneous fit script for Bs->J/psi Phi analysis (pseudo_data).

Each (year, trigger) is an independent sub-sample with its own PDF and
calibration (baked into the data upstream). Physics parameters (phi_s, etc.)
are shared across all sub-samples via MultiConfig(total_same=True).

Total NLL = sum of per-sub-sample NLL_i.
"""

from tf_pwa.config_loader import MultiConfig
from tf_pwa.amp import time_dep
import tensorflow as tf
import math
import json
import numpy as np
import os
import argparse
import logging
import traceback

from tf_pwa.config_loader.data import MultiData, register_data_mode
from tf_pwa.amp.preprocess import BasePreProcessor, register_preprocessor


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_args(args):
    errors = []

    if not args.years:
        errors.append("--years cannot be empty")
    else:
        try:
            years = [int(y.strip()) for y in args.years.split(',')]
            for y in years:
                if y not in [2015, 2016, 2017, 2018]:
                    errors.append(f"Invalid year: {y}. Must be 2015, 2016, 2017, or 2018")
        except ValueError:
            errors.append(f"Invalid year format: {args.years}")

    if args.trigger not in ['all', 'unbiased', 'biased']:
        errors.append(f"--trigger must be 'all', 'unbiased', or 'biased', got: {args.trigger}")

    if args.batch <= 0:
        errors.append(f"--batch must be positive, got: {args.batch}")

    if not os.path.exists(args.config):
        errors.append(f"Config file not found: {args.config}")

    if errors:
        print("\nInput validation failed:")
        for err in errors:
            print(f"  - {err}")
        raise ValueError("Invalid input parameters")


@register_data_mode("angles")
class Loader(MultiData):
    def load_p4(self, fnames):
        mmap_mode = "r" if self.lazy_file else None
        if fnames.endswith(".npy"):
            p = np.load(fnames).reshape((-1, 3))
        elif fnames.endswith(".npz"):
            p = np.load(fnames).reshape((-1, 3))
        else:
            p = np.loadtxt(fnames).reshape((-1, 3))
        return p


@register_preprocessor("angles")
class AnglesPreprocessor(BasePreProcessor):
    def call(self, data, **kwargs):
        angles = data["p4"]
        decay_chain = self.decay_struct[0].standard_topology()
        zeros = tf.zeros_like(angles[..., 0])
        ret = {
            "particle": {
                decay_chain[0].core: {"m": zeros},
                decay_chain[1].core: {"m": zeros},
                decay_chain[2].core: {"m": zeros},
            },
            "decay": {
                decay_chain: {
                    decay_chain[0]: {
                        decay_chain[0].outs[0]: {"ang": {"beta": zeros}}
                    },
                    decay_chain[1]: {
                        decay_chain[1].outs[0]: {"ang": {"beta": angles[..., 0]}}
                    },
                    decay_chain[2]: {
                        decay_chain[2].outs[0]: {
                            "ang": {"alpha": angles[..., 2], "beta": angles[..., 1]}
                        }
                    }
                }
            },
            "cp_swap": {
                "particle": {
                    decay_chain[0].core: {"m": zeros},
                    decay_chain[1].core: {"m": zeros},
                    decay_chain[2].core: {"m": zeros},
                },
                "decay": {
                    decay_chain: {
                        decay_chain[0]: {
                            decay_chain[0].outs[0]: {"ang": {"beta": zeros}}
                        },
                        decay_chain[1]: {
                            decay_chain[1].outs[0]: {"ang": {"beta": np.pi - angles[..., 0]}}
                        },
                        decay_chain[2]: {
                            decay_chain[2].outs[0]: {
                                "ang": {"alpha": -angles[..., 2], "beta": np.pi - angles[..., 1]}
                            }
                        }
                    }
                }
            }
        }

        for k, v in data["extra"].items():
            ret[k] = v
            ret["cp_swap"][k] = v

        return ret


def fix_selected_params(config, free_params_list=None):
    all_params = config.get_params()

    if free_params_list is None:
        free_params_list = ["Bs_poqi"]

    fixed_params = []
    free_params = []

    if len(free_params_list) == 0:
        return [], list(all_params.keys())

    for name, val in all_params.items():
        if any(fp in name for fp in free_params_list):
            free_params.append(name)
        else:
            config.vm.set_fix(name, val)
            fixed_params.append(name)

    return fixed_params, free_params


def validate_fit_result(config, logger):
    validation_results = {
        "nll_valid": True,
        "errors_reasonable": True,
        "phi_s_consistent": True,
        "warnings": []
    }

    try:
        errors = config.get_params_error(using_cached=True)
        for name, err in errors.items():
            if err > 100 or np.isnan(err) or np.isinf(err):
                validation_results["errors_reasonable"] = False
                validation_results["warnings"].append(f"Unreasonable error for {name}: {err}")
                logger.warning(f"Unreasonable error for {name}: {err}")
    except Exception as e:
        validation_results["errors_reasonable"] = False
        validation_results["warnings"].append(f"Error computing parameter errors: {e}")
        logger.warning(f"Error computing parameter errors: {e}")

    try:
        with config.params_trans() as pt:
            phi_s_val = float(pt["Bs_poqi"].numpy())
            phi_s_err = float(pt.get_error({"phi_s": pt["Bs_poqi"]})["phi_s"])

            expected_phi_s = -0.03
            expected_err = 0.022

            n_sigma = abs(phi_s_val - expected_phi_s) / phi_s_err if phi_s_err > 0 else float('inf')

            if n_sigma > 5:
                validation_results["phi_s_consistent"] = False
                validation_results["warnings"].append(
                    f"phi_s ({phi_s_val:.4f} ± {phi_s_err:.4f}) deviates from expected ({expected_phi_s:.4f} ± {expected_err:.4f}) by {n_sigma:.1f}σ"
                )
                logger.warning(
                    f"phi_s deviates from expected by {n_sigma:.1f}σ: {phi_s_val:.4f} ± {phi_s_err:.4f} vs expected {expected_phi_s:.4f}"
                )
            else:
                logger.info(
                    f"phi_s consistent with expected: {phi_s_val:.4f} ± {phi_s_err:.4f} (expected {expected_phi_s:.4f} ± {expected_err:.4f}, {n_sigma:.1f}σ)"
                )
    except Exception as e:
        validation_results["warnings"].append(f"Error validating phi_s: {e}")
        logger.warning(f"Error validating phi_s: {e}")

    return validation_results


def transform_params(config):
    trans_params = {}
    with config.params_trans() as pt:
        rho0 = pt["Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0r"]
        phi0 = pt["Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0i"]
        rho1 = pt["Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0r"]
        phi1 = pt["Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0i"]
        rho2 = pt["Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0r"]
        phi2 = pt["Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0i"]

        g0 = tf.complex(rho0 * tf.cos(phi0), rho0 * tf.sin(phi0))
        g1 = tf.complex(rho1 * tf.cos(phi1), rho1 * tf.sin(phi1))
        g2 = tf.complex(rho2 * tf.cos(phi2), rho2 * tf.sin(phi2))

        A0 = -g0 * math.sqrt(1/3) + g2 * math.sqrt(2/3)
        Aperp = -g1
        Aparallel = -g0 * math.sqrt(2/3) - g2 * math.sqrt(1/3)

        phi_range = lambda x: (x - 0) % (2*math.pi) + 0
        trans_params["δ⊥ - δ0"] = phi_range(-tf.math.angle(Aperp/A0))
        trans_params["δ∥ - δ0"] = phi_range(-tf.math.angle(Aparallel/A0))

        dom = tf.abs(Aperp)**2 + tf.abs(Aparallel)**2 + tf.abs(A0)**2
        trans_params["|A⊥|^2"] = tf.abs(Aperp)**2 / dom
        trans_params["|A0|^2"] = tf.abs(A0)**2 / dom
        trans_params["|A∥|^2"] = tf.abs(Aparallel)**2 / dom

        trans_params["∆Γ"] = -pt["Bs_delta_gamma"] + 0.0
        trans_params["Γ"] = pt["Bs_gamma"] + 0.0
        trans_params["∆m"] = pt["Bs_delta_m"] + 0.0
        trans_params["production asymmetry"] = pt["Bs_A_prod"] + 0.0
        trans_params["|λ|"] = pt["Bs_poqr"] + 0.0
        trans_params["φ_s"] = pt["Bs_poqi"] + 0.0

    return trans_params, pt


def parse_args():
    parser = argparse.ArgumentParser(description='Simultaneous fit for Bs->J/psi Phi (pseudo_data)')
    parser.add_argument('--years', type=str, default='2015,2016,2017,2018',
                        help='Years to fit, comma-separated (default: 2015,2016,2017,2018)')
    parser.add_argument('--trigger', type=str, default='all',
                        choices=['all', 'unbiased', 'biased'],
                        help='Trigger type to fit (default: all)')
    parser.add_argument('--config', type=str, default='config_conv_pseudo_data_simultaneous.yml',
                        help='Configuration file (default: config_conv_pseudo_data_simultaneous.yml)')
    parser.add_argument('--output', type=str, default='final_params_conv_pseudo_data_simultaneous',
                        help='Output file prefix (default: final_params_conv_pseudo_data_simultaneous)')
    parser.add_argument('--batch', type=int, default=250000,
                        help='Batch size for fitting (default: 250000)')
    parser.add_argument('--log-file', type=str, default='fit_pseudo_data_simultaneous_log.txt',
                        help='Log file path (default: fit_pseudo_data_simultaneous_log.txt)')
    return parser.parse_args()


def create_subsample_configs(base_config_path, years, trigger):
    """Generate one config per (year, trigger) sub-sample.

    Each config points to the per-sub-sample .npy files. Physics parameters
    are shared across all sub-samples via MultiConfig(total_same=True).
    Returns list of temp config paths.
    """
    trigger_list = ['unbiased', 'biased'] if trigger == 'all' else [trigger]

    with open(base_config_path, 'r') as f:
        base_content = f.read()

    # placeholder -> per-sub-sample file name mappings
    # these base names are defined in config_conv_pseudo_data_simultaneous.yml
    base_placeholders = {
        "pseudo_data_angles.npy": "pseudo_data_angles_{sub}.npy",
        "pseudo_data_t_smear.npy": "pseudo_data_t_smear_{sub}.npy",
        "pseudo_data_t_resolution.npy": "pseudo_data_t_resolution_{sub}.npy",
        "pseudo_data_tag.npy": "pseudo_data_tag_{sub}.npy",
        "pseudo_data_eta.npy": "pseudo_data_eta_{sub}.npy",
        "pseudo_data_weight.npy": "pseudo_data_weight_{sub}.npy",
        "pseudo_MC_angles_double.npy": "pseudo_MC_angles_double_{sub}.npy",
        "pseudo_MC_t_smear_double.npy": "pseudo_MC_t_smear_double_{sub}.npy",
        "pseudo_MC_t_resolution_double.npy": "pseudo_MC_t_resolution_double_{sub}.npy",
        "pseudo_MC_tag_double.npy": "pseudo_MC_tag_double_{sub}.npy",
        "pseudo_MC_eta_double.npy": "pseudo_MC_eta_double_{sub}.npy",
        "pseudo_MC_weight_cut_double.npy": "pseudo_MC_weight_cut_double_{sub}.npy",
    }

    temp_paths = []
    for year in years:
        for trig in trigger_list:
            sub = f"{year}_{trig}"
            content = base_content
            for placeholder, template in base_placeholders.items():
                content = content.replace(placeholder, template.format(sub=sub))

            temp_path = f"config_conv_pseudo_data_simultaneous_{sub}.yml"
            with open(temp_path, 'w') as f:
                f.write(content)
            temp_paths.append(temp_path)
            print(f"  Created sub-config: {temp_path}")

    return temp_paths


def main():
    args = parse_args()

    logger = setup_logging(args.log_file)

    logger.info("=" * 70)
    logger.info("Bs->J/psi Phi SIMULTANEOUS Fit (flavour_tag_mix, pseudo_data)")
    logger.info("=" * 70)
    logger.info(f"Command line arguments: {vars(args)}")

    try:
        validate_args(args)
        logger.info("Input validation passed")
    except ValueError as e:
        logger.error(f"Input validation failed: {e}")
        raise

    selected_years = [int(y.strip()) for y in args.years.split(',')]
    selected_trigger = args.trigger
    trigger_list = ['unbiased', 'biased'] if selected_trigger == 'all' else [selected_trigger]

    years_suffix = '_'.join(str(y) for y in selected_years)
    file_suffix = f"simultaneous_{years_suffix}_{selected_trigger}"
    output_file = f"{args.output}_pseudo_data_{file_suffix}.json"

    logger.info(f"Selected years: {selected_years}")
    logger.info(f"Selected trigger: {selected_trigger} -> sub-samples: {trigger_list}")
    logger.info(f"File suffix: {file_suffix}")
    logger.info(f"Output file: {output_file}")

    # Generate per-sub-sample configs
    logger.info("\n" + "=" * 70)
    logger.info("Creating sub-sample configs")
    logger.info("=" * 70)
    temp_config_paths = create_subsample_configs(args.config, selected_years, selected_trigger)
    logger.info(f"Total sub-samples: {len(temp_config_paths)}")

    # Validate required input files for each sub-sample
    required_files = []
    for year in selected_years:
        for trig in trigger_list:
            sub = f"{year}_{trig}"
            required_files.extend([
                f"pseudo_data_angles_{sub}.npy",
                f"pseudo_data_t_smear_{sub}.npy",
                f"pseudo_data_t_resolution_{sub}.npy",
                f"pseudo_data_tag_{sub}.npy",
                f"pseudo_data_eta_{sub}.npy",
                f"pseudo_data_weight_{sub}.npy",
                f"pseudo_MC_angles_double_{sub}.npy",
                f"pseudo_MC_t_smear_double_{sub}.npy",
                f"pseudo_MC_t_resolution_double_{sub}.npy",
                f"pseudo_MC_tag_double_{sub}.npy",
                f"pseudo_MC_eta_double_{sub}.npy",
                f"pseudo_MC_weight_cut_double_{sub}.npy",
            ])
    required_files.append("final_params_conv_pseudo_data_simultaneous.json")

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        logger.error("Missing required input files:")
        for f in missing:
            logger.error(f"  - {f}")
        for tp in temp_config_paths:
            try:
                os.remove(tp)
            except OSError:
                pass
        raise FileNotFoundError(f"Missing {len(missing)} input files")

    logger.info("All required input files found")

    logger.info("\n" + "=" * 70)
    logger.info("Loading MultiConfig (total_same=True: shared physics params)")
    logger.info("=" * 70)

    # MultiConfig with total_same=True: all sub-samples share one amplitude model
    # (physics params + g_ls). Each sub-sample has its own data/MC.
    config = MultiConfig(temp_config_paths, total_same=True)

    logger.info("\nDecay chain structure (from first sub-config):")
    first_config = config.configs[0]
    for i in first_config.get_decay():
        for j in i:
            logger.info(f"  {j}: ls_list={j.get_ls_list()}")

    logger.info("\nLoading initial amplitude parameters...")
    config.set_params("final_params_conv_pseudo_data_simultaneous.json")

    # ============================================================
    # Check parameters against physical results (shared, once)
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
    print("Parameter check completed")
    print("=" * 80 + "\n")

    logger.info("\n" + "=" * 70)
    logger.info("Fixing all parameters except phis")
    logger.info("=" * 70)

    free_params_list = ["Bs_poqi", "Bs_delta_gamma", "Bs_gamma"# "Bs_delta_m"
    #"Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0r",
    #"Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0i",
    #"Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0r",
    #"Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0i",
    #"Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0r",
    #"Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0i"
    ]

    fixed_params, free_params = fix_selected_params(config, free_params_list)
    logger.info(f"Fixed parameters ({len(fixed_params)}):")
    for p in fixed_params[:10]:
        logger.info(f"  - {p}")
    if len(fixed_params) > 10:
        logger.info(f"  ... and {len(fixed_params) - 10} more")

    logger.info(f"Free parameters ({len(free_params)}):")
    for p in free_params:
        logger.info(f"  - {p}")

    with config.params_trans() as pt:
        print("Before fit:")
        print(f"  phi10.total_0r = {float(pt['Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0r'].numpy())}")
        print(f"  phi10.total_0i = {float(pt['Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0i'].numpy())}")
        print(f"  phi11.total_0r = {float(pt['Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0r'].numpy())}")
        print(f"  phi11.total_0i = {float(pt['Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0i'].numpy())}")
        print(f"  phi12.total_0r = {float(pt['Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0r'].numpy())}")
        print(f"  phi12.total_0i = {float(pt['Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0i'].numpy())}")

    logger.info("\n" + "=" * 70)
    logger.info("Starting Simultaneous Fit")
    logger.info("=" * 70)

    try:
        fit_result = config.fit(
            batch=args.batch,
            print_init_nll=False
        )
        logger.info("Fit completed")
    except KeyboardInterrupt:
        logger.warning("Fit interrupted by user")
        config.save_params(f"break_params_conv_{file_suffix}.json")
        for tp in temp_config_paths:
            try:
                os.remove(tp)
            except OSError:
                pass
        raise
    except Exception as e:
        logger.error(f"Fit error: {e}")
        logger.error(traceback.format_exc())
        config.save_params(f"break_params_conv_{file_suffix}.json")
        for tp in temp_config_paths:
            try:
                os.remove(tp)
            except OSError:
                pass
        raise

    with config.params_trans() as pt:
        print("After fit:")
        print(f"  phi10.total_0r = {float(pt['Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0r'].numpy())}")
        print(f"  phi10.total_0i = {float(pt['Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0i'].numpy())}")
        print(f"  phi11.total_0r = {float(pt['Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0r'].numpy())}")
        print(f"  phi11.total_0i = {float(pt['Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0i'].numpy())}")
        print(f"  phi12.total_0r = {float(pt['Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0r'].numpy())}")
        print(f"  phi12.total_0i = {float(pt['Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0i'].numpy())}")

    logger.info("\n" + "=" * 70)
    logger.info("Validating Fit Results")
    logger.info("=" * 70)

    validation_results = validate_fit_result(config, logger)

    logger.info("\nValidation summary:")
    for key, value in validation_results.items():
        if key != "warnings":
            status = "✓" if value else "✗"
            logger.info(f"  {status} {key}: {value}")

    if validation_results["warnings"]:
        logger.warning("\nFit warnings:")
        for warn in validation_results["warnings"]:
            logger.warning(f"  - {warn}")

    logger.info("\n" + "=" * 70)
    logger.info("Extracting Physics Parameters")
    logger.info("=" * 70)

    trans_params, pt = transform_params(config)
    trans_errors = pt.get_error(trans_params)

    ref_params = {
        "δ⊥ - δ0": [3.08, 0.0075],
        "δ∥ - δ0": [3.26, 0.0061],
        "|A⊥|^2": [0.250000, 0.0023],
        "|A0|^2": [0.524176, 0.0017],
        "|A∥|^2": [0.225625, 0.0],
        "∆Γ": [0.08543, 0.0044],
        "Γ": [0.6614, 0.0014],
        "∆m": [17.8, 0.033],
        "production asymmetry": [0, 0],
        "|λ|": [1.0, 0.011],
        "φ_s": [-0.03, 0.022]
    }

    logger.info("\nFit Results:")
    logger.info(f"{'Parameter':<25} {'Value':<12} {'Error':<12} {'Reference':<12} {'n_sigma':<10}")
    logger.info("-" * 70)

    save_params = {}
    for name in trans_params:
        v, e = trans_params[name], trans_errors[name]
        v_f, e_f = float(v), float(e)
        ref_val = ref_params.get(name, [None, None])[0]
        ref_str = f"{ref_val:.5f}" if ref_val is not None else "N/A"

        if e_f > 0 and ref_val is not None:
            n_sigma = abs(v_f - ref_val) / e_f
            n_sigma_str = f"{n_sigma:.2f}σ"
        else:
            n_sigma = None
            n_sigma_str = "-"

        logger.info(f"{name:<25} {v_f:<12.5f} {e_f:<12.5f} {ref_str:<12} {n_sigma_str:<10}")
        save_params[name] = {
            "value": v_f,
            "error": e_f,
            "reference": ref_val,
            "n_sigma": n_sigma
        }

    # save fit_result (raw params)
    try:
        fit_result.extra = {
            "trans_params": save_params,
            "years": selected_years,
            "trigger": selected_trigger,
            "fit_type": "simultaneous flavour_tag_mix (shared physics, fixed except phis/dgamma/gamma)",
            "file_suffix": file_suffix,
            "sub_samples": [f"{y}_{t}" for y in selected_years for t in trigger_list],
            "validation": validation_results,
            "free_params": free_params,
            "fixed_params_count": len(fixed_params)
        }
        fit_result.save_as(output_file)
    except Exception as e:
        logger.warning(f"save_as failed ({e}); falling back to save_params")
        config.save_params(output_file.replace(".json", "_raw.json"))

    with open(f"trans_params_conv_{file_suffix}.json", "w") as f:
        json.dump(save_params, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Transformed parameters saved to: trans_params_conv_{file_suffix}.json")

    # clean up temp configs
    for tp in temp_config_paths:
        try:
            os.remove(tp)
        except OSError:
            pass

    logger.info("\n" + "=" * 70)
    logger.info("Generating Plots")
    logger.info("=" * 70)

    plot_dir = f"figure_conv_pseudo_data_{file_suffix}"
    os.makedirs(plot_dir, exist_ok=True)

    config.plot_partial_wave(plot_pull=True, prefix=f"{plot_dir}/")

    logger.info(f"Plots saved to: {plot_dir}/")

    logger.info("\n" + "=" * 70)
    logger.info("Simultaneous Fit Complete!")
    logger.info("=" * 70)

    return fit_result


if __name__ == "__main__":
    main()
