"""
Fit script for Bs->J/psi Phi analysis with:
1. flavour_tag_mix model (no time resolution calibration)
2. No flavour tagging calibration (using pre-calibrated inputs)
3. Strict parameter validation
4. All parameters free (full fit)
5. Comprehensive fit result validation
6. Detailed logging and visualization
"""

from tf_pwa.config_loader import ConfigLoader
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


def validate_input_files(file_suffix, required_files):
    missing_files = []
    for fname in required_files:
        if not os.path.exists(fname):
            missing_files.append(fname)
    
    if missing_files:
        print("\nMissing required input files:")
        for fname in missing_files:
            print(f"  - {fname}")
        raise FileNotFoundError("Required input files missing")


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


def fix_all_params_except_phis(config):
    all_params = config.get_params()
    
    phis_params = [
        "Bs_poqi"
    ]
    
    fixed_params = []
    free_params = []
    
    for name, val in all_params.items():
        if any(phis in name for phis in phis_params):
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
            phi_s_val = float(pt["Bs_poqi"]())
            phi_s_err = float(pt.get_error({"phi_s": pt["Bs_poqi"]})["phi_s"])
            
            expected_phi_s = -0.039
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
    parser = argparse.ArgumentParser(description='Fit script for Bs->J/psi Phi analysis (flavour_tag_mix)')
    parser.add_argument('--years', type=str, default='2015,2016,2017,2018',
                        help='Years to fit, comma-separated (default: 2015,2016,2017,2018)')
    parser.add_argument('--trigger', type=str, default='all',
                        choices=['all', 'unbiased', 'biased'],
                        help='Trigger type to fit (default: all)')
    parser.add_argument('--config', type=str, default='config_conv_Monte_Carlo.yml',
                        help='Configuration file (default: config_conv_Monte_Carlo.yml)')
    parser.add_argument('--output', type=str, default='final_params_conv_Monte_Carlo',
                        help='Output file prefix (default: final_params_conv_Monte_Carlo)')
    parser.add_argument('--batch', type=int, default=250000,
                        help='Batch size for fitting (default: 250000)')
    parser.add_argument('--log-file', type=str, default='fit_Monte_Carlo_log.txt',
                        help='Log file path (default: fit_Monte_Carlo_log.txt)')
    return parser.parse_args()


def create_combined_config(base_config_path, years, trigger):
    years_str = '_'.join(str(y) for y in years)
    suffix = f"{years_str}_{trigger}"
    
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    mappings = {
        "data_angles.npy": f"data_angles_{suffix}.npy",
        "data_t_smear.npy": f"data_t_smear_{suffix}.npy",
        "data_t_resolution.npy": f"data_t_resolution_{suffix}.npy",
        "data_tag.npy": f"data_tag_{suffix}.npy",
        "data_eta.npy": f"data_eta_{suffix}.npy",
        "data_weight.npy": f"data_weight_{suffix}.npy",
        "data_trigger.npy": f"data_trigger_{suffix}.npy",
        "data_year.npy": f"data_year_{suffix}.npy",
        "MC_angles_double.npy": f"MC_angles_double_{suffix}.npy",
        "MC_t_smear_double.npy": f"MC_t_smear_double_{suffix}.npy",
        "MC_t_resolution_double.npy": f"MC_t_resolution_double_{suffix}.npy",
        "MC_tag_double.npy": f"MC_tag_double_{suffix}.npy",
        "MC_eta_double.npy": f"MC_eta_double_{suffix}.npy",
        "MC_weight_cut_double.npy": f"MC_weight_cut_double_{suffix}.npy",
        "MC_trigger_double.npy": f"MC_trigger_double_{suffix}.npy",
        "MC_year_double.npy": f"MC_year_double_{suffix}.npy",
    }
    
    for old_name, new_name in mappings.items():
        config_content = config_content.replace(old_name, new_name)
    
    temp_config_path = f"config_conv_Monte_Carlo_{suffix}.yml"
    with open(temp_config_path, 'w') as f:
        f.write(config_content)
    
    return temp_config_path


def main():
    args = parse_args()
    
    logger = setup_logging(args.log_file)
    
    logger.info("=" * 70)
    logger.info("Bs->J/psi Phi Fit (flavour_tag_mix model)")
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
    
    years_suffix = '_'.join(str(y) for y in selected_years)
    file_suffix = f"{years_suffix}_{selected_trigger}"
    output_file = f"{args.output}_Monte_Carlo_{file_suffix}.json"
    
    logger.info(f"Selected years: {selected_years}")
    logger.info(f"Selected trigger: {selected_trigger}")
    logger.info(f"File suffix: {file_suffix}")
    logger.info(f"Output file: {output_file}")
    
    temp_config_path = create_combined_config(args.config, selected_years, selected_trigger)
    logger.info(f"Created temporary config: {temp_config_path}")
    
    required_files = [
        f"data_angles_{file_suffix}.npy",
        f"data_t_smear_{file_suffix}.npy",
        f"data_t_resolution_{file_suffix}.npy",
        f"data_tag_{file_suffix}.npy",
        f"data_eta_{file_suffix}.npy",
        f"data_weight_{file_suffix}.npy",
        f"MC_angles_double_{file_suffix}.npy",
        f"MC_t_smear_double_{file_suffix}.npy",
        f"MC_t_resolution_double_{file_suffix}.npy",
        f"MC_tag_double_{file_suffix}.npy",
        f"MC_eta_double_{file_suffix}.npy",
        f"MC_weight_cut_double_{file_suffix}.npy",
        "final_params_Monte_Carlo.json",
    ]
    
    try:
        validate_input_files(file_suffix, required_files)
        logger.info("All required input files found")
    except FileNotFoundError as e:
        logger.error(f"Missing input files: {e}")
        os.remove(temp_config_path)
        raise
    
    logger.info("\n" + "=" * 70)
    logger.info("Loading configuration")
    logger.info("=" * 70)
    
    config = ConfigLoader(temp_config_path)
    
    logger.info("\nDecay chain structure:")
    for i in config.get_decay():
        for j in i:
            logger.info(f"  {j}: ls_list={j.get_ls_list()}")
    
    logger.info("\nLoading initial amplitude parameters...")
    config.set_params("final_params_Monte_Carlo.json")
    
    logger.info("\n" + "=" * 70)
    logger.info("Fixing all parameters except phis")
    logger.info("=" * 70)
    
    fixed_params, free_params = fix_all_params_except_phis(config)
    logger.info(f"Fixed parameters ({len(fixed_params)}):")
    for p in fixed_params[:10]:
        logger.info(f"  - {p}")
    if len(fixed_params) > 10:
        logger.info(f"  ... and {len(fixed_params) - 10} more")
    
    logger.info(f"Free parameters ({len(free_params)}):")
    for p in free_params:
        logger.info(f"  - {p}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Starting Fit")
    logger.info("=" * 70)
    
    try:
        fit_result = config.fit(
            batch=args.batch,
            print_init_nll=False,
            improve=True
        )
        logger.info("Fit completed")
    except KeyboardInterrupt:
        logger.warning("Fit interrupted by user")
        config.save_params(f"break_params_conv_{file_suffix}.json")
        os.remove(temp_config_path)
        raise
    except Exception as e:
        logger.error(f"Fit error: {e}")
        logger.error(traceback.format_exc())
        config.save_params(f"break_params_conv_{file_suffix}.json")
        os.remove(temp_config_path)
        raise
    
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
        "δ⊥ - δ0": [2.903, 0.0075],
        "δ∥ - δ0": [3.146, 0.0061],
        "|A⊥|^2": [0.2463, 0.0023],
        "|A0|^2": [0.5179, 0.0017],
        "|A∥|^2": [1.0 - 0.2463 - 0.5179, 0.0],
        "∆Γ": [0.0845, 0.0044],
        "Γ": [-0.0056, 0.0014],
        "∆m": [17.743, 0.033],
        "production asymmetry": [0, 0],
        "|λ|": [1.001, 0.011],
        "φ_s": [-0.039, 0.022]
    }
    
    logger.info("\nFit Results:")
    logger.info(f"{'Parameter':<25} {'Value':<12} {'Error':<12} {'Reference':<12}")
    logger.info("-" * 70)
    
    save_params = {}
    for name in trans_params:
        v, e = trans_params[name], trans_errors[name]
        ref_val = ref_params.get(name, [None, None])[0]
        ref_str = f"{ref_val:.5f}" if ref_val is not None else "N/A"
        logger.info(f"{name:<25} {float(v):<12.5f} {float(e):<12.5f} {ref_str:<12}")
        save_params[name] = {
            "value": float(v),
            "error": float(e),
            "reference": ref_val
        }
    
    fit_result.extra = {
        "trans_params": save_params,
        "years": selected_years,
        "trigger": selected_trigger,
        "fit_type": "flavour_tag_mix (fixed params except phis)",
        #"fit_type": "flavour_tag_mix (all parameters free)",
        "file_suffix": file_suffix,
        "validation": validation_results,
        "free_params": free_params,
        "fixed_params_count": len(fixed_params)
    }
    
    fit_result.save_as(output_file)
    
    with open(f"trans_params_conv_{file_suffix}.json", "w") as f:
        json.dump(save_params, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Transformed parameters saved to: trans_params_conv_{file_suffix}.json")
    
    os.remove(temp_config_path)
    
    logger.info("\n" + "=" * 70)
    logger.info("Generating Plots")
    logger.info("=" * 70)
    
    plot_dir = f"figure_conv_Monte_Carlo_{file_suffix}"
    os.makedirs(plot_dir, exist_ok=True)
    
    config.plot_partial_wave(plot_pull=True, prefix=f"{plot_dir}/")
    config.plot_partial_wave(plot_pull=True, prefix=f"{plot_dir}/tagp_", 
                              cut_function=lambda x: x["tag"] > 0)
    config.plot_partial_wave(plot_pull=True, prefix=f"{plot_dir}/tagm_", 
                              cut_function=lambda x: x["tag"] < 0)
    
    logger.info(f"Plots saved to: {plot_dir}/")
    
    logger.info("\n" + "=" * 70)
    logger.info("Fit Complete!")
    logger.info("=" * 70)
    
    return fit_result


if __name__ == "__main__":
    main()
