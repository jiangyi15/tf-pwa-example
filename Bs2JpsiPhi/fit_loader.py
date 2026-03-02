from tf_pwa.config_loader import ConfigLoader
from tf_pwa.amp import time_dep
import tensorflow as tf
import math
import json

import numpy as np

from tf_pwa.config_loader.data import MultiData, register_data_mode
from tf_pwa.amp.preprocess import BasePreProcessor, register_preprocessor

@register_data_mode("angles")
class Loader(MultiData):
    def load_p4(self, fnames):
        particles = self.get_dat_order()
        mmap_mode = "r" if self.lazy_file else None
        if fnames.endswith(".npy"):
            p = np.load(fnames).reshape((-1,3))
        elif fnames.endswith(".npz"):
            p = np.load(fnames).reshape((-1,3))
        else:
            p = np.loadtxt(fnames).reshape((-1,3))
        return p

@register_preprocessor("angles")
class AnglesPreprocessor(BasePreProcessor):
    def call(self, data, **kwargs):
        angles = data["p4"]
        decay_chain = self.decay_struct[0].standard_topology()
        #print(decay_chain)
        #print(f"{angles=}")
        zeros = tf.zeros_like(angles[...,0])
        ret = {"particle": {
                    decay_chain[0].core: {"m": zeros},
                    decay_chain[1].core: {"m": zeros},
                    decay_chain[2].core: {"m": zeros},
                },
               "decay":
                {decay_chain: {
                    decay_chain[0]: {
                        decay_chain[0].outs[0]: {
                            "ang": {"beta": zeros}}
                    },
                    decay_chain[1]: {
                        decay_chain[1].outs[0]: {
                            "ang": {
                                "beta": angles[...,0]
                            }
                        }
                    },
                    decay_chain[2]: {
                        decay_chain[2].outs[0]: {
                            "ang": {
                                "alpha": angles[...,2],
                                "beta": angles[...,1]
                            }
                        }
                    }
                }
            },
            "cp_swap":  {"particle": {
                    decay_chain[0].core: {"m": zeros},
                    decay_chain[1].core: {"m": zeros},
                    decay_chain[2].core: {"m": zeros},},
               "decay":
                {decay_chain: {
                    decay_chain[0]: {
                        decay_chain[0].outs[0]: {
                            "ang": {"beta": zeros}}
                        },
                    decay_chain[1]: {
                        decay_chain[1].outs[0]: {
                            "ang": {
                                "beta": np.pi-angles[...,0]
                            }
                        }
                    },
                    decay_chain[2]: {
                        decay_chain[2].outs[0]: {
                            "ang": {
                                "alpha": -angles[...,2],
                                "beta": np.pi-angles[...,1]
                            }
                        }
                    }
                            }
            }
            }
        }
        #print(ret)
        for k, v in data["extra"].items():
            ret[k] = v
            ret["cp_swap"][k] = v
        return ret


config = ConfigLoader("config_loader.yml")
for i in config.get_decay():
    for j in i:
        print(j, j.get_ls_list())
config.set_params("final_params_amp.json")

# config.set_params({"Bs_delta_m": 17.979644571482126, "Bs_gamma": 0.6413414206046909})

try:
    fit_result = config.fit(batch=250000, print_init_nll=False)
except KeyboardInterrupt:
    config.save_params("break_params.json")
    raise
except Exception as e:
    print(e)
    config.save_params("break_params.json")
    raise
config.get_params_error(using_cached=True)
fit_result.save_as("final_params_loader.json")

trans_params = {}
with config.params_trans() as pt:
    rho0 = pt["Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0r"]
    phi0 = pt["Bs->Jpsi.phi10Jpsi->mup.mumphi10->Kp.Km_total_0i"]
    rho1 = pt["Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0r"]
    phi1 = pt["Bs->Jpsi.phi11Jpsi->mup.mumphi11->Kp.Km_total_0i"]
    rho2 = pt["Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0r"]
    phi2 = pt["Bs->Jpsi.phi12Jpsi->mup.mumphi12->Kp.Km_total_0i"]
    # rhos = pt["Bs->Jpsi.phi0Jpsi->mup.mumphi0->Kp.Km_total_0r"]
    # phis = pt["Bs->Jpsi.phi0Jpsi->mup.mumphi0->Kp.Km_total_0i"]
    # gs = tf.complex(rhos * tf.cos(phis), rhos * tf.sin(phis))
    g0 = tf.complex(rho0 * tf.cos(phi0), rho0 * tf.sin(phi0))
    g1 = tf.complex(rho1 * tf.cos(phi1), rho1 * tf.sin(phi1))
    g2 = tf.complex(rho2 * tf.cos(phi2), rho2 * tf.sin(phi2))
    # As = gs / math.sqrt(3)
    A0 = - g0 * math.sqrt(1/3) + g2 * math.sqrt(2/3)
    Aperp = -g1
    Aparallel = -g0 * math.sqrt(2/3) - g2 * math.sqrt(1/3)
    phi_range = lambda x: (x - 0)% (2*math.pi) + 0
    trans_params["δ⊥ - δ0"] = phi_range(-tf.math.angle(Aperp/A0))
    trans_params["δ∥ - δ0"] = phi_range(-tf.math.angle(Aparallel/A0))
    # trans_params["delta_s-delta_0"] = phi_range(tf.math.angle(As/A0))
    dom = tf.abs(Aperp)**2  + tf.abs(Aparallel)**2 + tf.abs(A0)**2
    trans_params["|A⊥|^2"] = tf.abs(Aperp)**2/dom
    trans_params["|A0|^2"] = tf.abs(A0)**2/dom
    # trans_params["|A_S|"] = tf.abs(As)
    trans_params["|A∥|^2"] = tf.abs(Aparallel)**2/dom
    trans_params["∆Γ"] = -pt["Bs_delta_gamma"] +0.
    trans_params["Γ"] = pt["Bs_gamma"]+0.
    trans_params["∆m"] = pt["Bs_delta_m"]+0.
    trans_params["production asymmetry"] = pt["Bs_A_prod"]+0.
    trans_params["|λ|"] = pt["Bs_poqr"]+0.
    trans_params["φ_s"] = pt["Bs_poqi"]+0.

trans_errors = pt.get_error(trans_params)

ref_params = {
    "δ⊥ - δ0":  [2.903, 0.0075],
    "δ∥ - δ0": [3.146, 0.0061],
    "|A⊥|^2": [0.2463, 0.0023],
    "|A0|^2": [0.5179, 0.0017],
    "|A∥|^2": [ 1.0 - 0.2463-0.5179, 0.0 ],
    "∆Γ": [0.0845, 0.0044],
    "Γ": [-0.0056, 0.0014],
    "∆m": [17.743, 0.033],
    "production asymmetry": [0,0],
    "|λ|": [1.001, 0.011],
    "φ_s": [-0.039, 0.022]
}
comments = {
    "δ⊥ - δ0":  "(δ0-δ⊥?)",
    "δ∥ - δ0": "(δ0- δ∥?)",
    "|A⊥|^2": "",
    "|A0|^2": "",
    "|A∥|^2": "",
    "∆Γ": "(-∆Γ)",
    "Γ": " not Γs - Γd",
    "∆m": "",
    "production asymmetry": "",
    "|λ|": "",
    "φ_s": "",
}
latex_name  = {
    "δ⊥ - δ0":  r"$\delta_\perp - \delta_0$",
    "δ∥ - δ0": r"$\delta_\parallel - \delta_0$",
    "|A⊥|^2": r"$|A_\perp|^2$",
    "|A0|^2": r"$|A_0|^2$",
    "|A∥|^2": r"$|A_\parallel|^2$",
    "∆Γ": r"$\Delta\Gamma$",
    "Γ": r"$\Gamma$",
    "∆m": r"$\Delta m$",
    "production asymmetry": "production asymmetry",
    "|λ|": r"$|\lambda|$",
    "φ_s": r"$\phi_s$",
}

latex_comments = {
    "δ⊥ - δ0":  r"$\delta_0 - \delta_\perp$?",
    "δ∥ - δ0":  r"$\delta_0 - \delta_\perp$?",
    "|A⊥|^2": "",
    "|A0|^2": "",
    "|A∥|^2": "",
    "∆Γ": r"$-\Delta \Gamma$",
    "Γ": r" not $\Gamma_s - \Gamma_d$",
    "∆m": "",
    "production asymmetry": "",
    "|λ|": "",
    "φ_s": "",
}


save_params = {}
print(" value  +-\terror \t input \t  name ")
for name in trans_params:
    v, e = trans_params[name], trans_errors[name]
    print("{: .5f}\t+-\t{:.5f}\t{: .5f}\t".format(v, e, ref_params[name][0]), name, comments.get(name))
    save_params[name] = [float(v), float(e)]

print(" value  +-\terror \t input \t  name ")
for name in trans_params:
    v, e = trans_params[name], trans_errors[name]
    print("${: .3f}\\pm{:.3f}$ & {: .4f} & ".format(v, e, ref_params[name][0]), latex_name[name], "&", latex_comments.get(name), r"\\\hline")


fit_result.extra = save_params
fit_result.save_as("final_params_loader.json")
with open("trans_params_loader.json", "w") as f:
    json.dump(save_params, f, indent=2)

exit()

config.plot_partial_wave(plot_pull=True, prefix="figure_loader/")
config.plot_partial_wave(plot_pull=True, prefix="figure_loader/tagp_", cut_function = lambda x: x["tag"] > 0)
config.plot_partial_wave(plot_pull=True, prefix="figure_loader/tagm_", cut_function = lambda x: x["tag"] < 0)

