from tf_pwa.config_loader import ConfigLoader
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.amp import time_dep
import tensorflow as tf
import numpy as np
import math
import json
import uproot
from tf_pwa.data import data_index
from tf_pwa.root_io import save_dict_to_root

config = ConfigLoader("config_gen.yml")
config.set_params("final_params_amp.json")

f = config.get_particle_function("phi10")

ha = f.ha

def generate_phsp(N):
    phi = np.random.random(N) *np.pi * 2  - np.pi
    c1 = np.random.random(N) *2  - 1
    c2 = np.random.random(N) *2  - 1
    p4 = ha.build_data({}, [np.array([0.]), c1, c2], [np.array([0.]), np.array([0.]), phi])
    data = config.data.cal_angle(p4)
    data["time"] = np.random.random(N) * 15
    data["tag"] = np.random.choice([1,-1], N)
    return data

N_data = 1000000
toy = config.generate_toy(N_data, gen=generate_phsp)

idx1 = config.get_data_index("angle", "Jpsi/mup")
idx2 = config.get_data_index("angle", "phi/Kp")
idx3 = config.get_data_index("mass", "phi")
ang1 = data_index(toy, idx1)
ang2 = data_index(toy, idx2)
mphi = data_index(toy, idx3)
all_sigma = np.random.random((1000,)) * 0.03 + 0.03 # as example
rnd_idx = np.random.randint(all_sigma.shape[0], size=N_data)
sigma = all_sigma[rnd_idx]
delta = np.random.normal(size=N_data)
ct = toy["time"] + delta * sigma
cut = ((ct >0)&(ct < 15)).numpy()

data = {
 "EvWeight": 1 * np.ones(N_data, dtype=np.int32)[cut],
 "SwId": 0* np.ones(N_data, dtype=np.int32)[cut],
 "cos_thetaK": np.cos(ang2["beta"])[cut],
 "cos_thetaL": np.cos(ang1["beta"])[cut],
 "ct": ct.numpy()[cut],
 "evtNum": 0* np.ones(N_data, dtype=np.int32)[cut],
 "mB": 0* np.ones(N_data, dtype=np.int32)[cut],
 "m_phi": mphi.numpy()[cut]*1000,
 "phi": ang2["alpha"].numpy()[cut],
 "runNum": 0* np.ones(N_data, dtype=np.int32)[cut],
 "sigma_ct": sigma[cut],
 "sigma_mB": 0* np.ones(N_data, dtype=np.int32)[cut],
 "tag_decision":  toy["tag"].numpy()[cut],
 "tag_decision_ss":  toy["tag"].numpy()[cut],
 "tag_dilution"   : 1* np.ones(N_data, dtype=np.int32)[cut],
 "tag_dilution_ss" : 1* np.ones(N_data, dtype=np.int32)[cut],
 "tag_omega"       : 0* np.ones(N_data, dtype=np.int32)[cut],
 "tag_omega_ss"    : 0* np.ones(N_data, dtype=np.int32)[cut],
 "true_cos_thetaK" : np.cos(ang2["beta"])[cut],
 "true_cos_thetaL" : np.cos(ang1["beta"])[cut],
 "true_ct"         : toy["time"].numpy()[cut],
 "true_pdf"        : config.get_amplitude()(toy).numpy()[cut],
 "true_phi"        : ang2["alpha"].numpy()[cut],
 "true_tag_decision" : toy["tag"].numpy()[cut],
 "true_tag_dilution" : 1* np.ones(N_data, dtype=np.int32)[cut],
 "true_tag_omega"  : 0* np.ones(N_data, dtype=np.int32)[cut],
}
save_dict_to_root(data, "Toy.root", ["DecayTree"])
print(toy)
print(toy.keys())

print(data)
