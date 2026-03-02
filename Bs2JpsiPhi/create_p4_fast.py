import uproot
from tf_pwa.config_loader import ConfigLoader
import numpy as np
from tf_pwa.angle import LorentzVector as lv
import sys

# idx = sys.argv[1]

config = ConfigLoader("config_data.yml")

f = config.get_particle_function("phi")
ha = f.ha

all_vars = ["true_cos_thetaK", "true_cos_thetaL", "true_phi", "true_ct", "true_tag_decision", "sigma_ct", "ct", "true_pdf"]
with uproot.open(f"./Toy.root") as f:
    t = f.get("DecayTree")
    data = t.arrays(all_vars)
data = {k: data[k] for k in all_vars}
# data = {k: v[data["trueTag"]<0] for k, v in data.items()}
data = {k: v for k, v in data.items()}

np.save("data_t.npy", data["true_ct"])
np.save("data_t_smear.npy", data["ct"])
np.save("data_t_sigma.npy", data["sigma_ct"])
np.save("data_true_pdf.npy", data["true_pdf"])
np.save("data_tag.npy", data["true_tag_decision"])
print(np.min(data["true_ct"]), np.max(data["true_ct"]))
np.save("data_angles.npy", np.stack([np.arccos(data["true_cos_thetaL"]), np.arccos(data["true_cos_thetaK"]), data["true_phi"]], axis=-1))
exit()

import matplotlib.pyplot as plt
for idx, i in enumerate(["mHH"]):
    plt.clf()
    # plt.hist(data[i], bins=50, range=(np.min(mi[idx])-0.1, np.max(mi[idx])+0.1))
    # plt.hist(mi[idx], bins=50, alpha=0.5, range=(np.min(mi[idx])-0.1, np.max(mi[idx])+0.1))
    plt.scatter(mi[idx], data[i], s=0.1)
    plt.savefig(i)
