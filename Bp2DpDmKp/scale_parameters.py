from tf_pwa.config_loader import ConfigLoader
from scipy.optimize import minimize
from pprint import pprint
import tensorflow as tf
import math
from tf_pwa import set_random_seed

def load_csv(file_name="fit_frac_ref.csv"):
    table = []
    with open(file_name) as f:
        ret = []
        for i in f.readlines():
            if not i.strip():
                continue
            table.append(i.strip().split(","))
    head = table[0][1:]
    ret = {}
    for i, namei in enumerate(head):
        for j, namej in enumerate(head):
            value = table[i+1][j+1]
            if value:
                if i==j:
                    ret[namei] = float(value)/100
                else:
                    ret[(namei, namej)] = float(value)/100
                    ret[(namej, namei)] = float(value)/100
    return ret

ref_frac = load_csv()
print(ref_frac)

config = ConfigLoader("config.yml")
config.set_params("phase_params.json")

old_params = config.get_params()
new_params = {}
set_random_seed(100)
mcdata = config.generate_phsp(1000000)
fit_frac, _ = config.cal_fitfractions(mcdata=mcdata, method="new")

for i in config.get_decay():
    name = str(i[1].core)
    total = str(i.total) + "_0r"
    scale = math.sqrt(ref_frac[name]/ fit_frac[name])
    new_params[total] = old_params[total] * scale

config.set_params(new_params)
config.save_params("scaled_params.json")

fit_frac_new, _ = config.cal_fitfractions(mcdata=mcdata, method="new")

for k, v in fit_frac_new.items():
    if k in ref_frac:
        if abs(ref_frac[k]) > 0.001:
            print(k, fit_frac_new[k], ref_frac[k])
