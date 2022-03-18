from tf_pwa.config_loader import ConfigLoader 
from tf_pwa import set_random_seed
import math
import json


set_random_seed(100)

def load_fitfrac(file_name="fit_frac.txt"):
    ret = {}
    with open(file_name) as f:
        for i in f.readlines():
            value = i.split(" ")
            ret[value[0]] = float(value[1])
    return ret

config = ConfigLoader("config.yml")
phsp = config.generate_phsp(50000)
params = config.get_params()


fit_frac, _ = config.cal_fitfractions(params, phsp)


fit_frac_diag = {}
for k, v in fit_frac.items():
    if isinstance(k, str) and k !="Jpsi":
        fit_frac_diag[k] = v

fit_frac_exp = load_fitfrac()

new_params = {}
for k, v in params.items():
    if k.endswith("total_0r"):
        for k2 in fit_frac_diag:
            if k2 in k:
                new_params[k] = v * math.sqrt(fit_frac_exp[k2]/fit_frac_diag[k2])
                break
    else:
        new_params[k] = v

with open("test_params.json", "w") as f:
    json.dump(new_params, f, indent=2)

config.set_params("test_params.json")
toy = config.generate_toy(5000)
config.plot_partial_wave(data=[toy], phsp=[phsp], plot_pull=True)

