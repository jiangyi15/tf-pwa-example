from tf_pwa.config_loader import ConfigLoader
from tf_pwa.config import set_config
import math
import tensorflow as tf
from tf_pwa import set_random_seed
from tf_pwa.data import batch_call_numpy

set_random_seed(1000)
set_config("polar", False)

config = ConfigLoader("config.yml")
config.set_params("paper_params.json")



phsp = config.generate_phsp(1000000)

ref_fit_frac = {
    "Ds1(2700)": 0.271,
    "Ds2(2573)": 0.0159,
    "Ds1(2860)": 0.060,
    "NR_S": 0.449,
    "NR_P": 0.271,
    "X0(2900)": 0.0259,
    "sum_diag": 1.093
}

old_params = config.get_params()
new_params = {}
amp = config.get_amplitude()

def cal_fitfractions():
    total_amp = tf.reduce_sum(amp(phsp))
    sum_frac = 0.0
    ret = {}
    for idx, chain in enumerate(config.get_decay()):
        with amp.temp_used_res([idx]):
            pw_amp = tf.reduce_sum(amp(phsp))
        fit_frac = float(pw_amp / total_amp)
        ret[str(chain[1].core)] = fit_frac
        sum_frac += fit_frac
    ret["sum_diag"] = sum_frac
    return ret


fit_frac = cal_fitfractions()
for chain in config.get_decay():
    res = str(chain[1].core)
    scale =  math.sqrt(ref_fit_frac[res]/fit_frac[res])
    prefix = str(chain.total) + "_0"
    new_params[prefix + "r"] = scale * old_params[prefix + "r"]
    new_params[prefix + "i"] = scale * old_params[prefix + "i"]

config.set_params(new_params)
config.save_params("scaled_params.json")

for k, v in cal_fitfractions().items():
    print(k, v, ref_fit_frac[k])
