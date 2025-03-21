from tf_pwa.config_loader import ConfigLoader
from scipy.optimize import minimize
from pprint import pprint
import tensorflow as tf

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
                    ret[f"{namei}x{namej}"] = float(value)/100
                    ret[f"{namej}x{namei}"] = float(value)/100
    return ret

ref_frac = load_csv()
print(ref_frac)

config = ConfigLoader("config.yml")
config.set_params("test_params.json")
pprint(config.get_params())
params =  config.get_params()
for i in params:
    if i.endswith("total_0i"):
        config.vm.set_fix(i)

phsp = config.generate_phsp(1000000)

amp = config.get_amplitude()
pw = []
for i, _ in enumerate(config.get_decay()):
    with amp.temp_used_res([i]):
        with amp.temp_total_gls_one():
            pw.append( tf.reshape(amp.decay_group.get_amp3(phsp),(-1,) ))

pw = tf.stack(pw, axis=-1)
pw_matrix = tf.reduce_sum( pw[:,:,None] * tf.math.conj(pw[:,None,:]), axis=0)


@tf.function
def cal_fitfractions():
    g_ls = []
    for i in config.get_decay():
        g_ls.append(i.get_amp_total())
    g_ls = tf.reshape(tf.stack(g_ls, axis=-1), (-1,))
    gls_matrix = g_ls[:,None] * tf.math.conj(g_ls[None,:])

    frac_matrix = gls_matrix * pw_matrix
    sw = tf.abs(tf.reduce_sum(frac_matrix))

    fit_frac = {}
    for i, di in enumerate(config.get_decay()):
        namei = str(list(di)[1].core)
        for j,dj in enumerate(config.get_decay()):
            namej = str(list(dj)[1].core)
            if i==j:
                fit_frac[namei] = tf.abs(frac_matrix[i,j])/sw
            if i> j:
                fit_frac[f"{namei}x{namej}"] = tf.math.real(frac_matrix[i,j])/sw *2
    return fit_frac




def nll():
    fit_frac = cal_fitfractions()
    ret = 0.0
    for k, v in fit_frac.items():
        if "x" not in k:
            ret += (ref_frac[k] - v)**2 * 1000
    return ret

def nll2():
    fit_frac = cal_fitfractions()
    ret = 0.0
    for k, v in fit_frac.items():
        if "x" not in k:
            ret += (ref_frac[k] - v)**2 * 1000
        else:
            ret += (ref_frac[k] - v)**2 * 100
    # print("chi2=", ret)
    return ret

# ret = config.vm.minimize(nll, jac=True)
# print(config.get_params())
best_fit = None
for i in range(10):
    config.reinit_params()
    ret = config.vm.minimize(nll, jac=True)
    if best_fit is None or best_fit.fun > ret.fun:
        best_fit = ret
    print(ret)
print("best", best_fit)
config.set_params(best_fit.x)
config.save_params("final_params.json")

print(cal_fitfractions())
