from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_merge, data_index, data_to_numpy
import numpy as np
import os
import tensorflow as tf

N_sig = [1351, 233, 1040, 1200, 1047, 3120,  906]
bg_frac = [0.139, 0.123,0.143, 0.145, 0.160, 0.171, 0.182]
emc = [4600, 4612, 4626, 4640, 4660, 4680, 4700]

config = ConfigLoader("config.yml")
config.set_params("gen_params.json")

particles = ["(p, pim)", "pip", "piz", "p", "pim"]


def random_charge(p4):
    charge = np.random.random((p4[0].shape[0],)) > 0.5
    p4_new = []
    for i in p4:
        i_trans = i * np.array([1,-1,-1,-1])
        p4_new.append(np.where(charge[:,None], i, i_trans))
    charge = (charge > 0.5) * 2 -1
    return p4_new, charge



def savedat(file_name, dat):
    p = data_to_numpy([data_index(dat, ("particle", i, "p")) for i in particles])
    p, charge = random_charge(p)
    data1 = np.stack([p[0], p[1], p[2]], axis=-2)
    np.savetxt(file_name + "_lmd_pi_pi0.dat", data1.reshape((-1,4)))
    data2 = np.stack([p[3], p[4]], axis=-2)
    np.savetxt(file_name + "_p_pi.dat", data2.reshape((-1,4)))
    np.savetxt(file_name + "_charge.dat", charge)

def generate_toy(a, b, c):
    sig = config.generate_toy(a)
    # use flat background
    bg_data = config.generate_phsp(int(a / (1-b) *b))
    bg = config.generate_phsp(int(a / (1-b) *b))
    phsp = config.generate_phsp(100000)
    data = data_merge(sig, bg_data)

    savedat("data/data" + str(c), data)
    savedat("data/bkg" + str(c), bg)
    savedat("data/phsp" + str(c), phsp)


os.makedirs("data", exist_ok=True)
for a, b, c in zip(N_sig, bg_frac, emc):
    generate_toy(a, b, c)

with tf.device("CPU"):
    phsp = config.generate_phsp(500000)
    savedat("data/genmc_noeff", phsp)
