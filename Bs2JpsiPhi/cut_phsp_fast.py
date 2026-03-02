import numpy as np
import sys

seed = 10
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
np.random.seed(seed+1000)

time = np.load("phsp_t_exp.npy")
sigma = np.load("data_t_sigma.npy")

idx = np.random.randint(sigma.shape[0], size=time.shape)

sigma_i = sigma[idx]

time_rec = time + np.random.normal(size=time.shape[0]) * sigma_i
np.save("phsp_t_smear.npy", time_rec)
np.save("phsp_t_sigma.npy", sigma_i)

cut = time_rec > 0

w = np.load("phsp_t_weight.npy")
new_w = w * cut
np.save("phsp_t_weight_cut.npy", new_w)
