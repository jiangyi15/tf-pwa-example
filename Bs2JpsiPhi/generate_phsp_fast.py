from tf_pwa.config_loader import ConfigLoader
import numpy as np
from tf_pwa import set_random_seed
import sys


seed = 10
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
set_random_seed(seed+1000)


config = ConfigLoader("config_data.yml")

N = 10000000
config = ConfigLoader("config_data.yml")
f = config.get_particle_function("phi")
ha = f.ha
phi = np.random.random(N) *np.pi * 2  - np.pi
c1 = np.random.random(N) *2  - 1
c2 = np.random.random(N) *2  - 1
m_min, m_max = 0.988, 1.200
m = np.random.random(N) *(m_max - m_min)  +m_min

np.save("phsp_angles.npy", np.stack([np.arccos(c1), np.arccos(c2), phi], axis=-1))

t_min, t_max = 0.0, 15
t = np.random.random(N) *(t_max - t_min) + t_min
np.save("phsp_t.npy", t)

gamma = 1.0
t = -np.log(np.random.random(N) * (np.exp(-t_min*gamma) - np.exp(-t_max*gamma)) + np.exp(-t_max*gamma))/gamma
np.save("phsp_t_exp.npy", t)
np.save("phsp_t_weight.npy", np.exp( gamma * t))


t = (np.random.random(N) > 0.5 ).astype(np.int32) * 2-1
np.save("phsp_tag_true.npy", t)
t = (np.random.random(N) > 0.5 ).astype(np.int32) * 2-1
np.save("phsp_tag1.npy", t)
t = (np.random.random(N) > 0.5 ).astype(np.int32) * 2-1
np.save("phsp_tag2.dat", t)
t = np.random.random(N) *0.5
np.save("phsp_eta1.npy", t)
t = np.random.random(N) *0.5
np.save("phsp_eta2.npy", t)
