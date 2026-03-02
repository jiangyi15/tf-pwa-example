import numpy as np


p4 = np.load("phsp_t_exp.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save("phsp_t_exp_double.npy", p4_new.reshape((-1,)))

p4 = np.load("phsp_t_weight.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save("phsp_t_weight_double.npy", p4_new.reshape((-1,)))

p4 = np.load("phsp_t_weight_cut.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save("phsp_t_weight_double_cut.npy", p4_new.reshape((-1,)))


p4 = np.load("phsp_t_smear.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save("phsp_t_smear_double.npy", p4_new.reshape((-1,)))

p4 = np.load("phsp_t_sigma.npy").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.save("phsp_t_sigma_double.npy", p4_new.reshape((-1,)))



p4 = np.load("phsp_tag1.npy").reshape((-1,1))
p4_new = p4 * np.array([1,-1]).reshape((2,))
np.save("phsp_tag1_double.npy", p4_new.reshape((-1,)))

p4 = np.load("phsp_angles.npy").reshape((-1,1,3))
p4_new = p4 * np.array([1,1]).reshape((2,1))
np.save("phsp_angles_double.npy", p4_new.reshape((-1,3)))



exit()

p4 = np.loadtxt("phsp_t_weight_cut2.dat").reshape((-1,1))
p4_new = p4 * np.array([1,1]).reshape((2,))
np.savetxt("phsp_t_weight_double_cut2.dat", p4_new.reshape((-1,)))




