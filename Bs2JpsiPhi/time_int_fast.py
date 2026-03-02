import numpy as np
from scipy.stats import norm

t = np.load("data_t_smear.npy")
sigma = np.load("data_t_sigma.npy")
mean = 0.0
N = 25

t_range = [0.0, 15]
print(t)
left = norm.cdf(t - t_range[0], -mean, sigma)
right = norm.cdf(t - t_range[1], -mean, sigma)
print(left, right)
delta = np.linspace(0, 1, N+1)
delta_prob = (delta[1:] + delta[:-1])/2

prob = (right[:,None] - left[:,None]) * delta_prob + left[:, None]
print(prob, delta_prob)
delta_x = norm.ppf(prob, -mean, sigma[:,None])
new_t = - delta_x + t[:,None]
print(new_t, np.min(new_t))

np.save("data_t_smear_int.npy", new_t.reshape((-1,)))
# p4 = np.loadtxt("data.txt").reshape((-1,1,4,4)) * np.ones((N,1,1))
# np.save("data_int.npy", p4.reshape((-1,4)))
p4 = np.load("data_tag.npy").reshape((-1,1)) * np.ones((N,))
np.save("data_tag_int.npy", p4.reshape((-1,)))
p4 = np.load("data_angles.npy").reshape((-1,1,3)) * np.ones((N,1))
np.save("data_angles_int.npy", p4.reshape((-1,3)))

#p4 = np.loadtxt("data_prob.dat").reshape((-1,1)) * np.ones((N,))
# np.savetxt("data_prob_int.dat", p4.reshape((-1,)))
# p4 = np.loadtxt("data_tag_eta.dat").reshape((-1,1)) * np.ones((N,))
# np.savetxt("data_tag_eta_int.dat", p4.reshape((-1,)))


