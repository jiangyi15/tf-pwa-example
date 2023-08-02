import csv
import numpy as np
import tensorflow as tf
from tf_pwa.config_loader import ConfigLoader

# load model, parameters and error matrix
config = ConfigLoader("config.yml")
config.set_params("gen_params.json")
config.inv_he = np.load("error_matrix.npy")

def read_complex(pt, name):
    r = pt[name+"r"]
    phi = pt[name+"i"]
    return tf.complex(r * tf.cos(phi), r * tf.sin(phi))

# alpha for rho(770)
with config.params_trans() as f1:
    g0 = read_complex(f1, "Lmdc->Lambda.rho(770)_g_ls_0")
    g1 = read_complex(f1, "Lmdc->Lambda.rho(770)_g_ls_1")
    g2 = read_complex(f1, "Lmdc->Lambda.rho(770)_g_ls_2")
    g3 = read_complex(f1, "Lmdc->Lambda.rho(770)_g_ls_3")
    # only constants can use numpy
    s3 = 1/np.sqrt(3) + 0j
    s6 = 1/np.sqrt(6) + 0j
    trans_matrix = np.array([[-s3,  s3, -s6,  s6],
                             [-s6, -s6, -s3, -s3],
                             [ s6, -s6, -s3,  s3],
                             [ s3,  s3, -s6, -s6]])
    # use tensorflow for variable related calculation
    hel = tf.linalg.matvec(trans_matrix, tf.stack([g0, g1, g2, g3]))
    num = tf.reduce_sum(tf.abs(hel)**2*np.array([-1., -1, 1, 1]))
    dom = tf.reduce_sum(tf.abs(hel)**2)
    alpha1 = num/dom
mean1 = float(alpha1)
err1 = float(f1.get_error(alpha1))
print("alpha Lambda rho = ",mean1, " +- ", err1)


# alpha for Sigma(1385)+
with config.params_trans() as f2:
    g0 = read_complex(f2, "Lmdc->piz.Sigma(1385)p_g_ls_0")
    g1 = read_complex(f2, "Lmdc->piz.Sigma(1385)p_g_ls_1")
    alpha2 = 2 * tf.math.real(g0*tf.math.conj(g1)) / (tf.abs(g0)**2+tf.abs(g1)**2)

mean2 = float(alpha2)
err2 = float(f2.get_error(alpha2))
print("alpha sigma(1385)+ pi0 = ",mean2, " +- ", err2)

# alpha for Sigma(1385)0
with config.params_trans() as f3:
    g0 = read_complex(f2, "Lmdc->pip.Sigma(1385)0_g_ls_0")
    g1 = read_complex(f2, "Lmdc->pip.Sigma(1385)0_g_ls_1")
    alpha3 = 2 * tf.math.real(g0*tf.math.conj(g1)) / (tf.abs(g0)**2+tf.abs(g1)**2)

mean3 = float(alpha3)
err3 = float(f3.get_error(alpha3))
print("alpha sigma(1385)0 pi+ = ",mean3, " +- ", err3)

# save as csv
name_table = [["alpha_Lambda_rho", "alpha_Sigma(1385)+_pi0", "alpha_Sigma(1385)0_pi+"]]
alpha_table = [[mean1,mean2,mean3]]
error_table = [[err1,err2,err3]]

with open("asymmetry.csv", "w") as f:
    f_csv = csv.writer(f)
    f_csv.writerows(name_table)
    f_csv.writerows(alpha_table)
    f_csv.writerows(error_table)
