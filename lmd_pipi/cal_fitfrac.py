import numpy as np
import csv
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.utils import tuple_table

def save_frac_csv(file_name, fit_frac):
    table = tuple_table(fit_frac)
    with open(file_name, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(table)

# load model, parameters and error matrix
config = ConfigLoader("config.yml")
config.set_params("gen_params.json")
config.inv_he = np.load("error_matrix.npy")

# calculate the fit fractions and interference with uncertainties from error propogation
value, err = config.cal_fitfractions(exclude_res=["Lambda"])

# save to csv file
save_frac_csv("fit_frac.csv", value)
save_frac_csv("fit_frac_err.csv", err)
