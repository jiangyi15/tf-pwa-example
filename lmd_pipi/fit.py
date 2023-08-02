from tf_pwa.config_loader import ConfigLoader
import numpy as np

config = ConfigLoader("config.yml")

fit_result = config.fit(batch=65000)

fit_error = config.get_params_error()
fit_result.set_error(fit_error)

np.save("error_matrix.npy", config.inv_he)
fit_result.save_as("final_params.json")

config.plot_partial_wave(fit_result, smooth=False, single_legend=True)
