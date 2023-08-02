from tf_pwa.config_loader import ConfigLoader
import numpy as np

# load configuration
config = ConfigLoader("config.yml")

# fit with the configuration
fit_result = config.fit(batch=65000)

# calculate the uncertainties
fit_error = config.get_params_error()
fit_result.set_error(fit_error)

# save fit parameters and error matrix
np.save("error_matrix.npy", config.inv_he)
fit_result.save_as("final_params.json")

# draw the projection of fit results
config.plot_partial_wave(fit_result, smooth=False, single_legend=True)
