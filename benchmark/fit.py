from tf_pwa.config_loader import ConfigLoader 
import tensorflow as tf

config = ConfigLoader("config.yml")
config.set_params("gen_params.json")

with tf.device("GPU"):
    fit_result = config.fit() # jac="2-points")
fit_result.set_error(config.get_params_error())
fit_result.save_as("final_params.json")

config.plot_partial_wave(smooth=False)
