from tf_pwa.config_loader import ConfigLoader

# load model and parameters
config = ConfigLoader("config.yml")
config.set_params("final_params.json")

# draw the projection of fit results
config.plot_partial_wave(smooth=False, single_legend=True)
