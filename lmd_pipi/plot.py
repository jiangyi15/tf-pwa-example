from tf_pwa.config_loader import ConfigLoader


config = ConfigLoader("config.yml")
config.set_params("final_params.json")
config.plot_partial_wave(smooth=False, single_legend=True)

