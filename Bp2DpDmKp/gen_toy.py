from tf_pwa.config_loader import ConfigLoader

config = ConfigLoader("config.yml")
config.set_params("scaled_params.json")
toy = config.generate_toy(10000)
phsp = config.generate_phsp(1000000)
config.plot_partial_wave(data=[toy], phsp=[phsp], plot_pull=True)



