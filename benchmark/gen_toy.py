from tf_pwa.config_loader import ConfigLoader

config = ConfigLoader("config.yml")
config.set_params("gen_params.json")

toy = config.generate_toy2(10000)
phsp = config.generate_phsp(200000)

toy.savetxt("data.dat", config.get_dat_order())
phsp.savetxt("phsp.dat", config.get_dat_order())
