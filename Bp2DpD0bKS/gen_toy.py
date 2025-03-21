from tf_pwa.config_loader import ConfigLoader
import json
from tf_pwa.config import set_config

set_config("polar", False)

pprint = lambda x: print(json.dumps(x, indent=2))

config = ConfigLoader("config.yml")
config.set_params("scaled_params.json")

pprint(config.get_params())

toy = config.generate_toy(10000)
phsp = config.generate_phsp(1000000)

config.plot_partial_wave(data=[toy], phsp=[phsp], plot_pull=True)


