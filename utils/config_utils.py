import importlib


def load_config(config_path: str):
	'''Load config file as module.'''
	config_path = config_path.replace("/", ".").replace("\\", ".")
	if config_path.endswith(".py"):
		config_path = config_path[:-3]
	print(f"Loading config file from {config_path}.")
	return importlib.import_module(config_path)