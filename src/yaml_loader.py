import yaml

def load_yaml(config_path=None):
    """Load a YAML file and return the contents as a dictionary."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
