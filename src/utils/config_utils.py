import yaml
from tdpy.verbosity import print

def load_config(filepath):
    print(f"Reading from {filepath}...")
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_config(config):
    # Implement your configuration validation logic
    pass
