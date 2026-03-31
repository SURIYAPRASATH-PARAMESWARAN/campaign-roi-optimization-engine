from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"

def load_config(path=None):
    config_path = Path(path) if path else _CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg = load_config()
