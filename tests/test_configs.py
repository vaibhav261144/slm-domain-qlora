"""
Unit tests for experiment configuration YAML files.
"""

import glob
import yaml
from pathlib import Path


def test_configs_are_valid_yaml():
    config_files = glob.glob("configs/*.yaml")
    assert len(config_files) > 0, "No config files found in configs/"

    for cfg_path in config_files:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{cfg_path} does not parse into a dictionary"
            assert "experiment" in data, f"{cfg_path} missing 'experiment' section"
            assert "model" in data, f"{cfg_path} missing 'model' section"
            assert "lora" in data, f"{cfg_path} missing 'lora' section"
            assert "training" in data, f"{cfg_path} missing 'training' section"
