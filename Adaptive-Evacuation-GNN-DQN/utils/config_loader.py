"""
config_loader.py

Utility for loading and validating YAML configuration files
for the evacuation environment.
"""

import os
from typing import Any, Dict

import yaml


_REQUIRED_SECTIONS = ["grid", "map", "dynamics", "rewards"]
_REQUIRED_MAP_KEYS = ["walls", "exits", "fire_sources", "agent_start"]


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and validate required keys.

    Args:
        path: Absolute or relative path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections or keys are missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    # Validate top-level sections
    for section in _REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(
                f"Missing required config section: '{section}' in {path}"
            )

    # Validate map keys
    map_cfg = config["map"]
    for key in _REQUIRED_MAP_KEYS:
        if key not in map_cfg:
            raise ValueError(
                f"Missing required map key: 'map.{key}' in {path}"
            )

    return config
