"""
config_loader.py

Utility for loading and validating YAML configuration files
for the evacuation environment and training pipelines.
"""

import os
from typing import Any, Dict

import yaml


_REQUIRED_SECTIONS = ["grid", "map", "dynamics", "rewards"]
_REQUIRED_MAP_KEYS = ["walls", "exits", "fire_sources", "agent_start"]


def load_config(path: str, validate: bool = True) -> Dict[str, Any]:
    """Load a YAML config file and optionally validate required keys.

    Args:
        path:     Absolute or relative path to the YAML config file.
        validate: If True, validates that environment-specific sections
                  (grid, map, dynamics, rewards) are present. Set to False
                  for non-environment configs like dqn.yaml or training.yaml.

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

    if validate:
        _validate_env_config(config, path)

    return config


def _validate_env_config(config: Dict[str, Any], path: str) -> None:
    """Validate that an environment config has all required sections.

    Args:
        config: Parsed config dictionary.
        path:   File path (for error messages).

    Raises:
        ValueError: If required sections or keys are missing.
    """
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

