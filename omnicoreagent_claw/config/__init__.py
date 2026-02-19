"""Configuration module for omnicoreagent-claw."""

from omnicoreagent_claw.config.loader import load_config, get_config_path
from omnicoreagent_claw.config.schema import Config

__all__ = ["Config", "load_config", "get_config_path"]
