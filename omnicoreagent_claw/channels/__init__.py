"""Chat channels module with plugin architecture."""

from omnicoreagent_claw.channels.base import BaseChannel
from omnicoreagent_claw.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
