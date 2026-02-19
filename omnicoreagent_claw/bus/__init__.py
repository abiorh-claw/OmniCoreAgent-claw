"""Message bus module for decoupled channel-agent communication."""

from omnicoreagent_claw.bus.events import InboundMessage, OutboundMessage
from omnicoreagent_claw.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
