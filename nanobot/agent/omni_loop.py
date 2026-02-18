"""
OmniLoop: The DeepAgent-powered brain for Nanobot.

This module replaces the standard AgentLoop with one backed by OmniCoreAgent's DeepAgent.
It bridges the gap between Nanobot's message bus and OmniCoreAgent's event router.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Awaitable

from loguru import logger

# Nanobot imports
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry as NanobotToolRegistry
from nanobot.session.manager import SessionManager

# OmniCoreAgent imports
try:
    from omnicoreagent.omni_agent.deep_agent.deep_agent import DeepAgent
    from omnicoreagent.core.memory_store.memory_router import MemoryRouter
    from omnicoreagent.core.events.event_router import EventRouter
    from omnicoreagent.core.events.base import Event, EventType
    from omnicoreagent.core.tools.local_tools_registry import ToolRegistry as OmniToolRegistry
except ImportError as e:
    # Fallback for when the environment isn't fully set up yet
    logger.error(f"OmniCoreAgent import failed: {e}")
    raise e


from omnicoreagent.core.tools.local_tools_registry import Tool as OmniTool

class NanobotToolAdapter:
    """Adapts Nanobot tools to OmniCoreAgent's ToolRegistry."""
    
    @staticmethod
    def adapt(nanobot_registry: NanobotToolRegistry) -> OmniToolRegistry:
        """Convert Nanobot tools to an OmniCoreAgent ToolRegistry."""
        omni_registry = OmniToolRegistry()
        
        for name, tool in nanobot_registry._tools.items():
            # Create a wrapper function that matches the signature expected by OmniCoreAgent
            # OmniCoreAgent expects a callable.
            
            async def wrapper(**kwargs):
                try:
                    result = await tool.execute(**kwargs)
                    if isinstance(result, (dict, list)):
                        return json.dumps(result, ensure_ascii=False)
                    return str(result)
                except Exception as e:
                    return f"Error executing {name}: {str(e)}"
            
            # Nanobot tools have .input_schema (dict).
            # We pass it directly to OmniTool.
            
            omni_tool = OmniTool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema if hasattr(tool, 'input_schema') else {},
                function=wrapper
            )
            
            # Register directly into the registry's internal dict
            omni_registry.tools[tool.name] = omni_tool
            
        return omni_registry


class OmniLoop:
    """
    The OmniLoop replaces the standard AgentLoop.
    It uses DeepAgent to process messages.
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: Any = None,
        cron_service: Any = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or "gpt-4o" # DeepAgent prefers strong models
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        
        # Initialize Nanobot components (for context building)
        self.context_builder = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        
        # Initialize OmniCoreAgent components
        self.memory_router = MemoryRouter(memory_store_type="in_memory")
        self.event_router = EventRouter(event_store_type="in_memory")
        
        # Configure Model
        # OmniCoreAgent expects a specific dict format for model config
        # We also need to inject the API key into the environment for LiteLLM
        import os
        
        # Extract API key from Nanobot provider
        api_key = None
        if hasattr(self.provider, "api_key"):
            api_key = self.provider.api_key
            
        # Set environment variable for LiteLLM (OmniCoreAgent relies on this)
        if api_key:
            # Set generic fallback used by some OmniCoreAgent paths
            os.environ["LLM_API_KEY"] = api_key
            
            # Detect provider type and set appropriate env var
            model_name = self.model.lower()
            if "gemini" in model_name:
                os.environ["GEMINI_API_KEY"] = api_key
            elif "claude" in model_name or "anthropic" in model_name:
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif "gpt" in model_name or "openai" in model_name:
                os.environ["OPENAI_API_KEY"] = api_key
            elif "deepseek" in model_name:
                os.environ["DEEPSEEK_API_KEY"] = api_key
            # Fallback for generic/custom providers
            else:
                os.environ["OPENAI_API_KEY"] = api_key
            
            # Fallback: Set LLM_API_KEY which OmniCoreAgent sometimes requires
            os.environ["LLM_API_KEY"] = api_key

        # Parse model string "provider/model" for OmniCoreAgent config
        if "/" in self.model:
            provider, model_name = self.model.split("/", 1)
        else:
            provider = "openai" # Default to openai if unspecified
            model_name = self.model

        self.model_config = {
            "provider": provider,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Build System Instruction from Nanobot's ContextBuilder
        # We only grab the static parts (Identity, Soul, User)
        # DeepAgent handles the memory/history injection itself.
        system_instruction = self.context_builder.build_system_prompt()
        
        # Initialize Nanobot Tools
        self.tools = NanobotToolRegistry()
        self._register_default_tools()
        
        # Adapt Nanobot Tools to OmniCoreAgent
        omni_local_tools = NanobotToolAdapter.adapt(self.tools)
        
        # Initialize DeepAgent
        self.agent = DeepAgent(
            name="OmniClaw",
            system_instruction=system_instruction,
            model_config=self.model_config,
            memory_router=self.memory_router,
            event_router=self.event_router,
            agent_config={"max_steps": 30}, # Give it room to think
            local_tools=omni_local_tools,
            debug=True,
        )
        
        self._running = False

    def _register_default_tools(self) -> None:
        """Register the default set of tools (Nanobot logic)."""
        from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
        from nanobot.agent.tools.shell import ExecTool
        from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
        from nanobot.agent.tools.message import MessageTool
        # We skip SpawnTool because DeepAgent handles subagents natively.
        from nanobot.agent.tools.cron import CronTool
        from nanobot.config.schema import ExecToolConfig

        exec_config = self.exec_config or ExecToolConfig()

        # File tools
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Cron tool
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def close_mcp(self) -> None:
        """Close MCP connections (stub)."""
        await self.agent.cleanup_mcp_servers()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        """
        # One-time init
        if not self.agent.is_initialized:
            await self.agent.initialize()
            
        session_id = f"{channel}_{chat_id}"
        
        try:
            if on_progress:
                # We can try to hook into event stream here if we want progress
                pass
                
            result = await self.agent.run(
                query=content,
                session_id=session_id
            )
            return result.get("response", "No response.")
        except Exception as e:
            logger.error(f"process_direct failed: {e}")
            return f"Error: {str(e)}"

    async def _process_message(self, msg: InboundMessage) -> None:
        """Process a message using DeepAgent."""
        logger.info(f"DeepAgent processing: {msg.content[:50]}...")
        
        # Generate a session ID that persists for this user/channel
        session_id = f"{msg.channel}_{msg.chat_id}"
        
        # Inject the user message into the session
        # DeepAgent.run() handles this, but we want to stream events
        
        # Start the run in a background task so we can consume the event stream
        # However, DeepAgent.run is atomic. We need to use stream_events separately 
        # or rely on the event router callbacks.
        
        # We'll attach a listener to the EventRouter for this session
        async def event_listener(event: Event):
            if event.session_id != session_id:
                return
                
            if event.type == EventType.AGENT_THOUGHT:
                # Log thoughts or send as "typing" status
                logger.info(f"🧠 {event.payload.message}")
            
            elif event.type == EventType.TOOL_CALL_STARTED:
                # Notify user of action
                content = f"🛠️ Using tool: {event.payload.tool_name}..."
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content
                ))
            
            # We don't send the final answer here because .run() returns it.

        # Run the agent
        try:
            # Note: We pass the query directly. DeepAgent manages history.
            result = await self.agent.run(
                query=msg.content,
                session_id=session_id
            )
            
            response_text = result.get("response", "No response generated.")
            
            # Send final response
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=response_text
            ))
            
        except Exception as e:
            logger.error(f"DeepAgent run failed: {e}")
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"⚠️ Agent Error: {str(e)}"
            ))
