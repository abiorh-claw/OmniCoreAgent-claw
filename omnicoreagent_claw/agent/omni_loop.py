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
from omnicoreagent_claw.bus.events import InboundMessage, OutboundMessage
from omnicoreagent_claw.bus.queue import MessageBus
from omnicoreagent_claw.providers.base import LLMProvider
from omnicoreagent_claw.agent.context import ContextBuilder
from omnicoreagent_claw.agent.tools.registry import ToolRegistry as ClawToolRegistry
from omnicoreagent_claw.session.manager import SessionManager

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

class ToolAdapter:
    """Adapts Nanobot tools to OmniCoreAgent's ToolRegistry."""
    
    @staticmethod
    def adapt(tool_registry: ClawToolRegistry) -> OmniToolRegistry:
        """Convert Nanobot tools to an OmniCoreAgent ToolRegistry."""
        omni_registry = OmniToolRegistry()
        
        for name, tool in tool_registry._tools.items():
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
        # File-backed memory: in-memory hot layer + JSONL file persistence
        from omnicoreagent_claw.agent.file_memory import FileBackedMemoryStore
        data_dir = Path.home() / ".omnicoreagent-claw" / "data" / "sessions"
        file_store = FileBackedMemoryStore(data_dir=data_dir)
        self.memory_router = MemoryRouter(memory_store_type="in_memory")
        self.memory_router.memory_store = file_store  # Override with file-backed store
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
            provider = None
            print(f"model name: {model_name}")
            if "gemini" in model_name:
                os.environ["GEMINI_API_KEY"] = api_key
                provider = "gemini"
            elif "claude" in model_name or "anthropic" in model_name:
                os.environ["ANTHROPIC_API_KEY"] = api_key
                provider = "anthropic"
            elif "gpt" in model_name or "openai" in model_name:
                os.environ["OPENAI_API_KEY"] = api_key
                provider = "openai"
            elif "deepseek" in model_name:
                os.environ["DEEPSEEK_API_KEY"] = api_key
                provider = "deepseek"
            # Fallback for generic/custom providers
            else:
                os.environ["OPENAI_API_KEY"] = api_key
                provider = "openai"
            
            # Fallback: Set LLM_API_KEY which OmniCoreAgent sometimes requires
            os.environ["LLM_API_KEY"] = api_key

       
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
        self.tools = ClawToolRegistry()
        self._register_default_tools()
        
        # Adapt Nanobot Tools to OmniCoreAgent
        omni_local_tools = ToolAdapter.adapt(self.tools)
        
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
        from omnicoreagent_claw.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
        from omnicoreagent_claw.agent.tools.shell import ExecTool
        from omnicoreagent_claw.agent.tools.web import WebSearchTool, WebFetchTool
        from omnicoreagent_claw.agent.tools.message import MessageTool
        # We skip SpawnTool because DeepAgent handles subagents natively.
        from omnicoreagent_claw.agent.tools.cron import CronTool
        from omnicoreagent_claw.config.schema import ExecToolConfig

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

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        # One-time init
        if not self.agent.is_initialized:
            await self.agent.initialize()
        logger.info("OmniLoop agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    await self._process_message(msg)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("OmniLoop agent loop stopping")

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

        # Handle slash commands (copied from standard loop.py)
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            await self._handle_new_command(session_id, msg.channel, msg.chat_id)
            return
        
        if cmd == "/help":
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="🐈 OmniClaw commands:\n/new — Start a new conversation (consolidate memory)\n/help — Show available commands"
            ))
            return
        
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

    async def _handle_new_command(self, session_id: str, channel: str, chat_id: str):
        """Handle /new command: Consolidate memory and clear session."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content="New session started. Memory consolidation in progress..."
        ))

        # 1. Fetch history from OmniCoreAgent memory
        try:
            history = await self.memory_router.get_messages(session_id, agent_name=self.agent.name)
        except Exception as e:
            logger.warning(f"Could not fetch history for consolidation: {e}")
            history = []

        # 2. Clear OmniCoreAgent memory
        await self.memory_router.clear_memory(session_id=session_id, agent_name=self.agent.name)

        # 3. Run consolidation logic (adapted from loop.py)
        # We run this in background so we don't block
        asyncio.create_task(self._consolidate_logic(history))

    async def _consolidate_logic(self, messages: list[dict]):
        """
        Consolidate messages into MEMORY.md using DeepAgent/LLMProvider.
        Adapted from AgentLoop._consolidate_memory.
        """
        if not messages:
            return

        from omnicoreagent_claw.agent.memory import MemoryStore
        import json_repair
        
        memory_store = MemoryStore(self.workspace)
        
        # Convert OmniCoreAgent messages to simple text format
        lines = []
        for m in messages:
            role = m.get("role", "unknown").upper()
            content = m.get("content", "")
            lines.append(f"[{m.get('timestamp', '?')}] {role}: {content}")
        
        conversation = "\n".join(lines)
        current_memory = memory_store.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM].

2. "memory_update": The updated long-term memory content. Add any new facts, preferences, or project context. If nothing new, return existing content.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON."""

        try:
            # We use self.provider directly to avoid side effects of agent state
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            
            text = (response.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            result = json_repair.loads(text)
            
            if entry := result.get("history_entry"):
                memory_store.append_history(entry)
            
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory_store.write_long_term(update)
            
            logger.info("Memory consolidation complete (OmniLoop).")
            
        except Exception as e:
            logger.error(f"Memory consolidation failed in OmniLoop: {e}")
