"""
FileBackedMemoryStore: In-memory hot layer + JSONL file persistence.

Implements OmniCoreAgent's AbstractMemoryStore interface.
Uses InMemoryStore for all runtime operations (summarization, context windowing)
and flushes every message to disk as append-only JSONL files.

Storage layout:
    {data_dir}/
        {session_id}.jsonl   — one JSON object per line, append-only
"""

import json
import threading
from pathlib import Path
from typing import Any, Callable, List

from omnicoreagent.core.memory_store.in_memory import InMemoryStore
from omnicoreagent.core.utils import logger


class FileBackedMemoryStore(InMemoryStore):
    """
    File-backed memory store.

    Inherits InMemoryStore for hot-path operations (summarization, sliding window,
    token budget) and adds transparent JSONL persistence.

    - store_message(): delegates to InMemoryStore, then appends to JSONL file
    - get_messages(): loads from disk on first access, then delegates to InMemoryStore
    - clear_memory(): delegates to InMemoryStore, then removes JSONL file
    """

    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_sessions: set[str] = set()
        self._file_lock = threading.RLock()

    def _session_file(self, session_id: str) -> Path:
        """Get the JSONL file path for a session."""
        # Sanitize session_id for filesystem safety
        safe_name = session_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.data_dir / f"{safe_name}.jsonl"

    def _ensure_loaded(self, session_id: str) -> None:
        """Load session from disk if not already in memory."""
        if session_id in self._loaded_sessions:
            return

        file_path = self._session_file(session_id)
        if file_path.exists():
            try:
                messages = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                            messages.append(msg)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Skipping corrupt line {line_num} in {file_path}: {e}"
                            )

                if messages:
                    with self._lock:
                        self.sessions_history[session_id] = messages
                    logger.info(
                        f"Loaded {len(messages)} messages from disk for session '{session_id}'"
                    )
            except Exception as e:
                logger.error(f"Failed to load session '{session_id}' from disk: {e}")

        self._loaded_sessions.add(session_id)

    def _append_to_file(self, session_id: str, message: dict) -> None:
        """Append a single message to the session's JSONL file."""
        file_path = self._session_file(session_id)
        with self._file_lock:
            try:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
            except Exception as e:
                logger.error(f"Failed to append message to {file_path}: {e}")

    def _rewrite_file(self, session_id: str) -> None:
        """Rewrite the full JSONL file from in-memory state (used after clear/filter)."""
        file_path = self._session_file(session_id)
        with self._file_lock:
            try:
                with self._lock:
                    messages = self.sessions_history.get(session_id, [])
                    with open(file_path, "w", encoding="utf-8") as f:
                        for msg in messages:
                            f.write(
                                json.dumps(msg, ensure_ascii=False, default=str) + "\n"
                            )
            except Exception as e:
                logger.error(f"Failed to rewrite session file {file_path}: {e}")

    # ── AbstractMemoryStore interface ──

    async def store_message(
        self,
        role: str,
        content: str,
        metadata: dict,
        session_id: str,
    ) -> None:
        """Store message in memory and flush to disk."""
        # Ensure session is loaded first (in case this is a resumed session)
        self._ensure_loaded(session_id)

        # Delegate to InMemoryStore (which creates the message dict with id, timestamp, etc.)
        await super().store_message(role, content, metadata, session_id)

        # Grab the last appended message and flush to disk
        with self._lock:
            messages = self.sessions_history.get(session_id, [])
            if messages:
                last_msg = messages[-1]
                self._append_to_file(session_id, last_msg)

    async def get_messages(
        self, session_id: str = None, agent_name: str = None
    ) -> List[dict[str, Any]]:
        """Get messages, loading from disk if needed."""
        session_id = session_id or "default_session"
        self._ensure_loaded(session_id)
        return await super().get_messages(session_id, agent_name)

    async def clear_memory(
        self, session_id: str = None, agent_name: str = None
    ) -> None:
        """Clear memory and remove disk files."""
        if session_id:
            self._ensure_loaded(session_id)

        # Delegate to InMemoryStore
        await super().clear_memory(session_id, agent_name)

        # Handle disk cleanup
        if session_id and not agent_name:
            # Full session clear — remove the file
            file_path = self._session_file(session_id)
            with self._file_lock:
                try:
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"Removed session file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove session file {file_path}: {e}")
            self._loaded_sessions.discard(session_id)

        elif session_id and agent_name:
            # Partial clear (agent-specific) — rewrite the filtered file
            self._rewrite_file(session_id)

        elif agent_name and not session_id:
            # Agent clear across all sessions — rewrite all affected files
            for sid in list(self._loaded_sessions):
                self._rewrite_file(sid)

        else:
            # Full clear — remove all session files
            with self._file_lock:
                try:
                    for f in self.data_dir.glob("*.jsonl"):
                        f.unlink()
                        logger.info(f"Removed session file: {f}")
                except Exception as e:
                    logger.error(f"Failed to clear all session files: {e}")
            self._loaded_sessions.clear()

    async def mark_messages_summarized(
        self,
        message_ids: list[str],
        summary_id: str,
        retention_policy: str = "keep",
    ) -> None:
        """Mark messages as summarized and sync to disk."""
        await super().mark_messages_summarized(message_ids, summary_id, retention_policy)

        # Rewrite affected session files to reflect the status changes
        with self._lock:
            for session_id in self.sessions_history:
                if session_id in self._loaded_sessions:
                    self._rewrite_file(session_id)

    def list_sessions(self) -> list[str]:
        """List all session IDs with data on disk."""
        sessions = set()
        for f in self.data_dir.glob("*.jsonl"):
            sessions.add(f.stem)
        with self._lock:
            sessions.update(self.sessions_history.keys())
        return sorted(sessions)
