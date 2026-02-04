"""CallTourAI LiveKit Basic Voice Agent (Gemini)

Minimal LiveKit Agents worker that you can run in `console` mode for quick
iteration, or connect to a LiveKit server (`dev`/`start`).

This version uses Google's Gemini Live (speech-to-speech) via
`livekit.plugins.google.realtime.RealtimeModel`.

Provider defaults:
- Realtime model: Gemini Live API
- VAD: Silero

Environment:
- Copy `livekit-agent/.env.example` -> `livekit-agent/.env`
- Set `GOOGLE_API_KEY` (Gemini Developer API key)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents import llm
from livekit.agents.llm import function_tool
from livekit.agents.voice import room_io
from livekit.plugins import google, silero

import os


# Load env from this folder by default
load_dotenv(Path(__file__).with_name(".env"))


@dataclass(frozen=True)
class AgentConfig:
    # For Gemini Live (bidiGenerateContent), use a *native audio* model.
    # These are the ones that actually work with the Live websocket API.
    gemini_live_model: str = os.getenv(
        "GEMINI_LIVE_MODEL", "models/gemini-2.5-flash-native-audio-latest"
    )
    gemini_voice: str = os.getenv("GEMINI_VOICE", "Puck")

    database_url: str = os.getenv("DATABASE_URL", "").strip()
    memory_max_items: int = int(os.getenv("MEMORY_MAX_ITEMS", "200"))
    # "room" (default) or "room_user"
    memory_scope: str = os.getenv("MEMORY_SCOPE", "room").strip().lower()
    memory_flush_debounce_s: float = float(os.getenv("MEMORY_FLUSH_DEBOUNCE_S", "1.0"))

    # How much prior conversation to inject into the agent's instructions on reconnect.
    memory_prompt_max_messages: int = int(os.getenv("MEMORY_PROMPT_MAX_MESSAGES", "20"))
    memory_prompt_max_chars: int = int(os.getenv("MEMORY_PROMPT_MAX_CHARS", "4000"))

    # If set, the agent will only (re)start sessions when this identity is present.
    # Useful when multiple participants join the room.
    target_participant_identity: str = os.getenv("TARGET_PARTICIPANT_IDENTITY", "").strip()


def _normalize_model_id(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return "models/gemini-2.5-flash-native-audio-latest"
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _read_contacts_text() -> str:
    contacts_path = Path(__file__).resolve().parents[1] / "data" / "contacts.txt"
    try:
        return contacts_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _safe_room_name(room: Any) -> str:
    name = getattr(room, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    sid = getattr(room, "sid", None)
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    return "unknown"


def _infer_single_remote_identity(room: Any) -> str:
    """Best-effort remote identity inference.

    If exactly one remote participant is present, use that identity.
    Otherwise, return an empty string.
    """

    remotes = getattr(room, "remote_participants", None)
    if not isinstance(remotes, dict):
        return ""

    identities: list[str] = []
    for p in remotes.values():
        ident = getattr(p, "identity", None)
        if isinstance(ident, str) and ident.strip():
            identities.append(ident.strip())

    identities = sorted(set(identities))
    if len(identities) == 1:
        return identities[0]
    return ""


def _pick_remote_identity(room: Any, *, target_identity: str = "") -> str:
    remotes = getattr(room, "remote_participants", None)
    if not isinstance(remotes, dict) or not remotes:
        return ""

    if target_identity:
        for p in remotes.values():
            ident = getattr(p, "identity", None)
            if ident == target_identity:
                return target_identity
        return ""

    identities: list[str] = []
    for p in remotes.values():
        ident = getattr(p, "identity", None)
        if isinstance(ident, str) and ident.strip():
            identities.append(ident.strip())
    identities = sorted(set(identities))
    return identities[0] if identities else ""


def _sanitize_chat_items(items: Iterable[llm.ChatItem]) -> list[dict[str, Any]]:
    """Convert LiveKit chat items into a JSON-safe, text-only representation."""

    out: list[dict[str, Any]] = []
    for it in items:
        if getattr(it, "type", None) != "message":
            continue
        if getattr(it, "role", None) not in ("user", "assistant"):
            continue

        content_parts: list[str] = []
        for c in (getattr(it, "content", None) or []):
            if isinstance(c, str) and c.strip():
                content_parts.append(c)
                continue

            # AudioContent / ImageContent (best-effort): keep transcript if present.
            transcript = getattr(c, "transcript", None)
            if isinstance(transcript, str) and transcript.strip():
                content_parts.append(transcript)

        if not content_parts:
            continue

        out.append(
            {
                "type": "message",
                "role": it.role,
                "content": content_parts,
                "created_at": float(getattr(it, "created_at", 0.0) or 0.0),
                "interrupted": bool(getattr(it, "interrupted", False)),
            }
        )

    return out


def _restore_chat_items(items: list[dict[str, Any]]) -> list[llm.ChatItem]:
    restored: list[llm.ChatItem] = []
    for d in items:
        if (d or {}).get("type") != "message":
            continue
        role = (d or {}).get("role")
        if role not in ("user", "assistant"):
            continue
        content = (d or {}).get("content")
        if isinstance(content, str):
            content = [content]
        if not isinstance(content, list) or not content:
            continue

        cleaned_content = [c for c in content if isinstance(c, str) and c.strip()]
        if not cleaned_content:
            continue

        kwargs: dict[str, Any] = {
            "role": role,
            "content": cleaned_content,
            "interrupted": bool((d or {}).get("interrupted") or False),
        }
        created_at = float((d or {}).get("created_at") or 0.0)
        if created_at > 0:
            kwargs["created_at"] = created_at

        restored.append(llm.ChatMessage(**kwargs))
    return restored


def _build_memory_prompt(
    *,
    saved_items: list[dict[str, Any]],
    max_messages: int,
    max_chars: int,
) -> str:
    """Build a compact, text-only memory block for the system/developer prompt."""

    if not saved_items:
        return ""

    max_messages = max(0, int(max_messages))
    max_chars = max(0, int(max_chars))
    items = saved_items[-max_messages:] if max_messages else []

    lines: list[str] = []
    for it in items:
        role = (it or {}).get("role")
        if role not in ("user", "assistant"):
            continue

        content = (it or {}).get("content")
        if isinstance(content, list):
            text = "\n".join([c for c in content if isinstance(c, str)])
        elif isinstance(content, str):
            text = content
        else:
            continue

        text = (text or "").strip()
        if not text:
            continue

        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {text}")

    block = "\n".join(lines).strip()
    if not block:
        return ""

    if max_chars and len(block) > max_chars:
        block = block[-max_chars:]
        # avoid starting mid-line
        nl = block.find("\n")
        if 0 <= nl < 200:
            block = block[nl + 1 :]

    return block.strip()


class CallTourAssistant(Agent):
    def __init__(
        self,
        *,
        memory_prompt: str = "",
        memory_store: Any | None = None,
        memory_key: Any | None = None,
    ):
        self._memory_store = memory_store
        self._memory_key = memory_key

        memory_prompt = (memory_prompt or "").strip()
        memory_block = (
            "\n\nHere is persisted conversation context from earlier in this room. "
            "Use it silently to stay consistent and remember user-provided details. "
            "Do NOT proactively repeat, quote, or summarize this context unless the user explicitly asks what you remember or asks for a recap.\n"
            f"{memory_prompt}"
            if memory_prompt
            else ""
        )
        super().__init__(
            instructions=(
                "You are a helpful, friendly voice assistant for CallTourAI. "
                "Keep responses concise and natural. "
                "If you need facts from the project knowledge base, call the `rag_answer` tool. "
                "If the provided context does not contain the answer, say you don't know. "
                "Do not proactively mention or reveal any persisted context. "
                "If the user asks whether you've spoken before, what you remember, or asks for a recap, call the `recall_persisted_memory` tool and answer based on it."
                + memory_block
            )
        )

    @function_tool
    async def recall_persisted_memory(self, context: RunContext, last_n: int = 8) -> str:
        """Recall persisted conversation memory for this room/user from Postgres.

        Use this when the user explicitly asks what you remember or whether you've spoken before.

        Args:
            last_n: Number of most recent messages to return (max 30).
        """

        if not self._memory_store or not self._memory_key:
            return "I don't have any persisted memory available for this room."

        try:
            items = await asyncio.to_thread(self._memory_store.load_items, self._memory_key)
        except Exception as e:
            return f"I couldn't read persisted memory right now ({type(e).__name__}: {e})."

        if not items:
            return "I don't have any earlier conversation saved for this identity in this room yet."

        n = int(last_n) if last_n is not None else 8
        n = max(1, min(n, 30))
        snippet = _build_memory_prompt(saved_items=items, max_messages=n, max_chars=2000)
        if not snippet:
            return "I have saved conversation history, but it doesn't contain any readable text messages."

        return "Yes — we have spoken before. Here are the most recent messages I have saved:\n" + snippet

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}."

    @function_tool
    async def lookup_contacts(self, context: RunContext, query: str) -> str:
        """Search contacts stored in `data/contacts.txt`.

        Args:
            query: Name or keyword to search for.
        """
        text = _read_contacts_text().strip()
        if not text:
            return "I don't have any contacts stored yet."

        q = query.strip().lower()
        matches = [line for line in text.splitlines() if q in line.lower()]
        if not matches:
            return f"No contacts matched '{query}'."
        return "Here are the matching contacts:\n" + "\n".join(matches)

    @function_tool
    async def rag_answer(self, context: RunContext, question: str) -> str:
        """Answer a question using the repo's LangGraph RAG pipeline."""
        try:
            from rag.graph import get_app

            app = get_app()
            result = app.invoke({"question": question})
            answer = (result or {}).get("answer", "")
            answer = (answer or "").strip()
            return answer or "I don't know."
        except Exception as e:
            return f"RAG is not available right now ({type(e).__name__}: {e})."


async def entrypoint(ctx: agents.JobContext):
    cfg = AgentConfig()

    # IMPORTANT: Connect to the room up-front.
    # We may wait for participant events before starting an AgentSession, but those
    # events won't fire unless the room is actually connected.
    try:
        await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to LiveKit room ({type(e).__name__}: {e})")

    room = ctx.room
    connected_id_queue: asyncio.Queue[str] = asyncio.Queue()
    active_identity: str | None = None
    active_left = asyncio.Event()
    seen_identities: set[str] = set()

    @room.on("participant_connected")
    def _on_participant_connected(participant: Any) -> None:
        ident = getattr(participant, "identity", None)
        if not isinstance(ident, str) or not ident.strip():
            return
        ident = ident.strip()
        if cfg.target_participant_identity and ident != cfg.target_participant_identity:
            return
        try:
            connected_id_queue.put_nowait(ident)
        except Exception:
            pass

    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: Any) -> None:
        nonlocal active_identity
        ident = getattr(participant, "identity", None)
        if not isinstance(ident, str) or not ident.strip():
            return
        if active_identity and ident.strip() == active_identity:
            active_left.set()

    while True:
        # Wait for a remote participant to (re)join.
        picked = _pick_remote_identity(room, target_identity=cfg.target_participant_identity)
        if not picked:
            picked = await connected_id_queue.get()

        active_identity = picked
        active_left.clear()

        session = AgentSession(
            llm=google.realtime.RealtimeModel(
                model=_normalize_model_id(cfg.gemini_live_model),
                voice=cfg.gemini_voice,
            ),
            vad=silero.VAD.load(),
            # Keep the worker alive while the user is away; we restart sessions ourselves.
            user_away_timeout=None,
        )

        # Optional persistent memory.
        memory_store = None
        memory_key = None
        memory_prompt = ""
        has_saved_memory = False
        if cfg.database_url:
            try:
                from postgres_memory import MemoryKey, PostgresMemoryStore

                room_name = _safe_room_name(room)
                user_identity = ""
                if cfg.memory_scope == "room_user":
                    user_identity = active_identity

                memory_key = MemoryKey(room_name=room_name, user_identity=user_identity)
                memory_store = PostgresMemoryStore(cfg.database_url)
                await asyncio.to_thread(memory_store.ensure_schema)

                saved_items = await asyncio.to_thread(memory_store.load_items, memory_key)
                if saved_items:
                    has_saved_memory = True
                    session.history.insert(_restore_chat_items(saved_items))

                memory_prompt = _build_memory_prompt(
                    saved_items=saved_items,
                    max_messages=cfg.memory_prompt_max_messages,
                    max_chars=cfg.memory_prompt_max_chars,
                )
                print(
                    f"[memory] enabled scope={cfg.memory_scope} room={memory_key.room_name} "
                    f"user={memory_key.user_identity or '-'} loaded_items={len(saved_items)}"
                )
            except Exception as e:
                print(
                    f"[memory] disabled ({type(e).__name__}: {e}). "
                    "Check livekit-agent/.env: DATABASE_URL should look like "
                    "postgresql+psycopg2://user:pass@127.0.0.1:5432/calltourai"
                )
                memory_store = None
                memory_key = None

        if memory_store and memory_key:
            dirty = False
            flush_task: asyncio.Task[None] | None = None
            flush_lock = asyncio.Lock()

            async def flush_memory() -> None:
                nonlocal dirty
                async with flush_lock:
                    if not dirty:
                        return
                    dirty = False

                    chat_ctx = session.history.copy(
                        exclude_function_call=True,
                        exclude_instructions=True,
                        exclude_empty_message=True,
                    )
                    sanitized = _sanitize_chat_items(chat_ctx.items)
                    if cfg.memory_max_items > 0:
                        sanitized = sanitized[-cfg.memory_max_items :]
                    await asyncio.to_thread(
                        memory_store.upsert_items,
                        key=memory_key,
                        items=sanitized,
                    )

            def schedule_flush() -> None:
                nonlocal dirty, flush_task
                dirty = True
                if flush_task and not flush_task.done():
                    return

                async def _debounced() -> None:
                    await asyncio.sleep(max(cfg.memory_flush_debounce_s, 0.0))
                    await flush_memory()

                flush_task = asyncio.create_task(_debounced())

            session.on("conversation_item_added", lambda _ev: schedule_flush())

        # Start a session linked to the room. We let RoomIO close it when the linked
        # participant disconnects, and our loop will create a fresh session on rejoin.
        await session.start(
            room=room,
            room_options=room_io.RoomOptions(close_on_disconnect=True),
            agent=CallTourAssistant(
                memory_prompt=memory_prompt,
                memory_store=memory_store,
                memory_key=memory_key,
            ),
        )

        is_returning = bool(active_identity and active_identity in seen_identities) or has_saved_memory
        if active_identity:
            seen_identities.add(active_identity)

        greet = "Say hi and ask how you can help."
        if is_returning:
            greet = "Say hi and a brief welcome-back (e.g., 'Welcome back'), then ask how you can help."
        await session.generate_reply(instructions=greet)

        # Wait for the active user to leave, then restart.
        await active_left.wait()
        print(f"[session] participant left ({active_identity}); restarting")
        try:
            await session.aclose()
        except Exception:
            pass
        active_identity = None


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
