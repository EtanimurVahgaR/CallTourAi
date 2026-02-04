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

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
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


class CallTourAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful, friendly voice assistant for CallTourAI. "
                "Keep responses concise and natural. "
                "If you need facts from the project knowledge base, call the `rag_answer` tool. "
                "If the provided context does not contain the answer, say you don't know."
            )
        )

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

    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model=_normalize_model_id(cfg.gemini_live_model),
            voice=cfg.gemini_voice,
        ),
        vad=silero.VAD.load(),
        # Keep the session alive when the user temporarily leaves the room.
        # The default auto-shutdown can make the agent go silent on reconnect.
        user_away_timeout=None,
    )

    # By default, the RoomIO can close the agent session when the linked
    # participant disconnects. If the user leaves and later rejoins, that can
    # make the agent go silent until the whole process restarts.
    await session.start(
        room=ctx.room,
        room_options=room_io.RoomOptions(close_on_disconnect=False),
        agent=CallTourAssistant(),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and ask how you can help."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
