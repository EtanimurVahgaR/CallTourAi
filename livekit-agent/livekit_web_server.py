"""CallTourAI LiveKit Web Helper (FastAPI)

Purpose
- Serve a tiny web UI that captures microphone audio in the *browser* and joins a
  LiveKit room.
- Generate LiveKit access tokens from your server-side API key/secret.

Why this exists
- In WSL, LiveKit Agents `console` voice mode often can't access a microphone.
- Using a browser on Windows for audio input avoids WSL audio device issues.

Run
- python -m uvicorn livekit_web_server:app --app-dir livekit-agent --reload
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from livekit.api import AccessToken, VideoGrants


_ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(_ENV_PATH)

_BASE_DIR = Path(__file__).parent
_templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required env var: {name}. "
            f"Set it in {str(_ENV_PATH)} (or export it in your shell)."
        )
    return value


def _build_token(*, room: str, identity: str, name: Optional[str] = None) -> str:
    api_key = _required_env("LIVEKIT_API_KEY")
    api_secret = _required_env("LIVEKIT_API_SECRET")

    grants = VideoGrants(room_join=True, room=room)

    at = AccessToken(api_key, api_secret).with_identity(identity).with_grants(grants)
    if name:
        at = at.with_name(name)

    return at.to_jwt()


app = FastAPI(title="CallTourAI LiveKit Web Helper")


@app.get("/")
async def index(request: Request):
    return _templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_room": os.getenv("LIVEKIT_ROOM", "calltour"),
            "default_identity": os.getenv("LIVEKIT_IDENTITY", "web-user"),
        },
    )


@app.get("/token")
async def token(
    room: str = Query("calltour", min_length=1, max_length=128),
    identity: str = Query("web-user", min_length=1, max_length=128),
    name: Optional[str] = Query(None, max_length=128),
):
    livekit_url = _required_env("LIVEKIT_URL")
    jwt = _build_token(room=room, identity=identity, name=name)
    return JSONResponse({"url": livekit_url, "token": jwt, "room": room, "identity": identity})


@app.get("/healthz")
async def healthz():
    return {"ok": True}
