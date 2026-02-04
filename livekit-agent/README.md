# CallTourAI LiveKit Agent (basic, Gemini)

This folder contains a minimal **LiveKit Agents** voice agent that can run:

- **Console mode** (no LiveKit server required) for quick local testing
- **Dev/Start mode** (connects to a LiveKit server) for real rooms/clients

The implementation is in `livekit_basic_agent.py`.

## 1) Python dependencies

This repo uses `requirements.txt` at the root. Install dependencies from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Environment variables

```bash
cp livekit-agent/.env.example livekit-agent/.env
# then edit livekit-agent/.env
```

Required:
- `GOOGLE_API_KEY`

Optional:
- `GEMINI_LIVE_MODEL` (default: `models/gemini-2.5-flash-native-audio-latest`)
- `GEMINI_VOICE` (default: `Puck`)

Optional (only if connecting to a LiveKit server):
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

### Persistent memory (PostgreSQL)

If you set `DATABASE_URL`, the agent will persist a **text-only** chat history
to PostgreSQL and restore it when it reconnects to the same room.

In `livekit-agent/.env`:

```dotenv
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/calltourai
MEMORY_MAX_ITEMS=200
MEMORY_SCOPE=room
```

Notes:
- `MEMORY_SCOPE=room` stores memory per room name (recommended).
- `MEMORY_SCOPE=room_user` scopes memory by `(room_name, user_identity)`. For
	this to work across reconnects, your client must rejoin with the **same**
	LiveKit token identity.

## 3) Download model files (first run)

LiveKit Agents uses model files for VAD / turn detection depending on plugins.
For this basic agent, run once:

```bash
python livekit-agent/livekit_basic_agent.py download-files
```

## 4) Run

### Console mode (fastest to test)

In WSL, audio device access often fails. If you want to test without audio devices, use text mode:

```bash
python livekit-agent/livekit_basic_agent.py console --text
```

If you *do* have Linux audio devices available, you can also list/select them:

```bash
python livekit-agent/livekit_basic_agent.py console --list-devices
```

```bash
python livekit-agent/livekit_basic_agent.py console
```

### WSL-friendly: use browser mic + LiveKit room

This is the recommended approach in WSL: the microphone is captured by your Windows browser and sent to LiveKit, while the agent runs in WSL.

1) Make sure `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` are set in `livekit-agent/.env`.

2) Start the web helper (token endpoint + simple UI):

```bash
python -m uvicorn livekit_web_server:app --app-dir livekit-agent --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your Windows browser.

3) Start the agent and connect it to the same room name you enter in the web UI:

```bash
python livekit-agent/livekit_basic_agent.py connect --room calltour
```

If you prefer a different identity:

```bash
python livekit-agent/livekit_basic_agent.py connect --room calltour --participant-identity agent
```

Talk in the browser — the agent should respond with audio.

Note: if you leave the room and rejoin, the agent should still respond. If you
observe the agent going silent after a disconnect/reconnect, ensure you're using
the updated agent code that disables `user_away_timeout`.

### Worker mode (advanced)

`start`/`dev` runs a worker that waits for server-side agent dispatch jobs. If you haven't configured agent dispatch, prefer `connect --room ...`.

```bash
python livekit-agent/livekit_basic_agent.py dev
# or
python livekit-agent/livekit_basic_agent.py start
```

## Built-in tools

- `get_current_date_and_time`
- `lookup_contacts` (reads `data/contacts.txt`)
- `rag_answer` (uses `rag.graph.get_app()`)
