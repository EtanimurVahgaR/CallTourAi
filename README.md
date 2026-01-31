# CallTourAI — Basic LangGraph RAG

This repo contains a minimal RAG (Retrieve → Generate) implemented as a LangGraph state machine.

You can mix-and-match:

- **Vector store**: local Chroma (no server), self-hosted Chroma server, or Chroma Cloud
- **LLM + embeddings**: local Ollama, or cloud Gemini (Google AI Studio)

The app reads config from `.env` (auto-loaded via `python-dotenv`).

## Quickstart (local Ollama + local Chroma, no server)

1) Activate your env

```bash
conda activate calltourai
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Install + run Ollama, then pull models

- Install Ollama: https://ollama.com/
- Start the server (if it isn’t already running):

```bash
ollama serve
```

- Pull models used by default:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

4) Add data to `./data/`

```bash
echo "My mother's contact name is Alice." > data/contacts.txt
```

5) Ingest + query

```bash
python -m rag.ingest
python -m rag.run_rag "What is the user's mother's contact name?"
```

## Configuration

Copy `.env.example` to `.env` and edit what you need. You do **not** need to export variables manually.

```bash
cp .env.example .env
```

### LLM + embeddings providers

The project supports separate providers for embeddings and the LLM:

- `RAG_LLM_PROVIDER`: `ollama` (local) or `gemini` (cloud)
- `RAG_EMBED_PROVIDER`: `ollama` (local) or `gemini` (cloud)

#### Option A: Local LLM (Ollama)

Required: Ollama installed + running.

`.env`:

```dotenv
RAG_LLM_PROVIDER=ollama
RAG_EMBED_PROVIDER=ollama

OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
# Optional if Ollama isn't on localhost:11434
# OLLAMA_BASE_URL=http://localhost:11434
```

#### Option B: Cloud LLM (Gemini / Google AI Studio)

Required: a Gemini API key.

`.env`:

```dotenv
RAG_LLM_PROVIDER=gemini
RAG_EMBED_PROVIDER=gemini

GOOGLE_API_KEY=YOUR_KEY_HERE
GEMINI_LLM_MODEL=gemini-flash-latest
GEMINI_EMBED_MODEL=text-embedding-004
```

If you’re unsure which model name your key supports, list them:

```bash
python -c "import os; from google import genai; c=genai.Client(api_key=os.environ['GOOGLE_API_KEY']); print([m.name for m in c.models.list()])"
```

Notes:

- When using a cloud provider, your question and retrieved context chunks are sent to that provider.
- If you change the embedding provider/model, re-run ingestion so vectors are consistent.

### Vector store (Chroma) setups

#### 1) Local Chroma (no server)

Stores vectors on disk in this repo.

`.env`:

```dotenv
CHROMA_MODE=local
CHROMA_DIR=./chroma_db
CHROMA_COLLECTION=calltourai
```

Run:

```bash
python -m rag.ingest
python -m rag.run_rag "..."
```

#### 2) Local/self-hosted Chroma server (HTTP)

This is useful when multiple apps/users share the same vector DB. The data is still stored wherever the server runs.

Start a server (one example):

```bash
chroma run --path ./chroma_db --host 0.0.0.0 --port 8000
```

Then point the app at it:

```dotenv
CHROMA_MODE=http
CHROMA_HTTP_HOST=localhost
CHROMA_HTTP_PORT=8000
CHROMA_HTTP_SSL=false
CHROMA_COLLECTION=calltourai
```

Run:

```bash
python -m rag.ingest
python -m rag.run_rag "..."
```

#### 3) Chroma Cloud

This stores vectors in Chroma Cloud.

Use the env vars your Chroma Cloud docs show:

```dotenv
CHROMA_MODE=cloud
CHROMA_HOST=api.trychroma.com
CHROMA_API_KEY=YOUR_CHROMA_KEY
CHROMA_TENANT=YOUR_TENANT_ID
CHROMA_DATABASE=YOUR_DATABASE_NAME
CHROMA_COLLECTION=calltourai
```

Run:

```bash
python -m rag.ingest
python -m rag.run_rag "..."
```

## Common issues

### "ValueError: Could not connect to a Chroma server"

- You’re in `CHROMA_MODE=http` but the server isn’t running or `CHROMA_HTTP_HOST/PORT/SSL` is wrong.
- For local (no server), use `CHROMA_MODE=local`.

### Gemini model NOT_FOUND

- Your `GEMINI_LLM_MODEL` doesn’t match what your key supports.
- Run the model list command above and set `GEMINI_LLM_MODEL` to one of the returned names (without the `models/` prefix).

## Level 1 upgrade: Real-time voice -> RAG

This project includes a “Level 1” streaming voice pipeline:

Mic (32ms frames) → Silero VAD (speech gate) → Faster-Whisper (transcribe) → LangGraph RAG.

### Install voice dependencies

```bash
pip install -r requirements.txt
```

Linux note: `sounddevice` may require PortAudio system libs. On Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y portaudio19-dev
```

### Run

```bash
python voice_rag.py
```

WSL note: live microphone capture often does not work in WSL because the Linux VM can’t see your Windows microphone devices (PortAudio reports no input devices). If you hit this, use either “File mode” or the “Browser mic” option below.

Useful flags (live mic):

- `--list-devices` (print available audio devices)
- `--mic 3` (select input device index)
- `--mic "USB"` (select by name substring)
- `--vad-threshold 0.5` (increase to ~0.65 in noisy rooms)
- `--end-silence-ms 500` (end-of-sentence trigger)
- `--model large-v3-turbo`

Example:

```bash
python voice_rag.py --vad-threshold 0.55 --end-silence-ms 500 --model large-v3-turbo
```

Notes:

- The script keeps a ~100ms pre-roll so the first syllable isn’t cut off.
- If you have CUDA, it will auto-select `device=cuda` and `compute=float16`.

### File mode (works anywhere)

If you can’t access a microphone (common on WSL), record an audio clip and run:

```bash
python voice_rag.py --audio-file path/to/clip.wav
```

### WSL-friendly alternative: Browser mic -> backend -> RAG

If WSL can’t see your microphone devices, you can still do “live” voice by capturing audio in your Windows browser and sending it to a WSL backend.

Start the web app in WSL:

```bash
python web_voice_rag.py
```

Then open (on Windows) in your browser:

- `http://localhost:5000`

Record → Stop → Send to RAG.

This option uses a small Flask server in WSL plus a static HTML page. The browser records audio, converts it to a 16kHz mono WAV, uploads it to the backend, and the backend runs Faster-Whisper + RAG.

## Project layout

- `rag/graph.py`: LangGraph wiring
- `rag/nodes.py`: retrieve + generate nodes
- `rag/ingest.py`: ingestion pipeline
- `rag/settings.py`: env-backed config
- `rag/vectorstore.py`: Chroma local/http/cloud connector
- `voice_rag.py`: live mic (when available) + `--audio-file` mode
- `web_voice_rag.py`: Flask server for browser mic uploads (WSL-friendly)
- `web/templates/index.html`: simple browser UI
