#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Runs the FastAPI helper that serves the web mic UI + token endpoint.
# Works well for WSL because the microphone is accessed in the Windows browser.

exec conda run -n calltourai --no-capture-output \
  python -m uvicorn livekit_web_server:app \
  --app-dir livekit-agent \
  --host 0.0.0.0 --port 8000 --reload
