#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BIN="/home/miniconda3/bin/conda"
ENV_NAME="calltourai"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "ERROR: conda not found at $CONDA_BIN" >&2
  echo "Edit livekit-agent/run_agent.sh and set CONDA_BIN to your conda path." >&2
  exit 1
fi

exec "$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output python livekit-agent/livekit_basic_agent.py "$@"
