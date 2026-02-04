#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper around run_agent.sh to avoid pasting markdown-link paths.
# Usage:
#   ./livekit-agent/run_connect.sh <room> [identity]
# Examples:
#   ./livekit-agent/run_connect.sh calltour
#   ./livekit-agent/run_connect.sh calltour agent

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
	exec "$(dirname "$0")/run_agent.sh" connect --help
fi

# If the first arg looks like an option, forward everything to `connect`.
if [[ "${1:-}" == -* ]]; then
	exec "$(dirname "$0")/run_agent.sh" connect "$@"
fi

room="${1:-calltour}"
identity="${2:-agent}"

exec "$(dirname "$0")/run_agent.sh" connect --room "$room" --participant-identity "$identity"
