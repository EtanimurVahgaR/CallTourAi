from __future__ import annotations

import argparse

from .graph import build_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run basic LangGraph RAG")
    parser.add_argument("question", nargs="+", help="The question to ask")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    app = build_app()

    result = app.invoke({"question": question})
    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
