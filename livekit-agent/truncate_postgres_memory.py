#!/usr/bin/env python3
"""Truncate CallTourAI LiveKit persisted-memory table in PostgreSQL.

This script deletes *all* rows from the agent memory table used by
`PostgresMemoryStore`.

Safety
- Requires an explicit `--yes` flag to run.
- Prints row counts before and after.

Usage
  # from repo root (recommended):
  python livekit-agent/truncate_postgres_memory.py --yes

  # or pass an explicit DB URL:
  python livekit-agent/truncate_postgres_memory.py --database-url "postgresql+psycopg2://..." --yes

Environment
- DATABASE_URL: SQLAlchemy URL, e.g. postgresql+psycopg2://user:pass@localhost:5432/calltourai
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import create_engine, text


TABLE_NAME = "calltour_livekit_room_memory"


def _required_database_url(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()

    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise SystemExit(
            "Missing DATABASE_URL. Set it in livekit-agent/.env (or export it), "
            "or pass --database-url."
        )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TRUNCATE all records from CallTourAI's persisted LiveKit memory table."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy DB URL (defaults to env DATABASE_URL).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually run the TRUNCATE (required).",
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Use TRUNCATE ... CASCADE (only if you know you need it).",
    )

    args = parser.parse_args(argv)

    if not args.yes:
        print(
            "Refusing to run without --yes.\n\n"
            "This will permanently delete ALL persisted conversation memory rows from:\n"
            f"  {TABLE_NAME}\n",
            file=sys.stderr,
        )
        return 2

    database_url = _required_database_url(args.database_url)
    engine = create_engine(database_url, pool_pre_ping=True, future=True)

    with engine.connect() as conn:
        # Validate table exists (and show pre-count).
        exists = conn.execute(
            text(
                "SELECT to_regclass(:tbl) IS NOT NULL AS exists",
            ),
            {"tbl": TABLE_NAME},
        ).scalar()

        if not exists:
            print(f"Table not found: {TABLE_NAME}. Nothing to truncate.")
            return 0

        before = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
        before_count = int(before.scalar() or 0)

        truncate_sql = f"TRUNCATE TABLE {TABLE_NAME}" + (" CASCADE" if args.cascade else "")
        conn.execute(text(truncate_sql))
        conn.commit()

        after = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}"))
        after_count = int(after.scalar() or 0)

    print(
        "Truncate complete.\n"
        f"- Table: {TABLE_NAME}\n"
        f"- Rows before: {before_count}\n"
        f"- Rows after:  {after_count}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
