from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import DateTime, String, UniqueConstraint, create_engine, func, select
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class AgentRoomMemory(Base):
    __tablename__ = "calltour_livekit_room_memory"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    room_name: Mapped[str] = mapped_column(String(255), nullable=False)
    user_identity: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    # JSON-serializable chat items (we store a sanitized subset of LiveKit ChatItems)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)

    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("room_name", "user_identity", name="uq_calltour_room_identity"),
    )


class UserProfile(Base):
    __tablename__ = "calltour_user_profiles"

    user_identity: Mapped[str] = mapped_column(String(255), primary_key=True)
    memories: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)

    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


@dataclass(frozen=True)
class MemoryKey:
    room_name: str
    user_identity: str = ""


class PostgresMemoryStore:
    def __init__(self, database_url: str):
        self._engine = create_engine(
            database_url,
            pool_pre_ping=True,
            future=True,
        )

    def ensure_schema(self) -> None:
        Base.metadata.create_all(self._engine)

    def load_items(self, key: MemoryKey) -> list[dict[str, Any]]:
        with Session(self._engine) as session:
            stmt = select(AgentRoomMemory.items).where(
                AgentRoomMemory.room_name == key.room_name,
                AgentRoomMemory.user_identity == (key.user_identity or ""),
            )
            row = session.execute(stmt).one_or_none()
            if not row:
                return []
            return list(row[0] or [])

    def upsert_items(self, *, key: MemoryKey, items: Sequence[dict[str, Any]]) -> None:
        payload = list(items)

        stmt = insert(AgentRoomMemory).values(
            room_name=key.room_name,
            user_identity=(key.user_identity or ""),
            items=payload,
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=[AgentRoomMemory.room_name, AgentRoomMemory.user_identity],
            set_={
                "items": payload,
                "updated_at": func.now(),
            },
        )

        with Session(self._engine) as session:
            session.execute(stmt)
            session.commit()

    def load_profile(self, user_identity: str) -> list[dict[str, Any]]:
        if not user_identity:
            return []
        with Session(self._engine) as session:
            stmt = select(UserProfile.memories).where(UserProfile.user_identity == user_identity)
            row = session.execute(stmt).one_or_none()
            if not row:
                return []
            return list(row[0] or [])

    def append_memories(self, user_identity: str, new_memories: list[dict[str, Any]]) -> None:
        if not user_identity or not new_memories:
            return

        with Session(self._engine) as session:
            # Load existing
            stmt_select = select(UserProfile.memories).where(UserProfile.user_identity == user_identity)
            row = session.execute(stmt_select).one_or_none()
            existing = list(row[0] or []) if row else []
            
            # Simple append + dedupe might be needed, but for now just append
            updated = existing + new_memories
            
            stmt_upsert = insert(UserProfile).values(
                user_identity=user_identity,
                memories=updated
            )
            stmt_upsert = stmt_upsert.on_conflict_do_update(
                index_elements=[UserProfile.user_identity],
                set_={
                    "memories": updated,
                    "updated_at": func.now(),
                }
            )
            session.execute(stmt_upsert)
            session.commit()
