"""Data models for Memory Nexus."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class MemoryMetadata:
    """Structured metadata attached to a memory."""

    tags: list[str] = field(default_factory=list)
    source: str = ""
    importance: float = 0.5  # 0–1
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tags": self.tags,
            "source": self.source,
            "importance": self.importance,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryMetadata":
        return cls(
            tags=data.get("tags", []),
            source=data.get("source", ""),
            importance=data.get("importance", 0.5),
            extra={k: v for k, v in data.items() if k not in ("tags", "source", "importance")},
        )


@dataclass
class Memory:
    """A single memory entry."""

    id: str
    content: str
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
    created_at: datetime = field(default_factory=datetime.utcnow)
    similarity: Optional[float] = None  # set during search results

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at.isoformat(),
            "similarity": self.similarity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        meta = data.get("metadata", {})
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=MemoryMetadata.from_dict(meta),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.utcnow(),
            similarity=data.get("similarity"),
        )
