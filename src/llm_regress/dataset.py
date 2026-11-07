"""Golden dataset loading and validation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .config import Category

Difficulty = Literal["easy", "medium", "hard"]
KNOWN_TAGS = {"ambiguous", "short", "typos", "mixed_language", "sarcastic",
              "multi_issue", "long", "prompt_injection"}


class Expected(BaseModel):
    category: Category
    summary: str = Field(min_length=1)


class GoldenCase(BaseModel):
    id: str = Field(pattern=r"^GS-\d{3,}$")
    email: str = Field(min_length=1)
    expected: Expected
    expected_difficulty: Difficulty
    tags: list[str] = []
    notes: str = Field(min_length=1)
    verified: bool = False


class GoldenDataset(BaseModel):
    dataset: str
    version: str
    updated_at: str
    labeling_guide: str | None = None
    changelog: list[dict] = []
    cases: list[GoldenCase]

    @model_validator(mode="after")
    def _unique_ids(self) -> "GoldenDataset":
        ids = [c.id for c in self.cases]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate case ids: {sorted(dupes)}")
        return self

    @property
    def content_hash(self) -> str:
        payload = json.dumps([c.model_dump() for c in self.cases], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path: str | Path) -> "GoldenDataset":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def lint(self) -> list[str]:
        """Non-fatal quality warnings."""
        warnings: list[str] = []
        unverified = [c.id for c in self.cases if not c.verified]
        if unverified:
            warnings.append(
                f"{len(unverified)}/{len(self.cases)} cases are not human-verified "
                f"(verified=false). Review them against golden/LABELING.md."
            )
        for c in self.cases:
            unknown = set(c.tags) - KNOWN_TAGS
            if unknown:
                warnings.append(f"{c.id}: unknown tags {sorted(unknown)}")
            words = len(re.findall(r"\S+", c.expected.summary))
            if words > 30:
                warnings.append(f"{c.id}: reference summary has {words} words (max 30)")
        counts: dict[str, int] = {}
        for c in self.cases:
            counts[c.expected.category] = counts.get(c.expected.category, 0) + 1
        for cat, n in counts.items():
            if n < 10:
                warnings.append(f"category '{cat}' has only {n} cases; per-category deltas will be noisy")
        return warnings
