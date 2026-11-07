"""Typed contracts: prompt configs, feature output, thresholds.

Everything the pipeline consumes is loaded through these models so a malformed
prompt file or threshold override fails loudly at load time, not mid-run.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

CATEGORIES: tuple[str, ...] = ("billing", "technical", "account", "general")
Category = Literal["billing", "technical", "account", "general"]


# --------------------------------------------------------------------------- #
# Feature output contract
# --------------------------------------------------------------------------- #
class ClassificationResult(BaseModel):
    """What the support-email classifier must return."""

    category: Category
    summary: str = Field(min_length=1, max_length=400)


CLASSIFICATION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "summary": {"type": "string"},
    },
    "required": ["category", "summary"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Prompt config (the "code" CI runs against)
# --------------------------------------------------------------------------- #
class FewShotExample(BaseModel):
    email: str
    category: Category
    summary: str


class ModelSettings(BaseModel):
    provider: Literal["openai", "mock"] = "openai"
    name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 200


class PromptConfig(BaseModel):
    """A versioned prompt. Model settings live here on purpose: changing the
    model is a change to a file under /prompts, so it triggers CI like any
    prompt edit."""

    version: str
    created_at: datetime
    description: str = ""
    model: ModelSettings = ModelSettings()
    system_prompt: str
    few_shot: list[FewShotExample] = []
    source_path: str | None = Field(default=None, exclude=True)

    @property
    def content_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptConfig":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = cls.model_validate(data)
        cfg.source_path = str(path)
        expected_stem = f"v{cfg.version}" if not cfg.version.startswith("v") else cfg.version
        if path.stem != expected_stem:
            raise ValueError(
                f"{path}: file name must match version (expected {expected_stem}.yaml)"
            )
        return cfg


def resolve_active_prompt(prompts_dir: str | Path = "prompts") -> Path:
    """prompts/active.yaml points at the version currently in production."""
    prompts_dir = Path(prompts_dir)
    pointer = yaml.safe_load((prompts_dir / "active.yaml").read_text(encoding="utf-8"))
    return prompts_dir / pointer["active"]


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
class PassCriteria(BaseModel):
    min_judge_score: int = Field(default=3, ge=1, le=5)


class DeltaThresholds(BaseModel):
    """Drops in percentage points (pp) vs baseline. 3.0 means 94% -> 91%."""

    warn: float = 3.0
    critical: float = 8.0


class JudgeMeanThresholds(BaseModel):
    """Drops in mean judge score (1-5 scale)."""

    warn: float = 0.2
    critical: float = 0.5


class SignificanceConfig(BaseModel):
    alpha: float = 0.05
    # If true, a threshold breach whose McNemar p-value >= alpha is downgraded
    # one level (critical -> warn, warn -> pass). Off by default: with 60-100
    # cases the test is underpowered and we prefer false alarms to misses.
    gate: bool = False


class DriftConfig(BaseModel):
    window: int = 7
    floor: float = 0.85  # alert if moving-average pass rate < floor
    max_drop: float = 3.0  # alert if MA dropped > max_drop pp vs previous window


class RunnerConfig(BaseModel):
    concurrency: int = 8
    max_retries: int = 4
    timeout_s: float = 60.0
    # If this share of cases errors (auth, quota, outage), the run is a
    # pipeline failure (exit 2), not a quality regression, and is never a baseline.
    max_error_rate: float = 0.5


class JudgeConfig(BaseModel):
    provider: Literal["openai", "mock"] = "openai"
    model: str = "gpt-4o"


class Thresholds(BaseModel):
    pass_criteria: PassCriteria = PassCriteria()
    overall: DeltaThresholds = DeltaThresholds()
    per_category: DeltaThresholds = DeltaThresholds(warn=10.0, critical=20.0)
    judge_mean: JudgeMeanThresholds = JudgeMeanThresholds()
    significance: SignificanceConfig = SignificanceConfig()
    drift: DriftConfig = DriftConfig()
    runner: RunnerConfig = RunnerConfig()
    judge: JudgeConfig = JudgeConfig()

    @classmethod
    def load(cls, path: str | Path | None = None, env: dict[str, str] | None = None) -> "Thresholds":
        """Load YAML, then apply env overrides of the form
        LLMREG_<SECTION>_<KEY>, e.g. LLMREG_OVERALL_CRITICAL=10."""
        data: dict = {}
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        env = dict(os.environ) if env is None else env
        for section, model in cls.model_fields.items():
            sub_model = model.annotation
            for key in sub_model.model_fields:  # type: ignore[union-attr]
                var = f"LLMREG_{section}_{key}".upper()
                if var in env:
                    data.setdefault(section, {})[key] = env[var]
        return cls.model_validate(data)
