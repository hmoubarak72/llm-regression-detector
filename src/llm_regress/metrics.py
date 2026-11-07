"""Aggregate metrics for a run (or a subset of a run's cases)."""

from __future__ import annotations

import statistics
from typing import Iterable

from pydantic import BaseModel

from .config import CATEGORIES
from .runner import CaseResult


class GroupStat(BaseModel):
    n: int
    correct: int
    accuracy: float


class RunMetrics(BaseModel):
    n_cases: int
    pass_rate: float
    category_accuracy: float
    judge_mean: float | None
    per_category: dict[str, GroupStat]
    per_difficulty: dict[str, GroupStat]  # pass rate by difficulty
    latency_p50_ms: float
    latency_p95_ms: float
    total_tokens: int
    judge_tokens: int
    errors: int


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[idx]


def compute_metrics(results: Iterable[CaseResult]) -> RunMetrics:
    results = list(results)
    n = len(results)
    per_cat: dict[str, GroupStat] = {}
    for cat in CATEGORIES:
        rows = [r for r in results if r.expected_category == cat]
        if rows:
            correct = sum(r.category_match for r in rows)
            per_cat[cat] = GroupStat(n=len(rows), correct=correct, accuracy=correct / len(rows))
    per_diff: dict[str, GroupStat] = {}
    for diff in ("easy", "medium", "hard"):
        rows = [r for r in results if r.difficulty == diff]
        if rows:
            ok = sum(r.passed for r in rows)
            per_diff[diff] = GroupStat(n=len(rows), correct=ok, accuracy=ok / len(rows))
    scores = [r.judge_score for r in results if r.judge_score is not None]
    lat = [r.latency_ms for r in results if r.latency_ms]
    return RunMetrics(
        n_cases=n,
        pass_rate=sum(r.passed for r in results) / n if n else 0.0,
        category_accuracy=sum(r.category_match for r in results) / n if n else 0.0,
        judge_mean=statistics.fmean(scores) if scores else None,
        per_category=per_cat,
        per_difficulty=per_diff,
        latency_p50_ms=_pct(lat, 0.50),
        latency_p95_ms=_pct(lat, 0.95),
        total_tokens=sum(r.prompt_tokens + r.completion_tokens for r in results),
        judge_tokens=sum(r.judge_tokens for r in results),
        errors=sum(r.error is not None for r in results),
    )
