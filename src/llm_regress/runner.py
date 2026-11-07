"""Runs every golden case through the feature and the judge, concurrently."""

from __future__ import annotations

import asyncio
import sys
from typing import Callable

from pydantic import BaseModel

from .config import PromptConfig, Thresholds
from .dataset import GoldenCase, GoldenDataset
from .feature import aclassify_email
from .judge import judge_summary
from .providers import Provider


class CaseResult(BaseModel):
    case_id: str
    expected_category: str
    difficulty: str
    predicted_category: str | None = None
    summary: str | None = None
    category_match: bool = False
    judge_score: int | None = None
    judge_reason: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    judge_tokens: int = 0
    passed: bool = False
    error: str | None = None


async def _run_case(case: GoldenCase, prompt: PromptConfig, classifier: Provider, judge: Provider,
                    thresholds: Thresholds, sem: asyncio.Semaphore) -> CaseResult:
    res = CaseResult(case_id=case.id, expected_category=case.expected.category,
                     difficulty=case.expected_difficulty)
    async with sem:
        try:
            out, resp, err = await aclassify_email(case.email, prompt, classifier)
            res.latency_ms = round(resp.latency_ms, 1)
            res.prompt_tokens, res.completion_tokens = resp.prompt_tokens, resp.completion_tokens
            if out is None:
                res.error = err
                return res
            res.predicted_category = out.category
            res.summary = out.summary
            res.category_match = out.category == case.expected.category

            jr = await judge_summary(case.email, case.expected.summary, out.summary, judge,
                                     thresholds.judge.model)
            res.judge_tokens = jr.tokens
            if jr.verdict is None:
                res.error = jr.error
            else:
                res.judge_score, res.judge_reason = jr.verdict.score, jr.verdict.reason
        except Exception as exc:  # network errors after SDK retries, etc.
            res.error = f"{type(exc).__name__}: {exc}"[:300]
            return res
    res.passed = bool(
        res.category_match
        and res.judge_score is not None
        and res.judge_score >= thresholds.pass_criteria.min_judge_score
    )
    return res


async def run_eval(prompt: PromptConfig, dataset: GoldenDataset, classifier: Provider, judge: Provider,
                   thresholds: Thresholds,
                   on_progress: Callable[[int, int], None] | None = None) -> list[CaseResult]:
    sem = asyncio.Semaphore(thresholds.runner.concurrency)
    tasks = [asyncio.create_task(_run_case(c, prompt, classifier, judge, thresholds, sem))
             for c in dataset.cases]
    done = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        done += 1
        if on_progress:
            on_progress(done, len(tasks))
    results = [t.result() for t in tasks]
    order = {c.id: i for i, c in enumerate(dataset.cases)}
    return sorted(results, key=lambda r: order[r.case_id])


def stderr_progress(done: int, total: int) -> None:
    if done == total or done % 10 == 0:
        print(f"  {done}/{total} cases", file=sys.stderr)
