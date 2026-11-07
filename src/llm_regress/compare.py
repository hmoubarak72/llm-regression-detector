"""Diff a run against its baseline: deltas, flips, significance, severity."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

from .config import CATEGORIES, Thresholds
from .metrics import compute_metrics
from .runner import CaseResult

Severity = Literal["pass", "warn", "critical"]
_RANK = {"pass": 0, "warn": 1, "critical": 2}


def worst(*levels: Severity) -> Severity:
    return max(levels, key=lambda s: _RANK[s])  # type: ignore[return-value]


class Check(BaseModel):
    name: str
    baseline: float | None
    current: float | None
    delta: float | None  # current - baseline, in display units (pp or score points)
    unit: str
    severity: Severity
    detail: str = ""


class CaseFlip(BaseModel):
    case_id: str
    old: CaseResult
    new: CaseResult


class Comparison(BaseModel):
    has_baseline: bool
    status: Severity
    checks: list[Check]
    regressions: list[CaseFlip]
    improvements: list[CaseFlip]
    still_failing: list[str]
    relabeled: list[str]
    added_cases: list[str]
    removed_cases: list[str]
    n_compared: int
    mcnemar_p: float | None
    significant: bool | None
    notes: list[str]


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pairs.
    b = pass->fail, c = fail->pass. Under H0 each discordant pair is a fair coin."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _drop_severity(drop: float, warn: float, critical: float) -> Severity:
    """drop > 0 means things got worse."""
    if drop > critical:
        return "critical"
    if drop > warn:
        return "warn"
    return "pass"


def _downgrade(s: Severity) -> Severity:
    return {"critical": "warn", "warn": "pass", "pass": "pass"}[s]  # type: ignore[return-value]


def compare_runs(current: list[CaseResult], baseline: list[CaseResult] | None,
                 thresholds: Thresholds) -> Comparison:
    if not baseline:
        return Comparison(
            has_baseline=False, status="pass", checks=[], regressions=[], improvements=[],
            still_failing=[r.case_id for r in current if not r.passed], relabeled=[],
            added_cases=[], removed_cases=[], n_compared=0, mcnemar_p=None, significant=None,
            notes=["No baseline run found. This run has nothing to be compared against."],
        )

    cur = {r.case_id: r for r in current}
    base = {r.case_id: r for r in baseline}
    common = sorted(cur.keys() & base.keys())
    relabeled = [cid for cid in common if cur[cid].expected_category != base[cid].expected_category]
    comparable = [cid for cid in common if cid not in relabeled]
    added = sorted(cur.keys() - base.keys())
    removed = sorted(base.keys() - cur.keys())

    notes: list[str] = []
    if added or removed:
        notes.append(f"Dataset changed: +{len(added)} / -{len(removed)} cases. "
                     f"Deltas use the {len(comparable)} cases present in both runs.")
    if relabeled:
        notes.append(f"{len(relabeled)} case(s) relabeled since baseline; excluded from deltas.")

    cm = compute_metrics(cur[c] for c in comparable)
    bm = compute_metrics(base[c] for c in comparable)
    t = thresholds
    checks: list[Check] = []

    d = (cm.pass_rate - bm.pass_rate) * 100
    checks.append(Check(name="Pass rate", baseline=bm.pass_rate * 100, current=cm.pass_rate * 100,
                        delta=d, unit="pp", severity=_drop_severity(-d, t.overall.warn, t.overall.critical)))
    d = (cm.category_accuracy - bm.category_accuracy) * 100
    checks.append(Check(name="Category accuracy", baseline=bm.category_accuracy * 100,
                        current=cm.category_accuracy * 100, delta=d, unit="pp",
                        severity=_drop_severity(-d, t.overall.warn, t.overall.critical)))
    if cm.judge_mean is not None and bm.judge_mean is not None:
        d = cm.judge_mean - bm.judge_mean
        checks.append(Check(name="Summary score (judge mean)", baseline=bm.judge_mean,
                            current=cm.judge_mean, delta=d, unit="pts",
                            severity=_drop_severity(-d, t.judge_mean.warn, t.judge_mean.critical)))
    for cat in CATEGORIES:
        if cat in cm.per_category and cat in bm.per_category:
            c_acc, b_acc = cm.per_category[cat].accuracy, bm.per_category[cat].accuracy
            d = (c_acc - b_acc) * 100
            checks.append(Check(name=f"Accuracy: {cat}", baseline=b_acc * 100, current=c_acc * 100,
                                delta=d, unit="pp", detail=f"n={cm.per_category[cat].n}",
                                severity=_drop_severity(-d, t.per_category.warn, t.per_category.critical)))
    # Errors (schema violations, timeouts) are informational unless they cause fails above.
    checks.append(Check(name="Errors", baseline=bm.errors, current=cm.errors, delta=cm.errors - bm.errors,
                        unit="", severity="warn" if cm.errors > bm.errors else "pass"))
    if bm.latency_p95_ms:
        d = cm.latency_p95_ms - bm.latency_p95_ms
        checks.append(Check(name="Latency p95", baseline=bm.latency_p95_ms, current=cm.latency_p95_ms,
                            delta=d, unit="ms", severity="pass",
                            detail="informational"))
    if bm.total_tokens:
        checks.append(Check(name="Feature tokens", baseline=bm.total_tokens, current=cm.total_tokens,
                            delta=cm.total_tokens - bm.total_tokens, unit="", severity="pass",
                            detail="informational"))

    regressions = [CaseFlip(case_id=c, old=base[c], new=cur[c])
                   for c in comparable if base[c].passed and not cur[c].passed]
    improvements = [CaseFlip(case_id=c, old=base[c], new=cur[c])
                    for c in comparable if not base[c].passed and cur[c].passed]
    still_failing = [c for c in comparable if not base[c].passed and not cur[c].passed]

    p = mcnemar_exact_p(len(regressions), len(improvements))
    significant = p < t.significance.alpha
    if t.significance.gate and not significant:
        for ch in checks:
            if ch.unit == "pp":
                ch.severity = _downgrade(ch.severity)
        notes.append(f"Significance gate on: flip pattern not significant (p={p:.3f}); "
                     f"accuracy severities downgraded one level.")

    status = worst("pass", *(c.severity for c in checks))
    return Comparison(
        has_baseline=True, status=status, checks=checks, regressions=regressions,
        improvements=improvements, still_failing=still_failing, relabeled=relabeled,
        added_cases=added, removed_cases=removed, n_compared=len(comparable),
        mcnemar_p=p, significant=significant, notes=notes,
    )
