"""Slow-drift detection over the main track's run history.

Per-run checks compare against the previous baseline, so a 1pp drop every
week never trips a 3pp threshold. Drift looks at the moving average instead:
  * floor:    MA(window) of pass rate below an absolute floor, or
  * max_drop: MA(window) dropped more than N pp vs the preceding window.
This is also what catches provider-side changes (a model alias being updated)
on nightly runs where nobody changed a prompt.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel

from .config import DriftConfig


class DriftResult(BaseModel):
    triggered: bool
    reasons: list[str]
    window: int
    n_points: int
    moving_average: float | None  # pass rate, 0-1
    previous_average: float | None
    series: list[float]  # pass rates, oldest first, including current


def detect_drift(series: list[float], cfg: DriftConfig) -> DriftResult:
    w = cfg.window
    ma = statistics.fmean(series[-w:]) if len(series) >= w else None
    prev = statistics.fmean(series[-2 * w:-w]) if len(series) >= 2 * w else None
    reasons: list[str] = []
    if ma is not None and ma < cfg.floor:
        reasons.append(f"{w}-run moving average pass rate {ma:.1%} is below the floor {cfg.floor:.0%}.")
    if ma is not None and prev is not None and (prev - ma) * 100 > cfg.max_drop:
        reasons.append(
            f"{w}-run moving average fell {(prev - ma) * 100:.1f}pp "
            f"({prev:.1%} -> {ma:.1%}) vs the preceding {w} runs (limit {cfg.max_drop:.1f}pp)."
        )
    return DriftResult(triggered=bool(reasons), reasons=reasons, window=w, n_points=len(series),
                       moving_average=ma, previous_average=prev, series=series)
