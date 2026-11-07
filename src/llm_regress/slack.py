"""Slack incoming-webhook alerts (stdlib only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .compare import Comparison
from .drift import DriftResult
from .report import STATUS_LABEL, headline
from .storage import RunRecord

EMOJI = {"pass": ":white_check_mark:", "warn": ":warning:", "critical": ":rotating_light:"}


def build_payload(*, run: RunRecord, baseline: RunRecord | None, comparison: Comparison,
                  drift: DriftResult, overall_status: str, report_url: str | None,
                  context: str | None = None) -> dict:
    title = f"{EMOJI[overall_status]} LLM eval {STATUS_LABEL[overall_status]}: support classifier"
    text = headline(comparison, run, baseline)
    fields = [
        {"type": "mrkdwn", "text": f"*Prompt*\n`{run.prompt_version}`"
                                   + (f" (baseline `{baseline.prompt_version}`)" if baseline else "")},
        {"type": "mrkdwn", "text": f"*Model*\n`{run.model}`"},
        {"type": "mrkdwn", "text": f"*Pass rate*\n{run.metrics.pass_rate:.1%}"},
        {"type": "mrkdwn", "text": f"*Regressions / improvements*\n"
                                   f"{len(comparison.regressions)} / {len(comparison.improvements)}"},
    ]
    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": title[:150], "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{text}*"}},
        {"type": "section", "fields": fields},
    ]
    failing = [c for c in comparison.checks if c.severity != "pass"]
    if failing:
        lines = "\n".join(f"• {c.name}: {c.baseline:.1f} → {c.current:.1f} ({STATUS_LABEL[c.severity]})"
                          for c in failing[:8] if c.baseline is not None and c.current is not None)
        if lines:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": lines}})
    if drift.triggered:
        blocks.append({"type": "section", "text": {"type": "mrkdwn",
                                                   "text": ":chart_with_downwards_trend: *Slow drift*\n"
                                                           + "\n".join(drift.reasons)}})
    if report_url:
        blocks.append({"type": "actions", "elements": [{
            "type": "button", "text": {"type": "plain_text", "text": "Open diff report"},
            "url": report_url, "style": "danger" if overall_status == "critical" else "primary"}]})
    ctx = f"run #{run.id} · {run.track}" + (f" · {run.git_sha[:8]}" if run.git_sha else "")
    if context:
        ctx += f" · {context}"
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": ctx}]})
    return {"text": f"{title}: {text}", "blocks": blocks}


def post(webhook_url: str, payload: dict, retries: int = 2) -> None:
    data = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(webhook_url, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"Slack rejected the alert: HTTP {exc.code} {exc.read()[:200]!r}")
        except urllib.error.URLError:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError("Slack alert failed after retries")
