"""Self-contained HTML diff report (no JS, no external assets) and a
Markdown summary for PR comments."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .compare import Comparison
from .dataset import GoldenDataset
from .drift import DriftResult
from .runner import CaseResult
from .storage import RunRecord

STATUS_LABEL = {"pass": "PASS", "warn": "WARN", "critical": "FAIL"}

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--text:#1d1d1f;--muted:#6b6b70;--border:#e3e3e0;
--pass:#1f7a4d;--pass-bg:#e6f4ec;--warn:#9a6200;--warn-bg:#fdf3e1;--crit:#b3261e;--crit-bg:#fbe9e7;
--accent:#3b5bdb;--accent2:#8a8f98;--code:#f0f0ed}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1c1c1f;--text:#ececef;--muted:#9a9aa2;
--border:#2e2e33;--pass:#5fd394;--pass-bg:#16301f;--warn:#f2b552;--warn-bg:#33270f;--crit:#ff8a80;
--crit-bg:#3a1714;--accent:#8ea2ff;--accent2:#6b6f78;--code:#26262a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 12px}
.muted{color:var(--muted)}.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;font-size:12px;letter-spacing:.04em}
.pass{color:var(--pass);background:var(--pass-bg)}.warn{color:var(--warn);background:var(--warn-bg)}
.critical{color:var(--crit);background:var(--crit-bg)}
.head{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:16px}
.meta div{font-size:13px}.meta b{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}.kpi .d{font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--border);vertical-align:top}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{overflow-x:auto}.diff td{width:33%}.diff td:first-child{width:auto}
.cat{font-family:ui-monospace,Consolas,monospace;font-size:12px;background:var(--code);padding:1px 6px;border-radius:4px}
.bad{color:var(--crit)}.good{color:var(--pass)}
details summary{cursor:pointer;color:var(--accent)}pre{white-space:pre-wrap;background:var(--code);padding:8px;border-radius:6px;font-size:12px;margin:6px 0 0}
.notes li{margin:4px 0}svg text{fill:var(--muted);font-size:11px}
.legend span{display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-size:12px;color:var(--muted)}
.sw{width:12px;height:3px;border-radius:2px;display:inline-block}
"""


def _fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "-"
    if unit == "pp":
        return f"{v:.1f}%"
    if unit == "pts":
        return f"{v:.2f}"
    if unit == "ms":
        return f"{v:,.0f} ms"
    return f"{v:,.0f}"


def _fmt_delta(v: float | None, unit: str) -> str:
    if v is None:
        return "-"
    sign = "+" if v > 0 else ""
    if unit == "pp":
        return f"{sign}{v:.1f}pp"
    if unit == "pts":
        return f"{sign}{v:.2f}"
    if unit == "ms":
        return f"{sign}{v:,.0f} ms"
    return f"{sign}{v:,.0f}"


def _cat(v: str | None) -> str:
    return f'<span class="cat">{escape(v or "none")}</span>'


def _output_cell(r: CaseResult, expected_cat: str) -> str:
    cls = "good" if r.category_match else "bad"
    parts = [f'<div class="{cls}">{_cat(r.predicted_category)}</div>']
    if r.summary:
        parts.append(f"<div>{escape(r.summary)}</div>")
    score = "-" if r.judge_score is None else f"{r.judge_score}/5"
    parts.append(f'<div class="muted">judge {score}'
                 + (f" &middot; {escape(r.judge_reason)}" if r.judge_reason else "") + "</div>")
    if r.error:
        parts.append(f'<div class="bad">{escape(r.error)}</div>')
    return "".join(parts)


def trend_svg(points: list[tuple[str, float, float | None, bool]], ma_window: int) -> str:
    """points: (label, pass_rate 0-1, judge_mean 1-5 | None, is_current)."""
    if len(points) < 2:
        return '<p class="muted">Trend appears after two or more runs on this track.</p>'
    W, H, L, R, T, B = 1100, 240, 44, 16, 12, 30
    pw, ph = W - L - R, H - T - B
    lo = max(0.0, min(p[1] for p in points) - 0.05)
    hi = 1.0

    def x(i: int) -> float:
        return L + pw * i / (len(points) - 1)

    def y(v: float) -> float:
        return T + ph * (1 - (v - lo) / (hi - lo or 1))

    grid = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--border)"/>'
                    f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>')
    pr = " ".join(f"{x(i):.1f},{y(p[1]):.1f}" for i, p in enumerate(points))
    ma_pts = []
    for i in range(ma_window - 1, len(points)):
        avg = sum(p[1] for p in points[i - ma_window + 1:i + 1]) / ma_window
        ma_pts.append(f"{x(i):.1f},{y(avg):.1f}")
    jm = [(i, p[2]) for i, p in enumerate(points) if p[2] is not None]
    judge_line = " ".join(f"{x(i):.1f},{y(lo + (hi - lo) * (v - 1) / 4):.1f}" for i, v in jm)
    dots, labels = [], []
    step = max(1, len(points) // 12)
    for i, p in enumerate(points):
        r = 5 if p[3] else 3
        fill = "var(--crit)" if p[3] else "var(--accent)"
        dots.append(f'<circle cx="{x(i):.1f}" cy="{y(p[1]):.1f}" r="{r}" fill="{fill}">'
                    f'<title>{escape(p[0])}: pass {p[1]:.1%}'
                    + (f", judge {p[2]:.2f}" if p[2] is not None else "") + "</title></circle>")
        if i % step == 0 or p[3]:
            labels.append(f'<text x="{x(i):.1f}" y="{H - 10}" text-anchor="middle">{escape(p[0])}</text>')
    ma_svg = (f'<polyline points="{" ".join(ma_pts)}" fill="none" stroke="var(--accent2)" '
              f'stroke-width="2" stroke-dasharray="5 4"/>' if ma_pts else "")
    judge_svg = (f'<polyline points="{judge_line}" fill="none" stroke="var(--warn)" stroke-width="1.5" '
                 f'opacity=".8"/>' if len(jm) > 1 else "")
    return (
        f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Pass rate trend">'
        + "".join(grid) + judge_svg + ma_svg
        + f'<polyline points="{pr}" fill="none" stroke="var(--accent)" stroke-width="2"/>'
        + "".join(dots) + "".join(labels) + "</svg>"
        + '<div class="legend"><span><i class="sw" style="background:var(--accent)"></i>Pass rate</span>'
        f'<span><i class="sw" style="background:var(--accent2)"></i>{ma_window}-run moving average</span>'
        '<span><i class="sw" style="background:var(--warn)"></i>Judge mean (1-5, same scale band)</span>'
        '<span><i class="sw" style="background:var(--crit);height:8px;width:8px;border-radius:50%"></i>This run</span></div>'
    )


def render_html(*, run: RunRecord, baseline: RunRecord | None, comparison: Comparison,
                drift: DriftResult, history: list[RunRecord], results: list[CaseResult],
                dataset: GoldenDataset, overall_status: str) -> str:
    cases = {c.id: c for c in dataset.cases}
    m = run.metrics
    status_cls = overall_status

    def meta(label: str, cur: str, base: str | None) -> str:
        b = f'<span class="muted"> &larr; {escape(base)}</span>' if base and base != cur else ""
        return f"<div><b>{label}</b>{escape(cur)}{b}</div>"

    meta_html = "".join([
        meta("Prompt version", f"{run.prompt_version} ({run.prompt_hash})",
             f"{baseline.prompt_version} ({baseline.prompt_hash})" if baseline else None),
        meta("Model", run.model, baseline.model if baseline else None),
        meta("Judge", run.judge_model, baseline.judge_model if baseline else None),
        meta("Dataset", f"{run.dataset_version} ({run.dataset_hash}), {m.n_cases} cases",
             f"{baseline.dataset_version} ({baseline.dataset_hash})" if baseline else None),
        meta("Run", f"#{run.id} on {run.track} at {run.created_at}", None),
        meta("Baseline", f"#{baseline.id} at {baseline.created_at}" if baseline else "none", None),
        meta("Commit", run.git_sha or "-", None),
    ])

    checks = {c.name: c for c in comparison.checks}

    def kpi(name: str, value: str, check_name: str | None = None) -> str:
        c = checks.get(check_name or name)
        d = ""
        if c is not None:
            cls = {"critical": "bad", "warn": "bad", "pass": "muted"}[c.severity]
            if c.delta and c.delta > 0 and c.unit in ("pp", "pts"):
                cls = "good"
            d = f'<div class="d {cls}">{_fmt_delta(c.delta, c.unit)} vs baseline</div>'
        return f'<div class="panel kpi"><div class="muted">{name}</div><div class="v">{value}</div>{d}</div>'

    kpis = "".join([
        kpi("Pass rate", f"{m.pass_rate:.1%}"),
        kpi("Category accuracy", f"{m.category_accuracy:.1%}"),
        kpi("Summary score", f"{m.judge_mean:.2f}/5" if m.judge_mean is not None else "-",
            "Summary score (judge mean)"),
        f'<div class="panel kpi"><div class="muted">Flips</div><div class="v">'
        f'<span class="bad">{len(comparison.regressions)}&darr;</span> '
        f'<span class="good">{len(comparison.improvements)}&uarr;</span></div>'
        f'<div class="d muted">McNemar p = '
        f'{"-" if comparison.mcnemar_p is None else f"{comparison.mcnemar_p:.3f}"}</div></div>',
        f'<div class="panel kpi"><div class="muted">Latency p50 / p95</div>'
        f'<div class="v">{m.latency_p50_ms:,.0f}<span class="muted" style="font-size:14px"> / '
        f'{m.latency_p95_ms:,.0f} ms</span></div></div>',
        f'<div class="panel kpi"><div class="muted">Tokens (feature + judge)</div>'
        f'<div class="v">{m.total_tokens:,}</div><div class="d muted">+ {m.judge_tokens:,} judge</div></div>',
    ])

    check_rows = "".join(
        f"<tr><td>{escape(c.name)}"
        + (f' <span class="muted">({escape(c.detail)})</span>' if c.detail else "")
        + f'</td><td class="num">{_fmt(c.baseline, c.unit)}</td><td class="num">{_fmt(c.current, c.unit)}</td>'
        f'<td class="num">{_fmt_delta(c.delta, c.unit)}</td>'
        f'<td><span class="badge {c.severity}">{STATUS_LABEL[c.severity]}</span></td></tr>'
        for c in comparison.checks
    ) or '<tr><td colspan="5" class="muted">No baseline to compare against.</td></tr>'

    def flip_rows(flips) -> str:
        rows = []
        for f in flips:
            case = cases.get(f.case_id)
            email = escape(case.email) if case else "(case no longer in dataset)"
            exp = (f"{_cat(case.expected.category)}<div>{escape(case.expected.summary)}</div>"
                   if case else "")
            tags = ", ".join(case.tags) if case and case.tags else ""
            rows.append(
                f"<tr><td><b>{f.case_id}</b><div class='muted'>{f.new.difficulty}"
                + (f" &middot; {escape(tags)}" if tags else "")
                + f"</div><details><summary>email</summary><pre>{email}</pre></details></td>"
                f"<td>{exp}</td><td>{_output_cell(f.old, f.old.expected_category)}</td>"
                f"<td>{_output_cell(f.new, f.new.expected_category)}</td></tr>"
            )
        return "".join(rows)

    reg_html = (
        '<div class="panel scroll"><table class="diff"><tr><th>Case</th><th>Expected</th>'
        "<th>Baseline output</th><th>This run</th></tr>" + flip_rows(comparison.regressions) + "</table></div>"
        if comparison.regressions else '<p class="muted">No regressions.</p>'
    )
    imp_html = (
        f'<details><summary>{len(comparison.improvements)} improvement(s)</summary>'
        '<div class="panel scroll" style="margin-top:8px"><table class="diff"><tr><th>Case</th>'
        "<th>Expected</th><th>Baseline output</th><th>This run</th></tr>"
        + flip_rows(comparison.improvements) + "</table></div></details>"
        if comparison.improvements else ""
    )

    cat_rows = "".join(
        f"<tr><td>{_cat(k)}</td><td class='num'>{v.correct}/{v.n}</td><td class='num'>{v.accuracy:.1%}</td></tr>"
        for k, v in m.per_category.items()
    )
    diff_rows = "".join(
        f"<tr><td>{k}</td><td class='num'>{v.correct}/{v.n}</td><td class='num'>{v.accuracy:.1%}</td></tr>"
        for k, v in m.per_difficulty.items()
    )

    points = [(f"#{h.id}", h.metrics.pass_rate, h.metrics.judge_mean, False) for h in history]
    points.append((f"#{run.id}", m.pass_rate, m.judge_mean, True))
    drift_html = (
        f'<div class="panel" style="margin-bottom:12px"><span class="badge warn">SLOW DRIFT</span> '
        + " ".join(escape(r) for r in drift.reasons) + "</div>"
        if drift.triggered else
        f'<p class="muted">No slow drift. {drift.window}-run moving average: '
        + (f"{drift.moving_average:.1%}" if drift.moving_average is not None
           else f"needs {drift.window} runs ({drift.n_points} so far)") + ".</p>"
    )

    all_rows = "".join(
        f"<tr><td>{r.case_id}</td><td>{r.difficulty}</td><td>{_cat(r.expected_category)}</td>"
        f"<td class='{'good' if r.category_match else 'bad'}'>{_cat(r.predicted_category)}</td>"
        f"<td class='num'>{r.judge_score if r.judge_score is not None else '-'}</td>"
        f"<td>{escape(r.summary or r.error or '')}</td>"
        f"<td class='num'>{r.latency_ms:,.0f}</td>"
        f"<td><span class='badge {'pass' if r.passed else 'critical'}'>{'PASS' if r.passed else 'FAIL'}</span></td></tr>"
        for r in results
    )
    notes = comparison.notes + ([f"Still failing on both runs: {', '.join(comparison.still_failing)}"]
                                if comparison.still_failing else [])
    notes_html = ('<ul class="notes">' + "".join(f"<li>{escape(n)}</li>" for n in notes) + "</ul>"
                  if notes else "")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eval run #{run.id}: {STATUS_LABEL[status_cls]}</title><style>{CSS}</style></head><body><main>
<div class="panel"><div class="head"><div><h1>Support classifier eval &middot; run #{run.id}</h1>
<div class="muted">{escape(run.prompt_version)} vs {escape(baseline.prompt_version) if baseline else "no baseline"}</div></div>
<span class="badge {status_cls}" style="font-size:14px;padding:4px 14px">{STATUS_LABEL[status_cls]}</span></div>
<div class="meta">{meta_html}</div></div>
<h2>Scorecard</h2><div class="kpis">{kpis}</div>
<h2>Checks vs baseline</h2><div class="panel scroll"><table><tr><th>Metric</th><th>Baseline</th><th>This run</th>
<th>Delta</th><th>Status</th></tr>{check_rows}</table>{notes_html}</div>
<h2>Regressions ({len(comparison.regressions)})</h2>{reg_html}{imp_html}
<h2>Trend</h2>{drift_html}<div class="panel">{trend_svg(points, drift.window)}</div>
<h2>Breakdown</h2><div class="kpis"><div class="panel"><table><tr><th>Category</th><th>Correct</th><th>Accuracy</th></tr>{cat_rows}</table></div>
<div class="panel"><table><tr><th>Difficulty</th><th>Passed</th><th>Pass rate</th></tr>{diff_rows}</table></div></div>
<h2>All cases</h2><details><summary>Show {len(results)} cases</summary><div class="panel scroll" style="margin-top:8px">
<table><tr><th>Case</th><th>Diff.</th><th>Expected</th><th>Predicted</th><th>Judge</th><th>Summary / error</th><th>ms</th><th></th></tr>
{all_rows}</table></div></details>
<p class="muted" style="margin-top:32px">A case passes when the category matches exactly and the judge scores the summary at or above the configured minimum.</p>
</main></body></html>"""


def headline(comparison: Comparison, run: RunRecord, baseline: RunRecord | None) -> str:
    m = run.metrics
    if not comparison.has_baseline or baseline is None:
        return f"No baseline yet. Pass rate {m.pass_rate:.1%}, category accuracy {m.category_accuracy:.1%}."
    pr = next(c for c in comparison.checks if c.name == "Pass rate")
    n = len(comparison.regressions)
    verb = "dropped" if (pr.delta or 0) < 0 else ("rose" if (pr.delta or 0) > 0 else "held")
    return (f"{n} regression{'s' if n != 1 else ''} detected, {len(comparison.improvements)} improved; "
            f"pass rate {verb} from {pr.baseline:.1f}% to {pr.current:.1f}%.")


def render_markdown(*, run: RunRecord, baseline: RunRecord | None, comparison: Comparison,
                    drift: DriftResult, overall_status: str, report_url: str | None) -> str:
    icon = {"pass": "✅", "warn": "⚠️", "critical": "❌"}[overall_status]
    lines = [f"## {icon} LLM eval: {STATUS_LABEL[overall_status]}", "",
             f"**{headline(comparison, run, baseline)}**", "",
             f"Prompt `{run.prompt_version}` on `{run.model}`"
             + (f" vs baseline `{baseline.prompt_version}` on `{baseline.model}` (run #{baseline.id})"
                if baseline else ""), ""]
    if comparison.checks:
        lines += ["| Metric | Baseline | This run | Delta | |", "|---|---:|---:|---:|---|"]
        for c in comparison.checks:
            lines.append(f"| {c.name} | {_fmt(c.baseline, c.unit)} | {_fmt(c.current, c.unit)} | "
                         f"{_fmt_delta(c.delta, c.unit)} | {STATUS_LABEL[c.severity]} |")
        lines.append("")
    if comparison.regressions:
        lines += ["<details><summary>Regressed cases</summary>", "",
                  "| Case | Expected | Baseline | This run |", "|---|---|---|---|"]
        for f in comparison.regressions[:25]:
            def cell(r: CaseResult) -> str:
                s = f"`{r.predicted_category}` judge {r.judge_score or '-'}"
                return s + (f" ({r.error[:60]})" if r.error else "")
            lines.append(f"| {f.case_id} | `{f.new.expected_category}` | {cell(f.old)} | {cell(f.new)} |")
        lines += ["", "</details>", ""]
    if drift.triggered:
        lines += ["**Slow drift:** " + " ".join(drift.reasons), ""]
    for n in comparison.notes:
        lines.append(f"> {n}")
    if report_url:
        lines += ["", f"[Full diff report]({report_url})"]
    if overall_status == "critical":
        lines += ["", "Merge is blocked: at least one metric crossed a critical threshold."]
    return "\n".join(lines) + "\n"


def write(path: str | Path, content: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
