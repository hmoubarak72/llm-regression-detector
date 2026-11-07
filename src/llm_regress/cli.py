"""Command line entry point: `llm-regress run | history | validate | report`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from . import slack
from .compare import Comparison, compare_runs, worst
from .config import PromptConfig, Thresholds, resolve_active_prompt
from .dataset import GoldenDataset
from .drift import DriftResult, detect_drift
from .metrics import compute_metrics
from .providers import build_provider
from .report import headline, render_html, render_markdown, write
from .runner import run_eval, stderr_progress
from .storage import RunRecord, Store

EXIT_OK, EXIT_REGRESSION, EXIT_ERROR = 0, 1, 2


def _git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:12]
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _default_report_url() -> str | None:
    if os.environ.get("LLMREG_REPORT_URL"):
        return os.environ["LLMREG_REPORT_URL"]
    if os.environ.get("GITHUB_RUN_ID"):
        return (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    return None


def _drift_for(store: Store, run: RunRecord, track: str, thresholds: Thresholds) -> DriftResult:
    hist = [h for h in store.history(track=track, limit=thresholds.drift.window * 2 + 1) if h.id != run.id]
    hist = hist[-(thresholds.drift.window * 2 - 1):]
    return detect_drift([h.metrics.pass_rate for h in hist] + [run.metrics.pass_rate], thresholds.drift)


def _outputs(args, store: Store, run: RunRecord, baseline: RunRecord | None, comparison: Comparison,
             dataset: GoldenDataset, thresholds: Thresholds) -> tuple[str, DriftResult]:
    args.drift_track = args.drift_track or getattr(args, "track", None) or "main"
    drift = _drift_for(store, run, args.drift_track, thresholds)
    overall = comparison.status
    if drift.triggered:
        overall = worst(overall, "warn")  # drift warns; it never blocks a merge on its own
    if not comparison.has_baseline and args.require_baseline:
        overall = worst(overall, "warn")
        comparison.notes.append("A baseline was required but none was found (CI cache miss?).")
    store.set_status(run.id, overall)

    history = [h for h in store.history(track=args.drift_track, limit=args.trend_runs) if h.id != run.id]
    results = store.results(run.id)
    report_url = args.report_url or _default_report_url()
    html = render_html(run=run, baseline=baseline, comparison=comparison, drift=drift, history=history,
                       results=results, dataset=dataset, overall_status=overall)
    out = Path(args.out_dir)
    report_path = write(out / "report.html", html)
    write(out / "summary.md", render_markdown(run=run, baseline=baseline, comparison=comparison,
                                              drift=drift, overall_status=overall,
                                              report_url=report_url))
    write(out / "summary.json", json.dumps({
        "run_id": run.id, "baseline_run_id": baseline.id if baseline else None, "status": overall,
        "headline": headline(comparison, run, baseline), "drift": drift.model_dump(),
        "comparison": comparison.model_dump(exclude={"regressions", "improvements"}),
        "regressions": [f.case_id for f in comparison.regressions],
        "improvements": [f.case_id for f in comparison.improvements],
        "metrics": run.metrics.model_dump(),
    }, indent=2))
    print(f"\n{overall.upper()}: {headline(comparison, run, baseline)}")
    if drift.triggered:
        print("SLOW DRIFT: " + " ".join(drift.reasons))
    print(f"Report: {report_path.resolve()}")

    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    should_alert = args.slack == "always" or (args.slack == "alert" and overall != "pass")
    if webhook and should_alert:
        try:
            slack.post(webhook, slack.build_payload(run=run, baseline=baseline, comparison=comparison,
                                                    drift=drift, overall_status=overall,
                                                    report_url=report_url,
                                                    context=os.environ.get("LLMREG_CONTEXT")))
            print("Slack alert sent.")
        except Exception as exc:  # alerting must never mask the eval result
            print(f"WARNING: Slack alert failed: {exc}", file=sys.stderr)
    elif should_alert and not webhook:
        print("(SLACK_WEBHOOK_URL not set; skipping Slack alert)")
    return overall, drift


def cmd_run(args) -> int:
    thresholds = Thresholds.load(args.thresholds)
    prompt_path = Path(args.prompt) if args.prompt else resolve_active_prompt(args.prompts_dir)
    prompt = PromptConfig.from_yaml(prompt_path)
    dataset = GoldenDataset.load(args.dataset)
    for w in dataset.lint():
        print(f"dataset warning: {w}", file=sys.stderr)

    clf_kind = args.provider or prompt.model.provider
    judge_kind = args.provider or thresholds.judge.provider
    if "openai" in (clf_kind, judge_kind) and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (use --provider mock for an offline run).", file=sys.stderr)
        return EXIT_ERROR
    rc = thresholds.runner
    classifier = build_provider(clf_kind, "classifier", rc.timeout_s, rc.max_retries)
    judge = build_provider(judge_kind, "judge", rc.timeout_s, rc.max_retries)

    model_label = prompt.model.name if clf_kind != "mock" else f"mock:{prompt.model.name}"
    judge_label = thresholds.judge.model if judge_kind != "mock" else f"mock:{thresholds.judge.model}"
    print(f"Evaluating prompt {prompt.version} ({prompt_path}) on {model_label}, "
          f"{len(dataset.cases)} cases, concurrency {rc.concurrency}", file=sys.stderr)
    results = asyncio.run(run_eval(prompt, dataset, classifier, judge, thresholds, stderr_progress))
    metrics = compute_metrics(results)

    store = Store(args.db)
    run_id = store.save_run(prompt_version=prompt.version, prompt_hash=prompt.content_hash,
                            model=model_label, judge_model=judge_label,
                            dataset_version=dataset.version, dataset_hash=dataset.content_hash,
                            track=args.track, git_sha=_git_sha(), metrics=metrics, results=results)
    run = store.get_run(run_id)
    assert run is not None

    if args.baseline == "auto":
        baseline = store.latest_baseline(exclude_id=run_id) or store.latest_run(args.track, exclude_id=run_id)
    elif args.baseline == "none":
        baseline = None
    else:
        baseline = store.get_run(int(args.baseline))
        if baseline is None:
            print(f"baseline run {args.baseline} not found", file=sys.stderr)
            return EXIT_ERROR
    comparison = compare_runs(results, store.results(baseline.id) if baseline else None, thresholds)
    overall, _ = _outputs(args, store, run, baseline, comparison, dataset, thresholds)

    if args.mark_baseline:
        store.mark_baseline(run_id)
        print(f"Run #{run_id} marked as the new baseline.")
    fail_levels = {"critical": {"critical"}, "warn": {"warn", "critical"}, "never": set()}[args.fail_on]
    return EXIT_REGRESSION if overall in fail_levels else EXIT_OK


def cmd_report(args) -> int:
    thresholds = Thresholds.load(args.thresholds)
    store = Store(args.db)
    run = store.get_run(args.run_id)
    baseline = store.get_run(args.baseline_id) if args.baseline_id else store.latest_baseline(exclude_id=args.run_id)
    if run is None:
        print(f"run {args.run_id} not found", file=sys.stderr)
        return EXIT_ERROR
    comparison = compare_runs(store.results(run.id), store.results(baseline.id) if baseline else None, thresholds)
    _outputs(args, store, run, baseline, comparison, GoldenDataset.load(args.dataset), thresholds)
    return EXIT_OK


def cmd_history(args) -> int:
    store = Store(args.db)
    rows = store.history(track=args.track, limit=args.limit)
    if not rows:
        print("no runs yet")
        return EXIT_OK
    print(f"{'id':>4}  {'created_at':<25} {'track':<12} {'prompt':<8} {'model':<22} "
          f"{'pass':>6} {'acc':>6} {'judge':>5}  status")
    for r in rows:
        m = r.metrics
        print(f"{r.id:>4}  {r.created_at:<25} {r.track[:12]:<12} {r.prompt_version:<8} {r.model[:22]:<22} "
              f"{m.pass_rate:>6.1%} {m.category_accuracy:>6.1%} "
              f"{(m.judge_mean or 0):>5.2f}  {r.status or '-'}{'  [baseline]' if r.is_baseline else ''}")
    return EXIT_OK


def cmd_validate(args) -> int:
    ok = True
    try:
        ds = GoldenDataset.load(args.dataset)
        print(f"dataset {ds.dataset} v{ds.version}: {len(ds.cases)} cases, hash {ds.content_hash}")
        for w in ds.lint():
            print(f"  warning: {w}")
    except Exception as exc:
        ok = False
        print(f"dataset INVALID: {exc}")
    prompts_dir = Path(args.prompts_dir)
    for p in sorted(prompts_dir.glob("v*.yaml")):
        try:
            cfg = PromptConfig.from_yaml(p)
            print(f"prompt {p.name}: ok ({cfg.model.provider}:{cfg.model.name}, hash {cfg.content_hash})")
        except Exception as exc:
            ok = False
            print(f"prompt {p.name}: INVALID: {exc}")
    try:
        active = resolve_active_prompt(prompts_dir)
        if not active.exists():
            raise FileNotFoundError(active)
        print(f"active prompt: {active.name}")
    except Exception as exc:
        ok = False
        print(f"active.yaml INVALID: {exc}")
    try:
        Thresholds.load(args.thresholds)
        print(f"thresholds {args.thresholds}: ok")
    except Exception as exc:
        ok = False
        print(f"thresholds INVALID: {exc}")
    return EXIT_OK if ok else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-regress", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=os.environ.get("LLMREG_DB", ".regress/history.db"))
    common.add_argument("--dataset", default=os.environ.get("LLMREG_DATASET", "golden/support_emails.json"))
    common.add_argument("--thresholds", default=os.environ.get("LLMREG_THRESHOLDS", "config/thresholds.yaml"))
    common.add_argument("--prompts-dir", default="prompts")
    out = argparse.ArgumentParser(add_help=False)
    out.add_argument("--out-dir", default=os.environ.get("LLMREG_OUT_DIR", "out"))
    out.add_argument("--report-url", default=None, help="link used in Slack/PR comment")
    out.add_argument("--slack", choices=["alert", "always", "never"], default=os.environ.get("LLMREG_SLACK", "alert"))
    out.add_argument("--drift-track", default=None,
                     help="track whose history feeds drift + trend (default: this run's --track)")
    out.add_argument("--trend-runs", type=int, default=20)
    out.add_argument("--require-baseline", action="store_true", help="warn if no baseline exists")

    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", parents=[common, out], help="evaluate a prompt against the golden set")
    r.add_argument("--prompt", help="prompt YAML (default: prompts/active.yaml target)")
    r.add_argument("--provider", choices=["openai", "mock"], default=os.environ.get("LLMREG_PROVIDER") or None,
                   help="override provider for feature and judge (env: LLMREG_PROVIDER)")
    r.add_argument("--track", default=os.environ.get("LLMREG_TRACK", "local"),
                   help="label for this run's lineage, e.g. main, pr-123, local")
    r.add_argument("--baseline", default="auto", help="'auto' (latest baseline), 'none', or a run id")
    r.add_argument("--mark-baseline", action="store_true", help="make this run the new baseline")
    r.add_argument("--fail-on", choices=["critical", "warn", "never"], default="critical")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", parents=[common, out], help="re-render the report for a stored run")
    rep.add_argument("run_id", type=int)
    rep.add_argument("--baseline-id", type=int)
    rep.set_defaults(func=cmd_report)

    h = sub.add_parser("history", parents=[common], help="list stored runs")
    h.add_argument("--track")
    h.add_argument("--limit", type=int, default=30)
    h.set_defaults(func=cmd_history)

    v = sub.add_parser("validate", parents=[common], help="validate dataset, prompts and thresholds")
    v.set_defaults(func=cmd_validate)
    return p


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines; real environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return EXIT_ERROR
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
