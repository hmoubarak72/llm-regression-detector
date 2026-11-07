"""End-to-end with the mock provider: dataset, prompts, CLI, storage, report."""

import json
from pathlib import Path

import pytest

from llm_regress.cli import main
from llm_regress.compare import compare_runs
from llm_regress.config import DriftConfig, Thresholds
from llm_regress.drift import detect_drift
from llm_regress.config import PromptConfig, resolve_active_prompt
from llm_regress.dataset import GoldenDataset
from llm_regress.feature import build_messages, classify_email
from llm_regress.providers import MockProvider
from llm_regress.slack import build_payload
from llm_regress.storage import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    return ["--db", str(tmp_path / "h.db"), "--out-dir", str(tmp_path / "out"), "--provider", "mock"]


def test_golden_dataset_is_valid():
    ds = GoldenDataset.load(ROOT / "golden/support_emails.json")
    assert len(ds.cases) >= 50
    cats = {c.expected.category for c in ds.cases}
    assert cats == {"billing", "technical", "account", "general"}
    tags = {t for c in ds.cases for t in c.tags}
    assert {"ambiguous", "short", "typos", "mixed_language", "sarcastic"} <= tags
    assert all(len(c.expected.summary.split()) <= 30 for c in ds.cases)


def test_prompts_load_and_active_resolves():
    active = resolve_active_prompt(ROOT / "prompts")
    cfg = PromptConfig.from_yaml(active)
    msgs = build_messages(cfg, "hello")
    assert msgs[0]["role"] == "system" and msgs[-1]["content"] == "hello"
    assert len(msgs) == 2 + 2 * len(cfg.few_shot)


def test_prompt_filename_must_match_version(tmp_path):
    p = tmp_path / "v9.9.9.yaml"
    p.write_text((ROOT / "prompts/v1.0.0.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError):
        PromptConfig.from_yaml(p)


def test_sync_feature_entrypoint():
    cfg = PromptConfig.from_yaml(ROOT / "prompts/v1.0.0.yaml")
    out = classify_email("Subject: refund\nPlease refund my invoice", cfg, MockProvider("classifier", 0))
    assert out.category == "billing"


def test_cli_end_to_end(env, tmp_path, monkeypatch):
    assert main(["run", *env, "--track", "main", "--mark-baseline"]) == 0
    # same prompt again: identical mock output -> pass
    assert main(["run", *env, "--track", "main"]) == 0
    # degraded candidate on a PR track
    monkeypatch.setenv("LLMREG_MOCK_ERROR_RATE", "0.35")
    code = main(["run", *env, "--prompt", "prompts/v1.1.0.yaml", "--track", "pr-7", "--drift-track", "main"])
    summary = json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_run_id"] == 1
    assert summary["regressions"], "error injection should cause regressions"
    assert (summary["status"] == "critical") == (code == 1)
    html = (tmp_path / "out/report.html").read_text(encoding="utf-8")
    assert "Regressions" in html and summary["regressions"][0] in html
    md = (tmp_path / "out/summary.md").read_text(encoding="utf-8")
    assert "LLM eval" in md

    store = Store(tmp_path / "h.db")
    runs = store.history()
    assert [r.track for r in runs] == ["main", "main", "pr-7"]
    assert runs[0].is_baseline and not runs[2].is_baseline

    payload = build_payload(run=runs[2], baseline=runs[0],
                            comparison=compare_runs(store.results(3), store.results(1), Thresholds()),
                            drift=detect_drift([0.9], DriftConfig()),
                            overall_status=summary["status"], report_url="https://example.test/r")
    assert payload["blocks"][0]["type"] == "header"
    assert any(b["type"] == "actions" for b in payload["blocks"])

    assert main(["history", "--db", str(tmp_path / "h.db")]) == 0
    assert main(["report", "3", "--db", str(tmp_path / "h.db"), "--out-dir", str(tmp_path / "out2")]) == 0


def test_missing_key_is_pipeline_error(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLMREG_PROVIDER", raising=False)
    assert main(["run", "--db", str(tmp_path / "h.db"), "--out-dir", str(tmp_path / "o")]) == 2


def test_validate_cli(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert main(["validate"]) == 0


def test_dotenv_loader(tmp_path, monkeypatch):
    from llm_regress.cli import load_dotenv
    f = tmp_path / ".env"
    f.write_text('# c\nA_KEY="x1"\nexport B_KEY=y2\nEMPTY=\nC_KEY=from-file\n', encoding="utf-8")
    for k in ("A_KEY", "B_KEY", "EMPTY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("C_KEY", "from-env")
    load_dotenv(f)
    import os
    assert os.environ["A_KEY"] == "x1" and os.environ["B_KEY"] == "y2"
    assert "EMPTY" not in os.environ          # empty values never mask a missing key
    assert os.environ["C_KEY"] == "from-env"  # real environment wins


def test_mass_api_failure_is_pipeline_error_not_baseline(env, tmp_path, monkeypatch, capsys):
    import llm_regress.cli as cli

    class AuthFail:
        async def complete(self, *a, **k):
            raise RuntimeError("Error code: 401 - Incorrect API key provided: sk-proj-abcdefghijklmnop")

    assert main(["run", *env, "--track", "main", "--mark-baseline"]) == 0  # good baseline, run #1
    monkeypatch.setattr(cli, "build_provider", lambda *a, **k: AuthFail())
    code = main(["run", *env, "--track", "main", "--mark-baseline"])      # broken run #2
    err = capsys.readouterr().err
    assert code == 2
    assert "PIPELINE ERROR: 60/60" in err and "401" in err
    assert "sk-proj-abcdefghijklmnop" not in err and "sk-***" in err      # key-shaped text redacted
    store = Store(tmp_path / "h.db")
    broken = store.get_run(2)
    assert broken.status == "error" and not broken.is_baseline
    assert store.latest_baseline().id == 1


def test_latest_baseline_skips_mostly_errored_runs(env, tmp_path, monkeypatch):
    import llm_regress.cli as cli
    assert main(["run", *env, "--track", "main", "--mark-baseline"]) == 0
    store = Store(tmp_path / "h.db")
    # Simulate an old, already-cached broken baseline (written before this guard existed).
    good = store.get_run(1)
    bad = good.metrics.model_copy(update={"errors": good.metrics.n_cases, "pass_rate": 0.0})
    rid = store.save_run(prompt_version="1.0.0", prompt_hash="x", model="m", judge_model="j",
                         dataset_version="1", dataset_hash="h", track="main", git_sha=None,
                         metrics=bad, results=[])
    store.mark_baseline(rid)
    assert store.latest_baseline().id == 1
