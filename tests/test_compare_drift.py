from llm_regress.compare import compare_runs, mcnemar_exact_p
from llm_regress.config import DriftConfig, Thresholds
from llm_regress.drift import detect_drift
from llm_regress.metrics import compute_metrics
from llm_regress.runner import CaseResult

CATS = ["billing", "technical", "account", "general"]


def make(n: int, failing: set[int], relabel: dict[int, str] | None = None) -> list[CaseResult]:
    out = []
    for i in range(n):
        cat = (relabel or {}).get(i, CATS[i % 4])
        ok = i not in failing
        out.append(CaseResult(
            case_id=f"GS-{i:03d}", expected_category=cat, difficulty="easy",
            predicted_category=cat if ok else "general" if cat != "general" else "billing",
            summary="s", category_match=ok, judge_score=4, passed=ok,
            latency_ms=100, prompt_tokens=10, completion_tokens=5,
        ))
    return out


T = Thresholds()


def test_no_baseline_passes_with_note():
    c = compare_runs(make(10, set()), None, T)
    assert not c.has_baseline and c.status == "pass" and c.notes


def test_identical_runs_pass():
    c = compare_runs(make(80, {1}), make(80, {1}), T)
    assert c.status == "pass" and not c.regressions and c.still_failing == ["GS-001"]


def test_small_drop_warns_large_drop_critical():
    base = make(100, set())
    warn = compare_runs(make(100, {0, 1, 2, 3}), base, T)  # -4pp overall
    assert warn.status in ("warn", "critical")
    assert next(ch for ch in warn.checks if ch.name == "Pass rate").severity == "warn"
    crit = compare_runs(make(100, set(range(0, 40, 4))), base, T)  # -10pp, all billing
    assert crit.status == "critical"
    assert len(crit.regressions) == 10


def test_improvements_detected():
    c = compare_runs(make(40, set()), make(40, {0, 5}), T)
    assert {f.case_id for f in c.improvements} == {"GS-000", "GS-005"}
    assert c.status == "pass"


def test_dataset_change_uses_shared_cases_only():
    c = compare_runs(make(44, set()), make(40, set()), T)
    assert c.added_cases == ["GS-040", "GS-041", "GS-042", "GS-043"]
    assert c.n_compared == 40 and c.status == "pass"


def test_relabeled_case_is_not_a_flip():
    base = make(40, set())
    cur = make(40, {2}, relabel={2: "billing"})  # GS-002 relabeled and now "fails"
    c = compare_runs(cur, base, T)
    assert c.relabeled == ["GS-002"] and not c.regressions


def test_mcnemar():
    assert mcnemar_exact_p(0, 0) == 1.0
    assert abs(mcnemar_exact_p(6, 0) - 0.03125) < 1e-9
    assert mcnemar_exact_p(5, 0) > 0.05
    assert mcnemar_exact_p(3, 3) == 1.0


def test_significance_gate_downgrades():
    t = Thresholds.model_validate({"significance": {"gate": True}})
    base = make(100, set())
    c = compare_runs(make(100, {0, 1, 2, 3}), base, t)  # 4 flips, p=0.125
    assert next(ch for ch in c.checks if ch.name == "Pass rate").severity == "pass"


def test_metrics():
    m = compute_metrics(make(8, {0}))
    assert m.pass_rate == 7 / 8 and m.per_category["billing"].accuracy == 0.5


def test_drift_floor_and_drop():
    cfg = DriftConfig(window=3, floor=0.8, max_drop=3.0)
    assert not detect_drift([0.9, 0.9], cfg).triggered  # not enough points
    assert not detect_drift([0.9] * 6, cfg).triggered
    slow = [0.92, 0.91, 0.90, 0.89, 0.88, 0.87]  # each step -1pp, never a 3pp per-run drop
    r = detect_drift(slow, cfg)
    assert r.triggered and "fell" in r.reasons[0]
    assert detect_drift([0.79, 0.78, 0.77], cfg).triggered


def test_threshold_env_override():
    t = Thresholds.load(None, env={"LLMREG_OVERALL_CRITICAL": "12", "LLMREG_JUDGE_MODEL": "x"})
    assert t.overall.critical == 12.0 and t.judge.model == "x"
