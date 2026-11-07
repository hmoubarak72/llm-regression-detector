# llm-regress: regression gate for the support email classifier

`llm-regress` runs our LLM-powered support email classifier against a
human-labeled golden dataset whenever a prompt, model setting, threshold or the
dataset itself changes. It scores every case on category correctness, summary
quality (LLM-as-judge), latency and token use, diffs the run against the
current production baseline, and reports the result three ways: a sticky
comment on the PR, an HTML diff report, and a Slack alert. A critical
regression fails the `prompt-eval` check and blocks the merge. A nightly run on
`main` tracks a moving average so slow degradation (including provider-side
model changes nobody on our side made) gets flagged even when no single run
crosses a threshold.

> **Before this gates anything:** all 60 golden cases ship with
> `verified: false`. They were drafted to cover the edge cases we care about,
> but the dataset is only ground truth once a person has checked each label and
> reference summary against [golden/LABELING.md](golden/LABELING.md). `llm-regress validate`
> warns until that's done.

---

## Contents

- [Setup](#setup)
- [Day to day: shipping a prompt change](#day-to-day-shipping-a-prompt-change)
- [Reading the results](#reading-the-results)
- [Adding test cases](#adding-test-cases)
- [Adjusting thresholds](#adjusting-thresholds)
- [CI setup](#ci-setup)
- [Docker](#docker)
- [How it works](#how-it-works)
- [Architecture decisions](#architecture-decisions)
- [Known limitations](#known-limitations)
- [Why slow drift is tracked separately](#why-slow-drift-is-tracked-separately)

---

## Setup

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
llm-regress validate
```

Offline run (no API key, deterministic mock model; checks the plumbing, not quality):

```bash
llm-regress run --provider mock --track main --mark-baseline
LLMREG_MOCK_ERROR_RATE=0.15 llm-regress run --provider mock --prompt prompts/v1.1.0.yaml --track pr-local --drift-track main
```

The mock is a keyword matcher with a pass rate around 60%, below the default
drift floor of 85%. After 7 mock runs on `main` you will see a slow-drift
warning; for demos, set `LLMREG_DRIFT_FLOOR=0.5`.

Real run:

```bash
export OPENAI_API_KEY=sk-...
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...   # optional
llm-regress run --track main --mark-baseline        # first time: create the baseline
llm-regress run --prompt prompts/v1.1.0.yaml        # compare a candidate prompt
```

Outputs land in `out/` (`report.html`, `summary.md`, `summary.json`); run
history is in `.regress/history.db`. `llm-regress history` lists past runs.

A full run is 60 classifier calls on `gpt-4o-mini` plus 60 judge calls on
`gpt-4o`, at concurrency 8. Expect well under a minute and a few cents.

## Day to day: shipping a prompt change

Prompts live in `prompts/` as immutable, versioned YAML. `prompts/active.yaml`
points at the one in production.

1. Copy the current version to a new file, e.g. `prompts/v1.2.0.yaml`, and bump
   `version` and `created_at`. The file name must match the version; `validate`
   enforces it. Do not edit an existing version file: run history refers to
   versions by name, and editing one makes old runs unreproducible.
2. Make your change. Model choice and parameters are in the same file, so a
   model swap is a prompt change as far as CI is concerned.
3. Point `prompts/active.yaml` at the new file.
4. Run it locally against the baseline: `llm-regress run --prompt prompts/v1.2.0.yaml`.
5. Open a PR. CI reruns the eval, posts the comparison as a PR comment, and
   fails the check if anything is critical.
6. On merge, the `main` run becomes the new baseline.

**If the check fails:** open the report artifact from the PR comment link and
start with the regressions table. It shows the old and new output side by side
for every case that flipped from pass to fail. Typical outcomes:

- The prompt really is worse on those cases. Fix the prompt.
- The new behavior is right and the label is wrong. Fix the label in a
  separate PR first (see [Changing a label](golden/LABELING.md#changing-a-label)),
  then rebase.
- The trade-off is intentional (e.g. cheaper prompt, small accuracy hit). Get a
  second reviewer to agree, then merge with an admin override and say why in the PR.
  Don't loosen thresholds to make one PR pass.

## Reading the results

A case **passes** when the predicted category exactly matches the label **and**
the judge scores the summary >= `pass_criteria.min_judge_score` (default 3/5).

Each run is compared to the latest baseline (the last `main` run) on the cases
both runs share. The report and PR comment show:

| Check | Default warn / critical | Notes |
|---|---|---|
| Pass rate | -3pp / -8pp | headline number |
| Category accuracy | -3pp / -8pp | category only, ignores summaries |
| Per-category accuracy | -10pp / -20pp | ~15 cases each, so one flip is ~6.7pp |
| Judge mean | -0.2 / -0.5 | on the 1-5 scale |
| Errors | any increase = warn | schema violations, timeouts |
| Latency p95, tokens | never | informational |

The **McNemar p-value** next to the flip count answers "is 5 regressions vs 3
improvements more than noise?". It is shown, not enforced, by default (see
[decisions](#architecture-decisions)).

**Overall status** is the worst check: `PASS`, `WARN` (alert, don't block) or
`FAIL` (critical, blocks merge, exit code 1). Exit code 2 means the pipeline
itself broke (bad config, missing key).

## Adding test cases

Full guide: [golden/LABELING.md](golden/LABELING.md). Short version:

- Write the email by hand or adapt a real anonymized ticket. No LLM-written
  cases or labels.
- Take the next ID (`GS-061`). IDs are permanent.
- Fill `expected.category`, `expected.summary` (<=30 words), `expected_difficulty`,
  `tags`, and a `notes` line saying why the case exists.
- `verified: true` only after a second person agrees with the label.
- Bump the dataset `version` and add a `changelog` line. Run `llm-regress validate`.

The best source of new cases is production misclassifications: every bug report
about a wrong category should become a golden case before it's fixed.

## Adjusting thresholds

`config/thresholds.yaml` is the source of truth and is commented inline. Any key
can be overridden per environment with `LLMREG_<SECTION>_<KEY>`:

```bash
LLMREG_OVERALL_CRITICAL=10 LLMREG_JUDGE_MODEL=gpt-4o llm-regress run
```

Rules of thumb:

- Threshold changes go through a PR like code, with the reason in the description.
- Keep per-category thresholds above one case's worth of accuracy (100 / cases-in-category),
  or a single flip goes straight to FAIL.
- Tighten thresholds as the dataset grows. At 200+ cases, -2pp / -5pp is reasonable
  and turning on `significance.gate` starts to make sense.
- Changing `judge.model` changes the ruler. Expect a one-off shift in judge
  scores; mark a fresh baseline right after.

## CI setup

The workflow is [.github/workflows/prompt-eval.yml](.github/workflows/prompt-eval.yml).

1. Repo secrets: `OPENAI_API_KEY`, `SLACK_WEBHOOK_URL` (optional).
   To run without a key, set the repository variable `LLMREG_PROVIDER=mock`.
2. Branch protection on `main`: require a pull request and the `eval` status
   check. Admins can still bypass for an intentional, reviewed trade-off.
3. Trigger the workflow once on `main` (Actions -> prompt-eval -> Run workflow)
   to create the first baseline.

| Trigger | Track | Marks baseline | Saves history |
|---|---|---|---|
| Any PR (eval runs only if `prompts/`, `golden/`, `config/thresholds.yaml`, `src/` or the workflow changed; otherwise `eval` is skipped, which counts as passing) | `pr-<n>` | no | no |
| Push to `main` (same paths) | `main` | yes | yes |
| Nightly 05:00 UTC | `main` | yes | yes |

Run history lives in the GitHub Actions cache. PRs read `main`'s history but
never write to it. If the cache is evicted (7 days unused; the nightly run keeps
it warm) the next run finds no baseline and reports `WARN` with an explicit
"baseline required" note rather than silently passing.

PRs from forks don't get secrets, so the eval exits with code 2 there. That's
intended: prompt changes come from branches in this repo.

The Slack message links to the Actions run page, where the HTML report is an
artifact. Set `LLMREG_REPORT_URL` if you publish reports somewhere browsable.

## Docker

```bash
docker build -t llm-regress .
docker run --rm \
  -e OPENAI_API_KEY -e SLACK_WEBHOOK_URL \
  -e LLMREG_OVERALL_CRITICAL=8 \
  -v "$PWD/.regress:/app/.regress" -v "$PWD/out:/app/out" \
  llm-regress run --track main
```

The image contains the runner, prompts, golden dataset and thresholds.
Mount `/app/prompts` or `/app/golden` to evaluate a different checkout without
rebuilding. Use `-e LLMREG_PROVIDER=mock` for an offline smoke test.

## How it works

```
prompts/vX.yaml ─┐
                 ├─> runner (async, semaphore-bounded) ──> classifier call ──> judge call
golden/*.json ───┘          │                                (gpt-4o-mini)     (gpt-4o)
                            v
                      CaseResult x N ──> metrics ──> SQLite (.regress/history.db)
                                                          │
                     baseline run (latest is_baseline) ───┤
                                                          v
                         compare: deltas on shared cases, flips, McNemar, severity
                         drift:   moving average over the `main` track
                                                          │
                 ┌───────────────────────┬────────────────┴─────────┬──────────────┐
                 v                       v                          v              v
          out/report.html          out/summary.md            Slack webhook     exit code
          (diff + trend)           (PR comment)              (warn/fail)       (gates merge)
```

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic contracts: `PromptConfig`, `ClassificationResult`, `Thresholds` |
| `feature.py` | The classifier itself. Production code and the eval call the same function. |
| `providers.py` | OpenAI client (structured outputs, SDK retries) and the offline mock |
| `judge.py` | LLM-as-judge rubric for summary relevance |
| `runner.py` | Concurrent execution, per-case scoring |
| `metrics.py` / `compare.py` / `drift.py` | Aggregation, diffing, significance, slow drift |
| `storage.py` | SQLite run + result history |
| `report.py` / `slack.py` | HTML report, Markdown PR summary, Slack blocks |
| `cli.py` | `run`, `report`, `history`, `validate` |

## Architecture decisions

**Diff against a baseline, not against a fixed bar.** An absolute "must be 90%"
gate either blocks every PR (if set high) or lets real regressions through (if
set low). The useful question for a PR is "did this make things worse than
what's in production?". The flips table answers it at the level of individual
cases, which is what a reviewer can act on.

**Deltas only on shared cases, relabels excluded.** When the dataset changes
between baseline and current run, comparing pass rates over different case sets
is meaningless. The comparison intersects case IDs and drops cases whose label
changed, and says so in the report.

**Model settings live in the prompt file.** A model swap is the riskiest change
we make and the easiest to forget to test. Keeping `model:` in the versioned
prompt YAML means it goes through the same path filter, PR and gate.

**Strict structured outputs, and schema violations count as failures.** We use
OpenAI's `json_schema` response format so the enum of categories is enforced
by the API. If output still fails Pydantic validation, the case fails with the
error recorded instead of crashing the run. A prompt that breaks the output
contract is a regression.

**A stronger, pinned judge.** The feature runs on `gpt-4o-mini`; the judge runs
on `gpt-4o` with temperature 0, a reason-before-score rubric anchored on
concrete failure modes, and access to the human reference summary. The judge is
configured separately from prompts, so a prompt PR can't move the ruler.
We wrote the judge directly instead of pulling in RAGAS or DeepEval: the rubric
is ~20 lines, our metric is narrow (summary faithfulness against a reference),
and those frameworks bring large dependency trees into a container that
otherwise needs three packages. If we add RAG-style features, DeepEval's G-Eval
is the first thing to revisit.

**Significance shown, not enforced (yet).** With 60 cases, an exact McNemar
test needs roughly 6+ net flips before p < 0.05, so gating on it would wave
through a real 5pp regression. Fixed pp thresholds are blunt but predictable.
`significance.gate: true` is there for when the dataset is big enough.

**SQLite in the Actions cache instead of a database.** Zero infrastructure and
the whole history is one file you can copy and open locally. The cost is that
cache eviction loses history; we mitigate with the nightly run and an explicit
"no baseline" warning. If history becomes something we report on, move it to
a real DB; `storage.py` is the only module that would change.

**A mock provider in the main code path.** It lets the whole pipeline (CI
wiring, report, Slack formatting, gating) be tested without network or cost,
and it's what the unit tests use. Its numbers mean nothing about model quality
and are labeled `mock:` everywhere they appear.

## Known limitations

- 60 cases is a starting point. Per-category numbers move ~6.7pp per flip.
- The judge is itself an LLM. Spot-check its reasons in the report; if it
  disagrees with you on a pattern of cases, fix the rubric in `judge.py` and
  re-baseline.
- Temperature 0 isn't fully deterministic on OpenAI models. Expect 0-2 flips
  between identical runs; that's what the warn band absorbs.
- Only one feature (the classifier) is wired in. Adding another means a new
  feature module, dataset and output schema; the runner, compare, drift and
  reporting layers are feature-agnostic except for the category breakdown.

## Why slow drift is tracked separately

Most teams that test prompt changes at all test them per-change: new prompt vs
old prompt, pass or fail. That catches the big mistakes. It misses two things.

The first is a regression that arrives in slices. Five PRs that each cost 1.5pp
all pass a 3pp gate, and a month later the classifier is 7pp worse with no
failing check anywhere in the history. Each run was compared to a baseline that
had already absorbed the previous slice.

The second is change nobody on the team made. Model aliases like `gpt-4o-mini`
get updated by the provider. No PR, no diff, no CI run. Behavior shifts anyway.

So drift is its own signal: the nightly run re-evaluates `main` with no code
change, and the pipeline keeps a 7-run moving average of pass rate. It fires
when that average drops below an absolute floor or falls more than 3pp below
the previous seven runs, regardless of whether any single run alerted. Drift
warns in Slack but never blocks a merge on its own, because the PR in front of
you is usually not the cause.
