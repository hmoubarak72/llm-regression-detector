---
name: llm-regress
description: Run and interpret the llm-regress regression gate for the LLM support-email classifier (repo hmoubarak72/llm-regression-detector). Use whenever the user wants to evaluate a prompt or model change against the golden dataset, compare a candidate prompt to the production baseline, read or explain a regression report (PASS/WARN/FAIL, flips, McNemar p-value, slow drift), add or relabel golden test cases, tune thresholds, check run history, or debug the prompt-eval CI check. Triggers on "llm-regress", "prompt eval", "golden dataset", "regression report", "baseline run", "prompt drift", "support email classifier", "thresholds.yaml", or a new prompts/vX.Y.Z.yaml file.
---

# llm-regress: prompt regression gate

`llm-regress` runs the support email classifier against a human-labeled golden
dataset (`golden/support_emails.json`, 60 cases). It scores each case, compares
the run with the production baseline, and writes an HTML report, a Markdown
summary for the PR, and an optional Slack alert. A critical regression returns
exit code 1 and blocks the merge.

## Repo layout you will touch

| Path | What it is |
|---|---|
| `prompts/vX.Y.Z.yaml` | Prompt versions. They can't be changed once created and include the model name and parameters. |
| `prompts/active.yaml` | `active: vX.Y.Z.yaml`, the prompt currently in production |
| `golden/support_emails.json` | Golden cases (`GS-001`...), with a dataset `version` and `changelog` |
| `golden/LABELING.md` | Rules for writing and relabeling cases. Read it before you edit the dataset. |
| `config/thresholds.yaml` | Pass criteria, warn/critical bands, drift, runner, judge model |
| `.regress/history.db` | SQLite history of runs and results |
| `out/` | `report.html`, `summary.md`, `summary.json` from the latest run |
| `src/llm_regress/` | Source code. `cli.py` is the entry point, `judge.py` has the rubric, `feature.py` has the classifier. |

## Setup (once per checkout)

Python 3.11+. Run these from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
llm-regress validate
```

Real runs need `OPENAI_API_KEY`, set in the environment or in `.env`.
`SLACK_WEBHOOK_URL` is optional. Never print, commit or echo these values. If
there is no key, use `--provider mock` and tell the user that mock numbers say
nothing about model quality. The mock is a keyword matcher that passes about
60% of cases.

## Commands

```bash
llm-regress validate                                  # dataset, all prompts, active.yaml, thresholds
llm-regress run                                       # evaluate the active prompt vs the latest baseline
llm-regress run --prompt prompts/v1.2.0.yaml          # evaluate a candidate prompt
llm-regress run --track main --mark-baseline          # create/refresh the baseline (only for main/production)
llm-regress run --provider mock                       # offline plumbing check, no API calls
llm-regress run --baseline 12                         # compare against a specific run id (or 'none')
llm-regress run --fail-on warn                        # stricter exit code (critical|warn|never)
llm-regress run --slack never                         # suppress Slack (alert|always|never)
llm-regress history [--track main] [--limit 30]       # list stored runs
llm-regress report <run_id> [--baseline-id N]         # re-render report for a stored run
```

Other `run` flags: `--track <label>` (default `local`; use `main` or `pr-<n>`),
`--drift-track`, `--require-baseline`, `--out-dir`, `--db`, `--dataset`,
`--thresholds`, `--prompts-dir`, `--report-url`.

A real run makes 60 classifier calls (gpt-4o-mini) and 60 judge calls (gpt-4o).
It takes under a minute and costs a few cents. Tell the user before you start
one.

**Exit codes:** `0` means pass, or WARN with the default `--fail-on critical`.
`1` means a regression at the fail level. `2` means the pipeline itself broke:
a bad config, a missing key, or at least 50% of cases erroring. A run with
exit 2 is never scored and never becomes the baseline.

## Workflow: evaluating a prompt or model change

1. Never edit an existing `prompts/v*.yaml`. Copy the current version to a new
   file such as `prompts/v1.2.0.yaml` and bump `version` and `created_at`. The
   file name must match the version. A model swap counts as a prompt change and
   goes in the same file.
2. Make the change, then point `prompts/active.yaml` at the new file.
3. `llm-regress validate`
4. `llm-regress run --prompt prompts/v1.2.0.yaml`
5. Read `out/summary.json` and `out/summary.md`, then report back as described
   below. Open `out/report.html` for case-level detail.
6. Do not use `--mark-baseline` for a local or PR run. The `main` run in CI
   becomes the baseline after the merge.

## Interpreting results

A case **passes** when the predicted category matches the label exactly **and**
the judge scores the summary at least `pass_criteria.min_judge_score` (default
3 out of 5).

The comparison only uses cases that both runs share and leaves out relabeled
cases. These are the default bands, as drops against the baseline:

| Check | Warn / critical |
|---|---|
| Pass rate | -3pp / -8pp |
| Category accuracy | -3pp / -8pp |
| Per-category accuracy | -10pp / -20pp (one flip is about 6.7pp) |
| Judge mean (1-5) | -0.2 / -0.5 |
| Errors | any increase gives a warn |
| Latency p95, tokens | shown for information only, never gate |

- The overall status is the worst check: `PASS`, `WARN` (alerts but doesn't
  block) or `FAIL` (blocks the merge).
- The McNemar p-value next to the flip count is informational. With 60 cases,
  p < 0.05 needs about 6 net flips. Don't call a FAIL "noise" because p is high.
- **Slow drift** is a 7-run moving average of pass rate on the `main` track.
  It fires below the floor (0.85) or more than 3pp under the previous window.
  It only warns and never blocks. It usually means the provider changed the
  model, not that the current PR broke something.
- Temperature 0 isn't fully deterministic. 0 to 2 flips between identical runs
  is normal.
- Mock runs are labeled `mock:`. Never present their numbers as quality.

When you report a run, give: the status, the headline, the pass-rate and
accuracy deltas, the categories that crossed a band, the IDs of the cases that
regressed (from `summary.json`) along with the old and new outputs from the
report, any drift, and the path to the report.

## When the check fails

Start with the regressions table in `report.html`. Then decide which of these
applies:

- **The prompt is worse.** Fix the prompt in a new version file.
- **The label is wrong.** Fix it in a separate PR first, following
  `golden/LABELING.md` ("Changing a label"), then rebase.
- **An intentional trade-off.** A second reviewer must agree. Then the user
  merges with an admin override and explains why in the PR.

Never loosen thresholds to make one PR pass.

## Adding or changing golden cases

- Cases are written by hand or adapted from anonymized real tickets. **Never
  write a case or a label with an LLM.** You may help format a case or check
  its fields, but the email text and the expected category and summary must
  come from the user.
- Take the next permanent ID (`GS-061`...). Fill in `expected.category`,
  `expected.summary` (30 words or fewer), `expected_difficulty`, `tags`, and a
  `notes` line saying why the case exists.
- Set `verified: true` only after a second person has agreed with the label.
  All 60 seed cases currently ship as `verified: false`, and `validate` warns
  until they are reviewed.
- Bump the dataset `version`, add a `changelog` line, then run
  `llm-regress validate`.

## Thresholds and overrides

`config/thresholds.yaml` is the source of truth. Any key can be overridden
with `LLMREG_<SECTION>_<KEY>`, for example `LLMREG_OVERALL_CRITICAL=10`,
`LLMREG_JUDGE_MODEL=gpt-4o` or `LLMREG_DRIFT_FLOOR=0.5` (useful for mock
demos). Change the file itself only through a PR that states the reason. Keep
per-category thresholds above `100 / cases-in-category`. Changing
`judge.model` shifts every judge score, so mark a fresh baseline right after.

## CI notes

The workflow is `.github/workflows/prompt-eval.yml`, and the status check is
named `eval`. It runs on PRs that touch `prompts/`, `golden/`,
`config/thresholds.yaml`, `src/` or the workflow file, on pushes to `main`
(which mark the baseline), and nightly at 05:00 UTC. History is kept in the
Actions cache. If the cache is evicted, the next run reports WARN with
"baseline required". PRs from forks have no secrets and exit 2, which is
intended.
