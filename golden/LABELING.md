# Golden dataset labeling guide

The golden set is the eval bar. If a label is wrong, the pipeline will
confidently block good prompts and pass bad ones. Treat edits here like edits
to a test suite: reviewed in a PR, with a reason.

## Categories

| Category    | Use when the customer's main need is about...                                                   |
|-------------|--------------------------------------------------------------------------------------------------|
| `billing`   | charges, invoices, refunds, payment methods, plan/price changes for existing customers, tax/VAT, trials converting to paid, cancelling a subscription |
| `technical` | the product misbehaving: errors, crashes, bugs, wrong numbers, performance, integrations, API, import/export, email delivery |
| `account`   | identity and access: login, password, 2FA, SSO, lockouts, changing login email or owner, roles/seats, deleting an account or its data, suspicious logins |
| `general`   | everything else: feature requests, how-to with no malfunction, praise, complaints with no concrete issue, sales/prospect questions, compliance docs, empty or unclear messages |

## Tie-break rules (in order)

These are also in the production prompt. If you change one, change both, and
bump the dataset version.

1. **Money back or a disputed charge -> `billing`**, even if a bug caused it.
2. **Customer cannot get into their own account -> `account`**, even with an error code.
   Integration credentials (Zapier, API tokens) are not "their own account".
3. **Something is broken -> `technical`.**
4. **Cancel subscription -> `billing`. Delete account or data -> `account`.**
5. Otherwise **`general`**.

## Summaries

- One sentence, at most 30 words, English.
- The customer's core problem/request plus the single most important specific
  (amount, error text, version, deadline).
- No invented details. For near-empty emails, say that the email has no detail.

The LLM judge compares candidate summaries against these, so write the summary
you would want a support agent to read in a queue, not a perfect paraphrase.

## Difficulty

- `easy` - a competent human labels it in under 5 seconds.
- `medium` - needs the category definitions, or has noise (typos, language).
- `hard` - needs a tie-break rule, is adversarial (sarcasm, injection), or has
  almost no content.

## Tags

`ambiguous`, `short`, `typos`, `mixed_language`, `sarcastic`, `multi_issue`,
`long`, `prompt_injection`. Use as many as apply.

## Adding a case

1. Write the email yourself or adapt a real (anonymized) ticket. Do not use an
   LLM to write emails or labels: the point is human-verified ground truth.
2. Pick the next free ID (`GS-061`, ...). IDs are permanent; never reuse one,
   even after a case is deleted - run history is keyed on them.
3. Fill `notes` with why the case exists. "Another billing case" is not a reason.
4. Set `verified: true` only after a second person agrees with the label.
5. Bump `version` in the file (minor for additions, major for label changes)
   and add a `changelog` entry.
6. `llm-regress validate` must pass.

## Changing a label

Label changes move the bar for every past run. Bump the major version and say
why in the changelog. The comparison stores the expected label with every
result, so a case whose label differs between baseline and current run is
reported as "relabeled" and excluded from the regression/improvement lists and
the deltas, instead of showing up as a fake flip.
