# Deployed review fixes

Repository: https://github.com/rasingod/ai-it-support-assistant

Source baseline: f7ca7029fbad842d7caff5625d0443f87400fe5e. Fix branch: `fix/deployed-review-findings`. This identifies the source checked out for review, not a verified deployed commit. The public app has not been redeployed by this change.

| Review finding | Change |
|---|---|
| Cross-employee ticket disclosure | Require an employee for ID lookup and apply both filters before returning records. Keep the chosen fictional identity fixed until conversation reset; model output cannot switch it. |
| Cancel stored as a slot | Intercept cancel/reset before slot collection, clear pending work, and leave saved tickets intact. |
| Missing confirmation | Show normalized draft and require exact confirm of that retained draft before creation. Confirmation bypasses LLM classification. |
| Malformed categories | Shared category allowlist at graph/tool/storage boundaries; split category-plus-description follow-ups; preserve category-level duplicate policy. |
| Unsupported capabilities | Deterministic supported-capability help and tool-result rendering; no generated promises of ticket updates or notifications. |
| Status grounding and draft pollution | Display sample-data/freshness caveat and recorded incident timestamp. Status requests do not populate creation fields and do not reuse an earlier system query. |
| Identity, model, persistence disclosure | Sidebar distinguishes demo profile lookup from authentication, shows configured OpenRouter model, and explains JSON persistence limits. |

Additional safeguards clear transient results each turn, validate routing result field types, bound provider timeouts/retries, and suppress raw exception details in the UI. Existing data files, presentation, video, and keepalive workflows are unchanged.

## Validation

The regression suite uses temporary ticket files and mocked LLM routes, so it neither changes seed data nor incurs API charges. It covers employee isolation, missing and unknown employees, sticky identity, every cancellation stage, confirmation and duplicate reuse, category validation, KB results, status freshness/grounding, malformed provider responses, and Streamlit creation/reset/persisted-ticket lookup. Compilation and Git whitespace checks are also required by this change. The added GitHub workflow runs tests and compilation on pushes and pull requests.

## Remaining deployment boundaries

This is still a fictional-data demo, not production authentication. Local JSON is not a transactional multi-writer store and is not guaranteed durable across redeployment. No migration of historical malformed records is attempted; review any non-seed ticket data before deploying. Replacing category-level deduplication with issue-level matching would change existing product behavior and is intentionally outside this patch. No status integration was added: status remains sample data with unknown refresh time.

Record the deployed commit when promoting this branch and keep the actual API key in environment variables or Streamlit Secrets. Never commit `.env` or `.streamlit/secrets.toml`.

Validation on 15 September 2026: 18 tests passed with the repository's pinned dependencies (including Streamlit AppTest); pip check found no broken requirements; compilation and git diff --check passed. Two read-only live OpenRouter routing checks (VPN guidance and email status) also passed using the existing local configuration. This does not establish broad model accuracy or verify the deployed revision.
