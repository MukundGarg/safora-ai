# Repository reconciliation — 14 November 2026

`origin/main` fetched at `72b2bee5eca331e542cb9ad362b1b9be3a5048bb` contained
Task 1 and the old Android demo, not Tasks 2–4. A merged badge is relative to each
PR's **base**, not evidence that its changes have propagated to main.

Verified git parents, not inferred from the badge:

| Commit / PR | First parent/base | Other parent/head | Where changes landed |
|---|---|---|---|
|72b2bee / #1|a2b78cb main|6ee9845 Task 1|main|
|7286637 / #2|6ee9845 Task 1|bf72c85 Task 2|feat/task1-scream-streaming|
|a8e5a79 / #3|bf72c85 Task 2|d53d369 Task 3|feat/task2-offline-asr|
|9043267 / #4|d53d369 Task 3|87c1f12 Task 4|feat/task3-conversation-risk|

Those earlier feature branches were never merged back into main after their later
stacked PRs landed. The merge base of main and Task 4 was6ee9845. Task 1 already
existed on main; replaying all four commits would duplicate Task 1 unnecessarily.

New branch `feat/reconcile-production-gates` starts at current main. Recovery merge
`055dc672592466054ab87fe64808d49c8281b444` has parents72b2bee and87c1f12,
with exact Task 4 tree `3ff2e28e6b434f309dddd4547ff7bd387a69c93f`. The merge
was conflict-free. Existing Task 2–4 commit identities/ancestry were preserved,
not cherry-picked or rewritten. Main was not modified or force-pushed. A later
hardening commit on this branch contains new work; the PR targets main and is not
automatically merged. Changes are restored to **main lineage on the feature
branch**, not claimed already present on main before PR review/merge.

The exact79restored changed paths, including their A/M status, are recorded in
[restored-paths.txt](verification/production/restored-paths.txt). Key recovered paths:

- Kotlin AsrPcm, hardened OfflineSpeech, MlController, MonitoringRules, independent
  AudioPipeline delegation, ModelAssets/Engine hooks and native/controller tests.
- Python ASR assets/offline/streaming, classifier text/bundle/risk, strict risk data/
  trainer/evaluation, coordinator and legacy ASR-uncertainty compatibility.
- ASR/risk evaluators and preparation/export/smoke scripts; shared rule/config data.
- Task 2/3 reports, contracts, training-data/taxonomy protocols, final Member 2
  handoff/test report and historical verification logs.

After reconciliation, before hardening, the exact recovered tree was rerun:
**147 Python passed,0failed/skipped;22Kotlin JVM passed,0failed/skipped**.
See [Python](verification/production/reconciled-python.txt) and
[Kotlin](verification/production/reconciled-kotlin.txt). Real local YAMNet/Whisper/
Vosk EN/HI and the previous APK were supplied for optional regressions. These
counts were measured again, not copied from the Task 4 report. New final counts
and upgraded-artifact checks are separately reported in production-readiness.md.
