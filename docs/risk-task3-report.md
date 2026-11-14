# Task 3 — offline conversation-risk classifier

## Result and branch

Implemented on `feat/task3-conversation-risk`, based on Task 2
`bf72c85b72e624c99932019aa4f7690ba21a8528`; the stacked PR targets
`feat/task2-offline-asr`. No PR is merged. Task 4 integration is not implemented.

**No licensed, independently labeled conversation-risk dataset or genuine
fine-tuned risk checkpoint exists here. No model was trained or exported.**
Authored unit text/numeric fixtures are software tests, not a validated dataset.
Existing YAMNet/Vosk/Whisper assets do not constitute risk-classifier weights.

## Inspection and implementation

Inspected existing `classifier/risk.py`, `training/train.py`,
`evaluation/metrics.py`, `pipeline.py`, Task 2 ASR API/controller, Task 1 ingestion,
Android `Engine.kt`/`OfflineSpeech.kt`, configs, tests, license/model notices and
prior reports. The old heuristic used sparse casefold/regex rules, while the old
trainer lacked rigorous schema/duplicate handling, used one global threshold and
exported without mandatory training provenance/parity. It was not trained ML.

| Area | Implemented status | Verification / limits |
|---|---|---|
| Multi-label taxonomy | Existing eight labels/order preserved; neutral exclusive | [Definitions/exclusions](risk-taxonomy.md); no intent/gender/consent inference |
| Text/context | Bounded NFC/whitespace handling, negation preserved, unsafe Unicode rejected, explicit ASR uncertainty/optional caller speaker IDs | English/Hindi/romanized/Hinglish authored fixtures only; no measured coverage |
| Heuristic | Threat/distress/coercion/refusal continuation/advance/secrecy/isolation rules; explicit quote/news/fiction/negation suppression | Always UNVALIDATED_HEURISTIC/uncertain; misses and false positives remain |
| Stable inference API | Original Risk fields plus optional scores/version/reasons, explicit fallback, local-only model load and close | Fake-tensor plumbing/integrity tests; genuine transformer inference not run |
| Legacy pipeline compatibility | Keeps prior ASR uncertainty on context turns instead of discarding it | Targeted regression; independent Task 1/2 workers unchanged |
| Data validation | Strict local JSONL, review/license provenance, annotation/label consistency, duplicates and source/session/group isolation | [Data guide](risk-training-data.md), [JSON schema](../configs/risk-data.schema.json); attestations do not prove permission |
| Training | Runnable local-only CPU supervised trainer, deterministic seed/config, metadata/checksums | Guard/preflight tested; actual optimization blocked by absent data/base/dependencies |
| Thresholds/evaluation | Validation-only per-positive fixed-grid selection, unavailable unsupported thresholds, neutral derived, per-class/macro/micro/confusion/hard-negative/language reports | Deterministic arithmetic/CLI tests; no actual risk accuracy measured |
| Held-out protocol | Frozen model/dataset/row identities, exclusive one-use local test receipt | Same-output reuse/failed-run reuse rejected; cannot prevent human test contamination elsewhere |
| ONNX export | Requires research trained bundle/tokenizer, float/int8 numerical + label parity, export metadata/checksums | Untrained export rejected; actual export/parity blocked, no artifact produced |
| Android | Contract supplied; no Android code or dependency changes | [Integration contract](risk-integration-contract.md); Kotlin rules remain earlier separate baseline |

## Actual tests — 14 November 2026

- **138 Python passed, 0 failed, 0 skipped**. Includes all Task 1/2 regressions,
  real local English/Hindi Vosk synthetic-speech tests, real YAMNet and optional
  Whisper CPU smoke, and packaging checks on the prior APK. This does not establish
  risk accuracy or changed Android behavior.
- **14 Kotlin JVM passed, 0 failed, 0 skipped**. Actual SDK35/Gradle8.9/JDK21 core
  compilation and tests rerun. No Kotlin/Android sources changed. Gradle reported
  a first Kotlin-daemon startup termination (code0), recovered and completed
  `BUILD SUCCESSFUL`; SDK analytics also reported a nonfatal unwritable-settings
  warning. Both messages remain visible in the committed log.
- 34 additional Python cases over Task 2 cover multilingual rule fixtures,
  contextual refusal/acknowledgement/ASR uncertainty, hard negatives/quotes/news/
  fiction/negation, empty/invalid Unicode, context limits, lifecycle/provenance,
  strict data/annotations/license/duplicate/leakage guards, model bundle corruption,
  fake-tensor local inference plumbing, threshold/metric arithmetic, score CLI
  identity binding, single held-out consumption and untrained-export rejection.
- Training CLI returned **exit2, training_blocked: no verified dataset**. Export
  CLI returned **exit2, export_blocked** for no trained bundle; no model/export
  output directory was created. These are expected guard results, not failed tests.
- No risk training, real learned-risk inference, ONNX graph execution, numerical
  export parity or Task 3 Android device/emulator run was performed.

Evidence: [full Python log](verification/risk-task3/python-tests.txt),
[Kotlin log](verification/risk-task3/kotlin-tests.txt),
[training status](verification/risk-task3/training-status.json),
[export status](verification/risk-task3/export-status.json),
[environment status](verification/risk-task3/environment-status.json).

Actual regression commands, run from repository root (Python3.11):

```bash
SAFORA_APK=/workspace/safora-ai/android/app/build/outputs/apk/debug/app-debug.apk \
SAFORA_YAMNET_MODEL=/workspace/work/model/1.tflite \
SAFORA_WHISPER_MODEL=/workspace/work/whisper-tiny \
SAFORA_VOSK_EN_MODEL=/workspace/work/asr-task2/en \
SAFORA_VOSK_HI_MODEL=/workspace/work/asr-task2/hi \
SAFORA_VOSK_HI_FIXTURE=/workspace/work/asr-task2/hindi-synthetic.pcm16 \
HF_HUB_OFFLINE=1 PYTHONPATH=python:/workspace/work/runtime:/workspace/work/asr-runtime:/workspace/work/vosk-runtime \
python -m unittest discover -s tests -v

ANDROID_USER_HOME=/workspace/work/android-user ANDROID_HOME=/workspace/work/android-sdk \
GRADLE_USER_HOME=/workspace/work/gradle-home /workspace/work/gradle-8.9/bin/gradle \
-p android :safora-ai-core:testDebugUnitTest --rerun-tasks --console=plain
```

Dependency-free focused cases: `PYTHONPATH=python python -m unittest discover -s tests -p 'test_risk*.py' -v`.
Actual blocker checks (intentionally absent input paths, not a supplied dataset):

```bash
PYTHONPATH=python python -m safora.training.train \
  --data /workspace/work/risk-task3/no-verified-dataset.jsonl \
  --base /workspace/work/risk-task3/no-compatible-base --check-data \
  --status-output docs/verification/risk-task3/training-status.json
PYTHONPATH=python python scripts/export_encoder.py \
  /workspace/work/risk-task3/no-trained-bundle /workspace/work/risk-task3/no-export \
  --probes /workspace/work/risk-task3/no-probes.json
```

## Data, training and export blockers

No real risk dataset, real annotation agreement, real hard-negative benchmark,
permission review records or compatible local risk encoder base were supplied.
`torch`, `transformers`, `onnx`, `onnxruntime` are **unavailable** in the primary
Python environment. Nothing was downloaded to fabricate completeness. CPU
training/export implementation is guarded and unit-tested at boundaries but its
native dependency path remains unverified until legitimate data/base/runtime exist.
No precision/recall/F1/confusion counts or calibration values here represent actual
classifier performance. Missing class/reference support produces unavailable
metrics (`null`); fixture arithmetic must never be quoted as model accuracy.

The selected ONNX opset17 / dynamic-int8 path still needs the real compact model's
architecture/operator support verified. Tokenizer parity and Android operator/ABI
compatibility are not established. Even a successful future export/parity run will
not establish safety performance or phone RAM/latency. Do not publish base/random
weights as a trained SAFORA classifier. Checksum metadata establishes file integrity,
not authenticity against forged provenance or independence of a benchmark.

## Privacy, remaining limitations and Task 4

Inference is local and stateless except owned model/tokenizer resources. Context is
caller-owned, bounded and transient; transcripts are not logged or sent to an API.
Plain strings/native allocators cannot guarantee secure erasure. Context must be
cleared on gaps/switches/restarts/session changes/opt-out. Heuristic rules cannot
reliably distinguish reported abuse, sarcasm, complex negation, indirect threats,
normal contextual uses, code-switch spelling or speakers. Conservative suppression
may hide real evidence; neutral is not proof of safety. Uncertain ASR stays uncertain.

Task 4 must implement the Android controller/core metadata handoff, reconcile the
separate existing Kotlin heuristic or integrate a genuinely trained/exported model,
match text/tokenizer/tensor/threshold parity, own background lifecycle/resources and
measure actual microphone-to-event behavior and memory/battery on low-RAM phones.
Task 1/2 queues and manual SOS must remain independent. This module does not send
SOS/location/contacts/alerts or change the decision engine. Text alone never proves
criminal intent or personal danger.

Before learned deployment, obtain permission-reviewed bilingual/multilingual
conversation data, source-isolated splits, independent human adjudication and
real hard negatives. Freeze taxonomy/guidelines, keep a genuinely untouched test
release, train only on train data, choose thresholds only on validation, evaluate
once, review every condition/language/minority label and document unsupported
coverage. See [data guide](risk-training-data.md) for exact schema/run protocol and
[integration contract](risk-integration-contract.md) for Task 4 interfaces.
