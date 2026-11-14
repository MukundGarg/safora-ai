# Model evaluation — NO DEMONSTRATED ACCURACY IMPROVEMENT

Date14 November2026. No independent permission-reviewed annotated audio benchmark,
ASR reference corpus or human-labeled multilingual conversation-risk training set
was supplied. No genuine risk encoder/base checkpoint is present. Only authored
synthetic integration audio is tracked, plus a locally generated Hindi TTS fixture;
those fixtures are not a real-world benchmark. The historical Kaggle/external audio
is absent from this checkout. Inventory: [local data](verification/production/local-data-inventory.txt).

## Before versus after

| System | Historical evidence supplied by user | New matching independent evaluation |
|---|---|---|
|YAMNet single-hit0.30|Reused30files (10weak/brief positives,20negatives):[[18,2],[8,2]], recall20%,precision50%,accuracy66.7%|NOT MEASURED; raw corpus/independent labels absent|
|LR v0.1|Same reused set:[[11,9],[6,4]],9/20negative false-positive files|Excluded; no promotion or retuning|
|MIL v0.2|Same reused set:[[10,10],[6,4]],10/20negative false-positive files|Excluded; no promotion or retuning|
|Live YAMNet2-of-3|Single-hit numbers do not validate temporal configuration|NOT MEASURED; unchanged0.30/2-of-3/10s defaults|
|Vosk EN0.15 /HI0.22|Synthetic host decoding, no representative references|English WER/CER NOT MEASURED; Hindi WER/CER NOT MEASURED; Hinglish unsupported/unvalidated|
|Conversation risk|UNVALIDATED_HEURISTIC; no trained model|No trained/exported weights; per-label learned metrics NOT MEASURED|

The historical30-file results were not recomputed here and are not independent
held-out results. They cannot support a before/after gain or final temporal safety
performance. Runtime-library upgrades and more passing tests do not establish
model-accuracy improvement. No thresholds/defaults were changed to fit these files.

## Implemented evaluation/training machinery

See [audio data protocol](audio-data-protocol.md) for strict JSONL, exact timed
positive/negative annotations, permission/license records, device/channel/noise,
speaker/session/source/root grouping, SHA/decoded/crop checks and limitations.
Strict validation runs over all splits before selection. Unknown spans are not
silently labeled negative; weak clip labels do not label every window positive.

New rule studies select only on validation, freeze catalog/model/manifest identity,
reserve test output before a single held-out run, and report clip/event precision/
recall/F1, delays/misses by condition and exposure-corrected false alarms. Unanalyzed
tails remain explicit. Conditional Wilson/Poisson95%intervals state independence/
process assumptions; unsupported denominators and unimplemented clustered/F1/delay
intervals remain unavailable. Human source/derivative review and external test
access-control remain necessary; software cannot prove consent or prevent copies.

New research-only250ms annotated-frame probe has a runnable NumPy fitter, train-only
normalization/weights, validation-only rule selection, matched YAMNet test comparison
and fresh test-consumption output. It is **not trained here**, not a demonstrated
robust compact scream detector and not read by any live runtime. Alternative neural
finetuning requires suitable licensed strong labels/local compatible bases and its
own validation/parity. A guarded training tool does not qualify a model.

Strict ASR evidence adds real references or explicit null, device/speaker metadata,
code-switch strata and pinned-model identity verification to the recovered
per-language WER/CER evaluator. Do not invent references or optimize VAD/chunking
on test data. Current8s/RMS0.008 segmentation defaults remain unvalidated; very short,
quiet, urgent/noisy speech can be missed. Low-RAM phone/ASR battery measurements are
absent. EN/HI remain explicit separate model choices, one resident, no automatic LID.

Recovered Task 3 taxonomy/provenance/duplicate/split/threshold/held-out/export guards
remain available. Genuine learned risk training/export is still BLOCKED by real
human-labeled permission-reviewed data, compatible base, missing torch/transformers/
onnx/onnxruntime and actual numerical/tokenizer/device parity. No generic/random
weights were saved under a production name; no synthetic fixture is training proof.

Actual guard commands returned exit2 without creating datasets/weights:

```bash
PYTHONPATH=python python -m safora.evaluation.audio_experiment \
 --manifest /workspace/work/no-reviewed-scream-data.jsonl --output /workspace/work/no-scream-evaluation.json
PYTHONPATH=python python -m safora.training.scream_probe \
 --manifest /workspace/work/no-reviewed-scream-data.jsonl --yamnet /workspace/work/model/1.tflite \
 --output /workspace/work/no-probe-weights --allow-research-fit
PYTHONPATH=python python -m safora.training.train \
 --data /workspace/work/no-reviewed-risk-data.jsonl --base /workspace/work/no-local-risk-base --check-data
```

Evidence: [audio status](verification/production/audio-data-status.json),
[scream fit status](verification/production/scream-training-status.json),
[risk fit status](verification/production/risk-training-status.json),
[environment](verification/production/environment.json). No actual training run or
learned export/parity is falsely claimed. Annotation acquisition and promotion
requirements are in the data protocol and production-readiness matrix.

## Executed engineering checks

Reconciled baseline:147Python/22Kotlin passed. Hardened tree:160Python/27Kotlin
passed,0failures/skips. Six actual APK checks validate embedded weights/file maps,
notices/ABI completeness, absence of test audio/untrained-risk claims and64-bit
ELF load alignment/congruence. APK signature/ZIP checks pass separately. These are
contract/integrity checks, not sensitivity/specificity or certification.

Real host coordinator smoke used YAMNet and verified EN/HI Vosk from local assets,
with31acoustic windows,2nonempty synthetic-speech results,0inference errors/queue
losses and no positive rule labels/ML check-in. Manual hook remained available
after stop. Single-run host wall8.578s/process CPU7.840s, peak RSS330004KiB. Blocks
were paced by idle barriers, not a real-time microphone. Peak process memory is not
incremental Android PSS or a low-RAM/battery result. No native Android score/event
or quantization parity was run; only12shared text-rule fixture contracts pass.
See [native JSON](verification/production/native-coordinator.json) and final logs.

## Required next independent evidence

Audio ML owner plus approved data/annotation reviewers must collect licensed or
consented event-annotated positive/negative audio, separate all roots/derivatives/
speakers/sessions, include very brief/faint/distant screams and every hard-negative
stratum, and independently review quiet/urgent/noisy/device-motion ASR references.
Risk data needs bilingual/multilingual contextual labels, hard negatives, adjudication
and known permission scope; no inferred gender/crime labels. Product/safety lead
must approve acceptance thresholds and confidence-interval assumptions **before**
validation selection/test access. Android owner must verify supported native models,
score/event/tokenizer/quantization parity, lifecycle faults and measured physical
phone memory/latency/power. Until then: **NO DEMONSTRATED IMPROVEMENT; NOT
PRODUCTION-QUALIFIED**.
