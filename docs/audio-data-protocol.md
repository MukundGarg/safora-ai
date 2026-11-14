# Audio evidence and experiment protocol

No independent annotated human audio corpus is available in this checkout. Only
two authored eSpeak PCM fixtures are tracked; local generated Hindi TTS and model
assets are available in the workspace. `infer-cli.jsonl` is an earlier score log,
not recordings, consent or labels. Kaggle/AudioSet links do not grant blanket rights
to download/redistribute original clips. The reused30-file external scream set is
not checked in and is not independent even if obtained later. Do not tune on it.

## Strict audio manifest (local private JSONL, never fabricated training data)

Common required fields: `id`, `path` (PCM16 WAV≤600s,8–48kHz,mono/stereo), actual
file `sha256`, exact `duration_seconds`, `source_id`, `session_id`,
`root_recording_id`, reviewed `speaker_ids` (nonempty opaque grouping, not inferred
identity/gender), `device_id`, `condition`, `channel_metadata`, `noise_metadata`,
`split` (train/validation/test/reused_external), `license`, and `provenance` containing
`verified:true`, `origin`, `reviewed_by`, `permission_basis`. Permission fields are
supplier attestations, not automatic legal approval. Real collection needs explicit
consent/withdrawal/access controls and compatible license/retention review.

For scream: `events:[[onset_seconds,offset_seconds],...]` and
`negative_segments:[[start,end],...]`, both explicit ordered nonoverlapping lists,
with no positive/negative overlap and no invented labels for unknown time. Empty
lists are allowed only if the other contains actual reviewed time. Clip-only weak
labels are not accepted by the supervised experiment. `parent_recording_id` links
cropped derivatives; always retain common source/root/speaker/session grouping.
Optional `hard_negative`: baby_crying, laughter, cheering, singing, shouting,
sirens, traffic, conversation, household, noisy_environment, device_handling, other.

For ASR: same permission/provenance/device/speaker metadata plus explicit `language`
(en/hi/hinglish/unknown), optional `asr_language` (en/hi), `reference` (real reviewed
text or null/omitted). Utterance≤12s; long recordings require separately reviewed
segment references, not guessed cropped transcripts. Condition should explicitly
label normal/urgent/distressed/shouting/quiet/noise/mic_movement/code_switch when
actually annotated. Missing references stay unavailable, not0%error. Hinglish is a
separate stratum and needs one explicitly chosen decoder; there is no automatic LID.
WER/CER use documented NFC/casefold/whitespace normalization, retaining punctuation;
Hindi code-point CER is not grapheme CER or a transliteration benchmark.

All manifest rows are checked before selecting a split: source/session/root/
speaker grouping, path/file hashes, exact decoded PCM hashes and0.25s acoustic PCM
chunk fingerprints (0.125s hop, low-variance/silent chunks excluded). This detects
header variants and some aligned crops, not arbitrary resampling, offset/noisy/gain
variants or semantic duplicates. Human derivative/source review remains required.
Hashes/group consistency cannot prove independent human labeling or speaker identity.
Device IDs record strata; unseen-device claims additionally need a reviewed device-
holdout policy, not merely a nonempty device field.
Fingerprint index is bounded to250000distinct chunks; larger corpora need a reviewed
external global index rather than silently dropping checks. Manifests≤64MiB,
100000rows; audio decoding≤600s/file. Real data must remain outside Git.

## Annotation/collection requirements

Recruit consented participants and documented legitimate negative sources; never
infer distress/consent from pitch/gender or label whole recordings positive for a
millisecond event. Independently annotate onset/offset, uncertain regions, reviewed
negative intervals and disagreement/adjudication. Include ordinary/distress/brief/
faint/distant screams, device/mic/distance/noise diversity and hard-negative classes.
Separate all derivative clips and speakers/sessions/sources before assigning splits.
Keep an access-controlled untouched test release, with dataset hashes and all run
identities. Test data reused for model/rule changes must be retired as independent.
Safety/product reviewers must set performance/latency/false-alarm acceptance gates
before selection. No arbitrary percentage is supplied here.

```bash
PYTHONPATH=python python -m safora.evaluation.audio_experiment \
  --manifest /private/reviewed-scream.jsonl --task audit --output /private/audit.json
PYTHONPATH=python python -m safora.evaluation.audio_experiment \
  --manifest /private/reviewed-scream.jsonl --task scream-validation \
  --model /local/yamnet.tflite --output /private/frozen-validation.json
PYTHONPATH=python python -m safora.evaluation.audio_experiment \
  --manifest /private/reviewed-scream.jsonl --task scream-test --split test \
  --model /local/yamnet.tflite --frozen /private/frozen-validation.json --output /private/test-once.json
PYTHONPATH=python python -m safora.evaluation.audio_experiment \
  --manifest /private/reviewed-asr.jsonl --task asr --split test \
  --english-model /local/models/en --hindi-model /local/models/hi --output /private/asr.json
```

Fixed provisional scream rule grid0.2/0.3/0.5 uses validation only and freezes the
model/whole-manifest identities and configuration; validation needs both event
positives and reviewed negative exposure. A local exclusive test receipt is
reserved before inference; failed runs remain consumed. External copies/reruns
cannot be prevented by software: record every experiment and preserve an untouched
release. Existing non-strict Task 1 evaluator remains compatible for legacy work;
use this strict wrapper for new permission-reviewed studies.

Reports include clip/event metrics, explicit missed-condition counts, delays,
full vs analyzed negative exposure (unscored tails are not silently treated as
observed negatives), and unavailable metrics where denominators are missing.
Unknown spans are excluded from event/false-alarm evaluation. Wilson95%clip
precision/recall intervals assume independent recordings; source-cluster correlation
is not corrected. Conditional Poisson95%false-alarm intervals use analyzed negative
hours and require an independent stationary event-process assumption; no unsupported
F1/delay/cluster confidence interval is invented. Reports can include opaque IDs,
never raw ASR/reference text or audio. Keep IDs free of private transcript text.

## Guarded research-only short-window experiment

```bash
PYTHONPATH=python python -m safora.training.scream_probe \
  --manifest /private/reviewed-scream.jsonl --yamnet /local/yamnet.tflite \
  --output /private/research-only-run --allow-research-fit
```

This is an unvalidated250ms handcrafted-feature linear probe (RMS, crossings, peak,
8spectral bands), not a claimed robust distress classifier and not promoted LR/MIL
weights. Train normalization/weights use train only; a frame is positive only if it
contains an annotated event, negative only when wholly inside reviewed negative
time; unknown spans are skipped. Features never receive weak recording labels.
Fixed200gradient steps/L2 regularization and threshold grid0.3/0.5/0.7 are research
configuration, not safety acceptance. Validation selects event F1; test executes
once after freeze, comparing against unchanged YAMNet2-of-3 on the same evidence.
Outputs are explicitly `EXPERIMENTAL_NOT_FOR_RUNTIME` and no live Python/Kotlin
loader consumes them. Missing reviewed data blocks before creating any weights.
No experiment was trained here. Learned compact alternatives/finetuning need their
own legitimate data, compatible local base/runtime and parity tests; not invented
weights or undocumented Android dependencies.

A candidate may be promoted only after predeclared stakeholder hazard gates,
independent event/negative-hour evidence, language/condition failure review, actual
CPU/RAM costs, Python/Kotlin score/event and quantization parity, device faults and
reviewed residual-risk signoff. No supplied dataset satisfies those gates today.
