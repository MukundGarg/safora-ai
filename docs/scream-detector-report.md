# Task 1 — YAMNet scream detector and streaming ingestion

14 November 2026. Scope: acoustic monitoring only. No ASR/conversation training,
Android UI changes, teammate-owned app changes, final Android integration or
emergency dispatch is implemented in this task.

## Inspection and architecture

Inspected existing audio/ASR/classifier/decision modules, their callers, Python
tests/evaluators, Android core interfaces/ring/capture pipeline, and architecture,
evaluation and prior test-report documentation before editing. Reused the existing
`Yamnet`, `WindowStream`, `TemporalDetector`, `AcousticEvent` and metric helpers.
The shared legacy Python `Pipeline` waits on an ASR lock and is not a thread-safe
microphone callback API. It remains unchanged for subsequent integration tasks.

The Task 1 entrypoint is `safora.audio.ingestion.RealtimeScreamMonitor`:

    opt-in capture callback → bounded drop-newest PCM queue → dedicated worker
      → validated mono float32 → stateful causal 16kHz resampling
      → metadata-sized windows → CPU YAMNet → temporal votes/cooldown
      → AcousticEvent(monitoring_only=True)

`ScreamProcessor` is the same synchronous core used by the worker and WAV evaluator.
The worker imports no ASR, risk classifier, incident manager or network client.
Slow inference or receivers cannot wait inside capture; the producer only validates
shape/type/bounds, copies one bounded block and attempts `put_nowait`. Numerical
validation, downmixing, resampling, inference and callbacks run on the worker.
This is bounded soft real-time Python, not a hard-real-time scheduling guarantee.

Existing detector constructor signatures and selected-score dictionary keys remain
compatible. `AcousticEvent` adds optional monitoring/timing fields; `WindowStream`
adds an absolute reset origin and explicit EOF tail operation. The WAV CLI retains
`time`, `scores`, and Boolean `event` keys and adds timing/monitoring metadata.
Out-of-range floating input that was previously tolerated near full-scale is now
rejected unless saturation is explicitly enabled. Android interfaces/code are
untouched; parity/porting and final app wiring are reserved for a later task.

## Model input/output and interpretation

- Exactly one fixed float32 waveform tensor `[N]` or `[1,N]`; N is read from the
  allocated model tensor metadata, rather than assuming15,600 in the model wrapper.
  Dynamic/multi-input variants require an explicitly compatible fixed model.
- Exactly one float32 classification tensor `[521]` or `[1,521]`. Reject embeddings,
  multi-output variants, wrong runtime shape/dtype, NaN/Inf and scores outside[0,1].
- YAMNet's **16,000Hz** input rate comes from the published model contract; tensor
  dimensions alone do not encode a sampling frequency. Input is normalized mono
  float32, never PCM integer bytes interpreted as floats.
- Screaming is **zero-based index11**. Auxiliary shout/yell/crying/whimper/wail scores
  are exposed but do not cast scream votes. No softmax or cross-class renormalization
  is added. These scores are not calibrated probabilities of personal danger.
- No logistic-regression/MIL postprocessor, trained experimental weights or speaker
  gender inference is loaded or used for decisions. Only the original pretrained
  YAMNet class scores enter the temporal detector.

PCM16 (including endianness-aware NumPy values) and normalized real floats are
supported. Two-dimensional input means `[frames,channels]`; channel-first is not
guessed. Each channel is checked before averaging, so invalid opposing channels
cannot cancel into apparently valid mono. Reject nonfinite data and out-of-range
input by default; `clip_input:true` explicitly enables saturation without gain
normalization. Resampler ringing is clamped to the model's normalized range.

## Provisional configuration

`configs/scream-stream-v1.json` is separate from the existing combined-pipeline
config to avoid changing teammate/fusion behavior.

| Setting | Default / meaning |
|---|---|
| Model window | Tensor-derived; supplied model is15,600 samples =0.975s |
| Normalized rate |16,000Hz, fixed YAMNet contract |
| Source rate |16,000Hz by default; declare actual capture rate |
| Hop |8,000 normalized samples =0.5s |
| Screaming threshold |0.30, **experimental** |
| Confirmation |2 positive scores in the last3 analyzed windows |
| Cooldown |10s on the source-audio timeline |
| Producer block |At most0.1s by default; configurable up to1s |
| Queue |4 blocks, drop newest on full; configurable capacity1..8 |
| Channels |At most2 by default; mono/stereo downmix |
| Float saturation |Off by default |
| Live EOF padding |Never |

A four-block default queue bounds nominal queued audio to0.4s, **not** wall-clock
latency: a stalled inference can hold an older in-flight block. Inference is single
owner and CPU-thread count is configurable in `Yamnet(threads=...)`. At16kHz/stereo
PCM16, queued sample payload is at most25,600 bytes with these defaults, plus an
in-flight block, framing/resampling scratch, Python and native model memory.
This is a payload bound, not a measured phone-RAM guarantee. No private recordings
or model weights are committed by this task.

Two-of-three confirmation can reject isolated peaks and adds delay; it also loses
brief/weak events. The earliest two full overlapping windows end at0.975s and
1.475s. Neither this policy nor threshold0.30 is validated for safety. A caller can
explicitly select single-hit monitoring for comparison; no threshold is selected
or tuned using the reused external set.

## Source clocks, resampling, loss and lifecycle

`submit(..., start_sample=...)` accepts a non-overlapping absolute **source** sample
index, or assigns one automatically. The clock advances even when a block is
dropped. The consumer detects gaps at the actual missing boundary, rather than
resetting older queued blocks or compressing elapsed time. Gaps clear the rolling
waveform, resampler state and confirmation votes, while preserving cooldown and
prior source time. Invalid blocks/model failures report a processing error and
force fresh continuity before more votes can accumulate.

For hardware-reported losses with no known missing-sample count, pass
`discontinuity=True`. This clears context but does not invent the lost duration.
The sample clock then represents received samples, including known app drops;
unknown driver loss remains unknown. A rate change must start a new monitor.
Common8k/16k/22.05k/24k/32k/44.1k/48k/96k rates are supported; excessive coprime
ratios are rejected to bound FIR size/CPU. Maximum source rate is192kHz.

The causal FIR resampler retains bounded history and rational phase, so arbitrary
callback partitions match whole-signal causal filtering. It does not independently
zero-phase-resample each block. Group delay is reported (for example0.625ms at
48k→16k). Startup filtering and an unflushed FIR tail are explicit limitations.
The offline `waveform(..., rate=...)` utility still offers zero-phase resampling;
the new file evaluator deliberately uses the live causal processor instead.

Events use normalized window-end timestamps on the source-audio timeline.
`window_start` and measured `inference_seconds` accompany events. A live
`capture_to_decision_seconds` uses a caller's monotonic first-sample timestamp;
without one, the producer estimates it from block arrival minus block duration.
Hardware clock offsets/jitter are not corrected automatically. The optional
microphone CLI maps PortAudio ADC/current-time differences to the monotonic clock.
Offline file runs leave capture latency null and separately report predict-call
wall time; they do not pretend to measure live capture/queue latency.

`stop(timeout)` rejects new blocks, discards queued data, suppresses pending event
callbacks and waits for the worker. Native inference cannot be forcibly cancelled.
False means it is still running: **do not close/reuse the interpreter until stop
returns True**. The CLI first stops capture, then joins before closing its model.
Owned queued/window buffers are cleared on cleanup; this is not a claim of secure
erasure of every native/OS copy. Only metrics/scores are reported; no raw audio is
written by the runtime. Slow or failing user callbacks cannot block capture, though
they can delay the worker and cause counted drops.

## Reproducible evaluation

```bash
pip install -e '.[yamnet]'
python scripts/fetch_yamnet.py models/yamnet.tflite
python scripts/infer_wav.py models/yamnet.tflite /path/to/consented.wav
python -m safora.evaluation.scream --model models/yamnet.tflite \
  --manifest /path/to/recordings.jsonl --split test \
  --config configs/scream-stream-v1.json --output work/scream-evaluation.json
```

Manifest can be a JSON array or JSONL. Schema template only, **not labeled data**:

```json
{"id":"recording-id","path":"relative.wav","source_id":"original-source-id","group_id":"session-or-group-id","split":"test","label":null,"events":null}
```

Supply true/false labels only from reviewed provenance. `events` is null/omitted
for unknown timing, or a complete sorted non-overlapping list of `[onset,end]`
seconds. `[]` means known no events, not unknown timing. Positive labels with empty
complete annotations and negative labels with positive intervals are rejected.

The entire manifest is checked **before** split selection. Source/group/session
IDs, canonical paths and supplied recording-hash provenance keys cannot span
splits. `source_id` is mandatory. Metadata is caller-supplied: software cannot infer
original sessions or detect mislabeled source ancestry; provided hash keys are not
a substitute for an independent data audit. Independent validation/test grouping
must be decided before calibration, including derived/cropped/noisy copies.

Use split `reused_external` for the repeatedly used30-clip set. Such reports say
`reused_not_independent`. Fixed-config evaluation performs no fitting/sweeps;
`calibrate.sweep` now rejects explicit reused markers (even if called validation)
as well as non-validation splits. Do not relabel reused data as fresh validation.
No reused files or independent positive/negative evaluation dataset were supplied
for this task; no thresholds were optimized and no new accuracy was measured.

Outputs include confirmed clip detections, separate diagnostic single-hit/peak
scores, confusion matrix `[[TN,FP],[FN,TP]]`, precision/recall/F1, missed positive
IDs, per-clip coverage and errors. Clip labels missing/null are excluded rather
than invented. Timing annotations missing/null are skipped independently. With no
labeled/annotated examples, the corresponding precision/recall/F1 are null.
Missing/unsupported files produce an explicit incomplete report and CLI exit1;
manifest leakage/invalid labels fail before inference.

Event matching is chronological one-to-one within `[onset,end+tolerance]`, default
1s post-end tolerance. Duplicate detections are unmatched false events. The delay
is window-end minus matched onset, not a measured phone emergency-response delay.
Processing wall time is separate. Repeated votes can emit again after cooldown;
long-event annotation/evaluation conventions must account for that policy.

False alarms per negative hour count **events**, not false-positive files. Both
full negative recording hours and uniquely analyzed negative hours are reported;
overlapping windows are not double-counted as exposure. Unknown labels add no
negative hours. If no negative audio was actually analyzed, rates are null rather
than claiming zero false alarms/hour.

Short clips remain visible: incomplete windows are dropped by default, peaks are
null when no window was scored, and labeled positive misses count as false
negatives with `insufficient_audio` status. Optional `--pad-tail` scores at most one
fresh EOF tail, records padding and clamps decision time to actual EOF. It never
creates repeated artificial votes from a short file. Padding overlapping tails can
still alter confirmation, so it is an explicit offline experiment, never live
behavior. No temporal padding policy repairs faint/millisecond scream generalization.

## Experimental evidence supplied by the project owner

These are **provided observations, not results rerun in this task**. Matrices use
`[[TN,FP],[FN,TP]]` and must not be presented as fresh held-out safety performance.

| Experiment | Provided observation |
|---|---|
| Original YAMNet,0.30 single hit, development |Recall22/34; one nominal false-positive file |
| Recording-level logistic regression, original76 clips |External30 at0.80: `[[11,9],[6,4]]` |
| Weakly supervised streaming/MIL classifier |External30: `[[10,10],[6,4]]` |
| Original YAMNet,0.30, external30 |`[[18,2],[8,2]]` |

External positives include faint and extremely brief screams, some milliseconds
long. The30 clips have been reused and are **not independent held-out data**.
Logistic-regression/MIL weights must not drive production decisions and are not
used here. The new two-hit runtime is not the same policy as the provided single-hit
experiment; those figures cannot be claimed as its measured recall. Human scream
sensitivity, hard negatives/media, traffic/distance, diverse voices and conditions
still need a fresh, appropriately licensed, source-isolated evaluation.

## Test results and precise limits

- Python: **69 tests passed,0 failed,0 skipped**, including40 new focused logic/DSP/
  evaluator tests and a new real-model48kHz stereo ingestion smoke test. Existing
  real YAMNet/Whisper and APK packaging regression tests also ran using already
  provisioned assets; no ASR/training implementation was changed.
- Tests cover PCM16/full-scale/invalid clipping, silence, short/empty tails, metadata
  shape/dtype/indexing, resampler chunk equivalence/anti-aliasing, deterministic
  votes, threshold equality, cooldown, gaps, source clocks, overflow, slow native
  work, stop/callback lifecycle, label handling, isolation and false-alarm arithmetic.
- Real YAMNet tests use synthetic silence, **not representative scream recordings**.
  The evaluator CLI was exercised on an unlabeled generated48kHz stereo silence
  WAV; metrics correctly remain null. The bounded WAV runner emitted0.975/1.475s
  windows with no acoustic events. This establishes plumbing only.
- Kotlin core regression: **11 passed,0 failed,0 skipped** on a forced rerun,
  unchanged source. Current run/evidence
  is recorded under `docs/verification/scream-task1/`; no new Android runtime,
  microphone or APK integration claims are made by this task.
- Python compileall and CLI help were checked. Physical microphone capture through
  sounddevice/PortAudio was not tested; install the optional dependency and use
  an authorized input device to verify it locally. No phone RAM/thermal/battery
  guarantee, real-world recall, false-alarm rate or safety certification is claimed.

Raw evidence: `docs/verification/scream-task1/python-tests.txt`,
`kotlin-regression.txt`, `evaluator-smoke.json`, `evaluator-cli.txt`,
`infer-cli.jsonl`. Prior whole-project APK results in `docs/test-report.md` are
historical and do not validate porting this new Python ingestion path to Android.

```bash
pip install -e '.[yamnet,microphone]'
python scripts/monitor_screams.py --model models/yamnet.tflite \
  --config configs/scream-stream-v1.json --opt-in-microphone
# Optional: --device 0 --channels 1. Output is monitoring-only JSON, never dispatch.
```

Keep physical SOS independent. Connecting acoustic evidence to a validated alert
policy/main Android incident manager, ASR training and conversation-risk training
is explicitly outside Task1 and remains reserved for subsequent tasks.
