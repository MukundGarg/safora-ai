# Offline ASR contract for Android integration

Task 2 provides a tested Python reference controller plus isolated Kotlin codec/
batch-adapter fixes. It does **not** wire a new controller into the Android app.
The existing `SpeechRecognizer.transcribe(FloatArray): Recognition` interface stays
compatible. App/UI/firmware and Task 1 acoustic code are unchanged.

## Input and ownership

- Mono, 16,000 samples/second. Float input is finite, normalized `[-1,1]`;
  no automatic gain normalization, stereo mixing or sample-rate guessing.
- PCM wire/native bytes: signed **little-endian PCM16**, round nearest/even after
  multiplying floats by32768, saturate +1 to32767 and -1 to-32768. Kotlin helpers
  `asrPcm16`/`asrPcm16LeBytes` match Python `pcm16le`. Vosk's Kotlin ShortArray
  overload represents the same signed PCM values, without a byte-order ambiguity.
- Resample upstream once with a stateful resampler; do not resample isolated speech
  chunks independently. Preserve Task 1's dedicated capture/queue/worker.
- Python `submit` copies bounded speech chunks (default ≤1second); the caller
  retains and must clear its original buffer. Capture never calls native ASR.
  Float/PCM temporaries owned by the adapter/queue are cleared after use. Python
  strings/native buffers are not guaranteed secure erasure.

## Clock, queue and language

`StreamingASR.submit(samples, start_sample=N, sample_rate=16000, final=False)`
accepts monotonic source sample offsets on a shared 16kHz clock. End is exclusive;
seconds = sample offset /16000. Keep advancing this clock through dropped audio,
VAD-excluded audio and disabled recognition. Gaps discard unfinished recognition;
missing speech must never be silently bridged. Caller-supplied `discontinuity=True`
forces the same reset. This controller is not a VAD or microphone driver.

Use one selected decoder: `en`, `hi`, or `None`/`unknown`. Unknown means recognition
suspended, input rejected/counted, model released after in-flight work finishes.
`auto` and `hinglish` are not decoder modes. There is no automatic LID. Language
switch discards queued/unfinished old-language audio, releases the old model before
loading the new model and invalidates old-generation output. Never join independent
English/Hindi recognitions as one transcript.

A separate FIFO holds two default chunks, with one in flight. Full queue drops the
**newest** chunk immediately, counts lost samples and advances the clock; the next
accepted gap resets recognition. Configure capacity1..8, chunk≤8seconds and
segment≤12seconds. Use `final=True` for a complete speech segment; the default
`final=True` treats each submission as complete. For streaming, use `final=False`
then an ordered `flush()`/final chunk. `flush()` returns false if full/disabled/
stopped; retry from a background producer, never wait in capture. It counts as a
queued chunk but adds no samples. At segment limit the preceding bounded segment
is finalized as uncertain before starting the next. No audio is queued for later
replay when recognition is disabled or full.

## Output and lifecycle

Python `ASREvent` carries transcript, language, source start/end sample offsets,
segment ID, generation and accumulated native processing seconds. Partial emission
is off by default. Partial text is provisional; replace it, never append it.
Native endpoints finalize the current segment. Low-level `accept()` returns a
partial/endpoint **delta**, while `finish()` aggregates committed endpoints plus
the final tail for one bounded segment/model. Do not append both API levels.
Event timestamps bracket submitted audio, not forced-aligned word times. Processing
seconds excludes queue wait/model load; it is not end-to-end latency.

`Transcript.uncertain` covers empty/silent input, malformed responses, provisional
partials, unknown language, missing word confidence and low backend confidence.
The provisional Vosk threshold is0.75 mean word confidence. This is not a calibrated
correctness probability. `average_log_score` for Vosk is log(mean word confidence),
**not** Whisper's average token log probability; check `confidence_kind`. Neither
score measures intent, personal danger or the need for emergency dispatch.

Run one native model/recognizer owner per worker. `reset()` discards pending/context
but keeps the model; `restart()` also reloads it. `wait_idle(timeout)` is for tests/
background coordination. `stop(timeout)` rejects submissions, discards queued audio
and suppresses future results; false means native work has not yet returned.
Never destroy its backend from another thread. On completion the worker releases
it; native Python Vosk0.3.45 releases handles through object destruction. Close is
idempotent. Result/status callbacks run on the worker, must be fast and must marshal
UI work themselves. A callback already executing cannot be recalled: consumers
must check the event generation after resets/switches, especially after UI dispatch.
Only counters/status kinds/error class names are recorded, not raw audio or text.

Kotlin `VoskSpeech` remains a synchronous, synchronized **batch** adapter (≤12s),
collects endpoint results and final tail, creates/closes each recognizer and closes
its model once. It returns existing `Recognition(text, uncertain)`, without language,
timestamp or generation fields; the Android owner must carry those separately.
Call it on a dedicated worker, never capture/UI or an acoustic inference lock.

## Assets and low RAM

Use the existing pinned Small English/Hindi archives in `models/android-assets.json`
and existing notices in `models/android-model-notices.md`/`LICENSES.md`. Provision
and verify before inference; no network, account or API key is needed at runtime.
New Python preparation validates archives and all extracted-file receipt hashes.
Direct directory loading checks structure and optional supplied file hashes; it
cannot authenticate a directory using an archive hash alone. Native model loading
is the final structural compatibility check.

One selected model is loaded lazily. Default float32 queued payload is at most128KB
(two 1s chunks), plus64KB in flight and bounded native/PCM working storage. Caller
float64 arrays use twice that payload. Native Vosk/model memory is substantially
larger and **not measured on a low-RAM phone here**. Provisioned model directories
and archive sizes are not RAM estimates. Do not preload both languages on low-RAM
phones. Suspend ASR on memory pressure; expose unavailable/unknown/gap coverage and
keep Task 1 acoustic monitoring independent. Optional Python Whisper stays opt-in;
its complete local model/tokenizer assets are required to prevent fallback fetches.

## Exact work remaining in main app

Port the independently bounded ASR queue/generation/flush policy to its background
core owner, connect the existing VAD/segmentation and shared resampled clock, surface
explicit language and coverage state, marshal transient results without persistent
transcripts, and connect lifecycle/memory-pressure suspension. Preserve manual user
controls; ASR content alone must not trigger SOS or establish criminal intent.

Then run the changed native adapter on actual Android runtimes and physical low-RAM
phones: microphone routing/resampling, speech segmentation, real-time contention,
queue loss, language switches, lifecycle cleanup, peak RAM, battery, quiet/noisy
English/Hindi and separately annotated Hinglish. Task 2 JVM compilation/codec tests
and host-native smoke tests do not verify these Android behaviors.
