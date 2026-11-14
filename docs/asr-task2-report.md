# Task 2 — offline English/Hindi ASR

## Scope and inspection

Branch `feat/task2-offline-asr` is based on Task 1
`feat/task1-scream-streaming` at `6ee9845ec8be4f10c911055fb627089fb1b3155f`.
The PR targets that branch. No merge is requested.

Inspected Python `asr/offline.py`, legacy `pipeline.py`, Task 1 `audio/ingestion.py`,
Android `Engine.kt`/`OfflineSpeech.kt`/asset preparation, existing tests, configuration,
model manifests, licenses, and prior test reports. The old Vosk batch adapter could
lose native endpoint results, lacked bounded validation/lifecycle, and shared an
acoustic inference lock in the legacy pipeline. The Android adapter used a different
PCM scale and kept only the final native result. Existing Small en/hi model choices
and public batch interfaces are retained. No UI, firmware, ASR training, risk
training, emergency contacts or Task 1 acoustic implementation changes.

## Implemented

- `asr/offline.py`: validated normalized mono16kHz input, exact signed LE PCM16,
  bounded chunk/segment/native JSON/text parsing, provisional partials, endpoint+
  final aggregation, uncertain malformed/empty output, reset/idempotent close,
  explicit declared language and confidence semantics. No transcript logger;
  transcript repr hides text. Optional Whisper requires complete local assets,
  including tokenizer, rather than allowing its library's download fallback.
- `asr/streaming.py`: independent bounded nonblocking producer/FIFO/worker,
  deterministic drop-newest/gap reset, source timestamps, ordered flush, generation
  invalidation, language switch, restart, low-memory suspension and counters.
  One decoder model resident; no automatic LID or cross-model concatenation.
- Legacy pipeline uses separate speech/acoustic locks; targeted concurrency test
  verifies ASR cannot hold the acoustic lock. That legacy asyncio API is not a
  real-time capture callback: use Task 1 + the independent Task 2 controller.
- `asr/assets.py`/`prepare_vosk.py`: verified safe bounded archive extraction,
  required nonempty files, optional supplied SHA validation, atomic fresh output,
  extracted-file receipts. Android asset script reuses archive validation. Hindi
  0.22 lacks English's `ivector/online_cmvn.conf`; only actual common required
  files are enforced. No model binary enters Git.
- `evaluation/asr.py`: JSON/JSONL labeled-manifest evaluator with language/source/
  session/split isolation, per-language and condition WER/CER, explicit missing
  reference/unavailable metrics, empty-reference insertion accounting, bounded
  PCM16 WAV input, decoder selection, processing time, no saved transcripts.
- Isolated Kotlin codec and Vosk batch fixes preserve existing interfaces; no new
  Android streaming integration. See [Android contract](asr-android-contract.md).

Provisional defaults are in `configs/asr-stream-v1.json`: mono16kHz, two queued
chunks, ≤1s per chunk, ≤12s per segment, partial emission disabled. Model language
must be selected; unknown suspends decoding. Confidence threshold0.75 is a backend
heuristic, not a validated correctness/safety threshold. See contract for flush,
clock/gap semantics, callbacks and memory bounds.

## Reproduce preparation and tests

Install locally available dependencies (`pip install -e '.[vosk]'` during setup).
Inference has no network path; provisioning is explicitly separate. Existing pinned
specs/checksums/licenses remain in `models/android-assets.json`,
`models/android-model-notices.md` and `LICENSES.md` (Apache-2.0 as recorded upstream).

```bash
PYTHONPATH=python python scripts/prepare_vosk.py --language en --archive /local/vosk-en.zip --output /local/models/en
PYTHONPATH=python python scripts/prepare_vosk.py --language hi --archive /local/vosk-hi.zip --output /local/models/hi
PYTHONPATH=python python scripts/prepare_vosk.py --language hi --archive /local/vosk-hi.zip --output /local/models/hi --validate-only
# Add --download only during explicit provisioning if an archive is absent.

SAFORA_VOSK_EN_MODEL=/local/models/en SAFORA_VOSK_HI_MODEL=/local/models/hi \
PYTHONPATH=python python -m unittest discover -s tests -v

PYTHONPATH=python python scripts/asr_smoke.py --english-model /local/models/en \
  --hindi-model /local/models/hi \
  --english-pcm16 android/app/src/androidTest/assets/synthetic-refusal.pcm16 \
  --output /local/asr-smoke.json
# Optional --hindi-pcm16: authored synthetic or authorized speech, mono16kHz LE PCM16.
# Without a fixture, smoke uses synthetic silence; no accuracy score is produced.
```

Actual host run used Python3.11, NumPy, Vosk0.3.45; local optional runtimes
ai-edge-litert2.3.0 and faster-whisper1.2.1/CTranslate2 4.9.0. `PYTHONPATH` also
included `/workspace/work/{runtime,asr-runtime,vosk-runtime}`. Full regression command:

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

## Actual results — 14 November 2026

See committed [Python log](verification/asr-task2/python-tests.txt),
[Kotlin log](verification/asr-task2/kotlin-tests.txt) and
[native smoke JSON](verification/asr-task2/native-smoke.json).

- Python: **104 passed, 0 failed, 0 skipped** after cleanup/fallback edge tests.
  All Task 1 tests preserved, including real YAMNet smoke; optional Whisper and
  existing-APK packaging regressions also passed. Packaging checks used the prior
  APK, not an APK rebuilt with Task 2 code.
- Kotlin: **14 passed, 0 failed, 0 skipped** (three new PCM codec tests); core
  production and unit-test sources compiled with actual SDK35/Gradle8.9/JDK21.
  SDK analytics initialization printed a nonfatal unwritable-settings warning.
- Real pinned English0.15 and Hindi0.22 archives were SHA256-verified, safely
  prepared, receipt-validated and loaded sequentially on host CPU. English authored
  8s synthetic speech: nonempty final output, ~1.14s load/~2.97s decode. Hindi
  authored ~2.04s synthetic speech: nonempty final output, ~1.29s load/~3.44s decode.
  Timings are single-run host observations, not phone latency guarantees.
- Real integration tests cover English partial/reset/final/silence/close with HTTP
  requests forbidden, and Hindi controller final/suspension/model release. Native
  Hindi decoding and streaming ran offline from local assets. Deterministic tests
  cover PCM endpoints, empty/short/silent input, malformed native responses,
  overflow/gaps/flush, language switch, stale output, reset/restart/stop, failed
  cleanup recovery, slow ASR versus independent acoustic capture, privacy and
  evaluator clean/quiet/noise/code-switch metadata. Fake recognizer fixtures test
  contracts and arithmetic, not acoustic accuracy.

English archive SHA256:
`30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498`.
Hindi archive SHA256:
`7c50a10866889f0ac21d912c20537a055a597ed09fc1d3e5bcd798f9f0017e48`.
Both match the existing pinned manifest. Hindi smoke fixture was generated locally
with espeak-ng1.52, voice `hi`, speed135, authored text `नमस्ते आप कैसे हैं`, then
linearly resampled from its WAV rate to16kHz and converted with `pcm16le`. The
fixture is synthetic and not committed as a model/evaluation corpus. Recreate it
with `espeak-ng -v hi -s 135 -w hindi.wav 'नमस्ते आप कैसे हैं'`; decode PCM WAV,
interpolate at `arange(round(n*16000/rate))*rate/16000`, convert normalized floats
with `pcm16le`, and supply `SAFORA_VOSK_HI_FIXTURE`. No transcript text from the
recognizer is retained in the smoke report.

## Evaluation usage and honest limits

Each JSON/JSONL row declares `id`, `path` (PCM16 mono16kHz WAV ≤12s), `language`
(`en`, `hi`, `hinglish`, `unknown`), `source_id`, `session_id`, `split` (`train`,
`validation`, `test`, `reused_external`), optional `group_id`, `condition` and
`reference`. Real reference text must be supplied by authorized labeling; null or
omitted means unknown. An empty string means explicitly labeled no speech, with
undefined per-clip rate denominator and insertion counts retained. Hinglish/unknown
require an explicit `asr_language: en|hi` or are unavailable; no language guessing.
Example **unlabeled** row (no accuracy can be computed):

```json
{"id":"recording-1","path":"authorized.wav","language":"hi","source_id":"source-1","session_id":"session-1","split":"test","condition":"quiet","reference":null}
```

```bash
PYTHONPATH=python python -m safora.evaluation.asr --manifest /local/manifest.jsonl \
  --split test --english-model /local/models/en --hindi-model /local/models/hi \
  --output /local/asr-evaluation.json
```

The entire manifest is checked for path/source/session/group leakage before split
selection. Metrics use NFC, casefold, collapsed whitespace; punctuation is retained,
CER counts Unicode code points including spaces (not graphemes). Rates are corpus
edit counts divided by reference counts, separately for declared English/Hindi and
other strata. Quiet/noise/clean/code-switch conditions remain caller annotations;
no labels are inferred. Missing references yield `null`, not perfect accuracy.
Unavailable recordings are counted and unscored; CLI exits1 for incomplete runs.
A zero-exit run without labels still has unavailable WER/CER. Provenance consistency
does not itself prove dataset independence. Reused external data is explicitly
marked non-independent. JSON contains IDs/counts/status/timings, not hypotheses or
reference text; use opaque IDs rather than private text.

**No representative labeled English/Hindi/Hinglish evaluation corpus was supplied.**
WER/CER, quiet/noisy speech coverage and code-switch accuracy are not measured.
EnergyVAD remains an unvalidated fallback and this controller does not add a new
VAD. Short/quiet speech may be missed upstream; Vosk may hallucinate words from
noise. Confidence does not prove transcript correctness or personal danger.

No Task 2 APK was run on an Android device or emulator. Historical device/emulator
results in `test-report.md` apply to the older artifact only. Remaining work:
Android independent ASR worker and lifecycle/clock/gap wiring, selected-language
UX/coverage state, real microphone tests, native malformed-response/device tests,
low-RAM peak memory/battery/contention measurement, and separately licensed labeled
speech evaluation. No claim of criminal intent or automated SOS is introduced.
