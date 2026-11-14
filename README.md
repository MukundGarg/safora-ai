# SAFORA AI — offline Android demonstration and Python tooling

A buildable Android demo and modular Python safety pipeline. This is development
software, not a validated emergency detector. Conversation risk uses a clearly
marked heuristic baseline: **no trained SAFORA risk-classifier weights exist in this repository**.
Real pretrained YAMNet and offline English/Hindi Vosk ASR models are provisioned
and bundled in the debug APK. English/Hindi models are separate; code-switching
and real-world safety/ASR accuracy are not established.

## Build and install Android

Install Android SDK35/build-tools35.0.0/platform-tools and JDK17 or compatible newer
JDK. The verified build used OpenJDK21. Model downloads happen explicitly at build
preparation; the installed app has **no INTERNET permission**.

```bash
python scripts/prepare_android_assets.py
export ANDROID_HOME=/path/to/android-sdk
cd android
./gradlew :safora-ai-core:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The Gradle8.9 wrapper is included with a distribution checksum. Pre-build checks
reject missing/stale/corrupt model assets. Source exports omit model weights;
`prepare_android_assets.py` recreates them from pinned public assets and retains
license notices. The APK is self-contained and needs no cloud API or model download
on the phone. First-use ASR extraction uses app-private storage.

Open the app, allow local notifications, select English or Hindi, opt in to visible
microphone monitoring and press Start. The app starts in Lite while the resource
controller observes capacity; Standard/Enhanced enable optional ASR. Eight-second
VAD-gated speech chunks run independently of scream scores. Level1 creates a
local check-in; confirmed acoustic events and manual SOS request local incident
verification. **No contacts are notified**, including on manual SOS. Stop/pause and
manual SOS remain available. Monitoring coverage and performance counters are
visible. Export anonymous metrics explicitly with the app's export button.

Native ABIs: arm64-v8a, armeabi-v7a, x86, x86_64. Targets API35/min API26. Device
compatibility, 16KB native page support, real speech quality, battery/thermal
behavior and RAM targets require physical-phone validation. Consult the actual
executed results in `docs/test-report.md`; an APK build is not a runtime test.

## Python

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
pip install -e '.[yamnet,asr]'
python scripts/fetch_yamnet.py models/yamnet.tflite
python scripts/fetch_whisper.py models/whisper-tiny
python scripts/infer_wav.py models/yamnet.tflite /path/to/consented.wav
python -m safora.evaluation.calibrate validation_scores.json calibration.json
python benchmarks/asr.py models/whisper-tiny asr_manifest.json
```

Optional real-model tests use `SAFORA_YAMNET_MODEL` and `SAFORA_WHISPER_MODEL`;
real APK packaging tests use `SAFORA_APK`. Tests skip explicitly when assets are
not configured. Whisper Tiny CPU int8 and Vosk ASR adapters are implemented.
The Android demo uses Vosk, not the Python CTranslate2 runtime. Energy VAD is an
unvalidated fallback; optional Python Silero exists but was not model-tested.

YAMNet's classification contract is float32[15600] → float32[1,521] at 16kHz.
Threshold .30, two positives in three windows, 10-second cooldown are experimental
defaults. Calibration/evaluation must use independent source/session-grouped data.
The four supplied recording summaries are not population accuracy evidence.

## Train/export conversation encoder

```bash
pip install -e '.[training]'
python -m safora.training.train --base /local/multilingual-encoder --data licensed.jsonl --output models/risk
python scripts/export_encoder.py models/risk models/risk-int8.onnx
```

No licensed labeled safety dataset was supplied, so no risk encoder was trained.
Training rejects source/session leakage and missing license metadata. Locally
fine-tuned encoder loading/export and multilabel evaluation are implemented but
not validated on real safety data. Unvalidated model metadata keeps text risk
uncertain. Synthetic fixtures are integration tests only, never training evidence.

## Verification and handoff

See `docs/test-report.md`, `docs/integration.md`, `docs/evaluation.md`,
`docs/architecture.md`, `models/android-model-notices.md` and `LICENSES.md`.
The Android instrumented suite must run on an authorized device/emulator;
`assembleDebugAndroidTest` merely compiles it. Physical-device tests remain separate
from emulator tests. The integration guide describes actual callbacks, lifecycle,
model access, incident IDs, privacy boundaries and performance counter meanings.

```bash
python scripts/verify_apk.py android/app/build/outputs/apk/debug/app-debug.apk --build-tools /sdk/build-tools/35.0.0 --output docs/verification/apk-verification.json
python scripts/export_project.py /writable/deliverables
```

Exports include source ZIP, signed debug APK, instrumented-test APK, guides and
checksums. They contain no signing keys, private recordings or trained safety
encoder. Do not publish a debug APK as a production safety application.
