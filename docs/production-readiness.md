# SAFORA release gates — NOT PRODUCTION-QUALIFIED

Review date14 November2026. No stakeholder hazard-review approval, agreed safety
performance thresholds or product release signoff was supplied. **No safety-critical
product gate is marked PASS merely because software, unit tests or builds work.**
The final status is **NOT PRODUCTION-QUALIFIED**.

## Gate matrix

For every required gate, product/safety lead must approve the criterion, evidence
scope, accepted residual risks and signoff identity/date before PASS. Numeric
recall/false-alarm/latency/power/button-delay limits remain **TBD by hazard review**;
none were invented or selected post hoc here. Engineering subchecks below are not
certification or permission to dispatch contacts from model scores.

| Required product gate | Release status | Engineering evidence / exact gap | Criterion to approve / owner |
|---|---|---|---|
|Artifact correctness, provenance, distribution rights|BLOCKED|Pinned YAMNet/Vosk archive+all extracted hashes pass; new runtime AAR hashes/POM license records match resolved files; runtime/model notices bundled. Final legal/NOTICE/supply-chain approval absent|Approved provenance/license policy and signed artifact/version review; Audio ML + legal/security/release owner|
|Scream event performance/false alerts|BLOCKED / NOT MEASURED|No independent annotated audio benchmark; reused30-file numbers are historical and do not validate2-of-3. New event/negative-hour evaluator exists|Predeclare event recall/FA-hour CI/latency/condition targets, untouched source/speaker/session/root isolation; safety lead + Audio ML/data reviewers|
|English ASR WER/CER|BLOCKED / NOT MEASURED|EN0.15 runs on synthetic host fixtures; no representative human reference corpus|Approved normal/urgent/shout/quiet/noise/mic-motion WER/CER/latency/coverage criteria; bilingual annotation + Audio ML|
|Hindi ASR WER/CER and Hinglish coverage|BLOCKED / NOT MEASURED|HI0.22 runs; Hinglish automatic LID not implemented; no code-switch accuracy|Language-specific approved metrics/reference normalization and unknown coverage policy; Hindi/Hinglish reviewers + Audio ML|
|Learned conversation-risk model|BLOCKED|UNVALIDATED_HEURISTIC only; no genuine licensed human-labeled dataset/base/checkpoint, training/export/parity; required optional dependencies absent|Approved taxonomy, consent, independent labels/splits, per-label/hard-negative/language metrics/calibration and residual risk; safety/data/Audio ML leads|
|Python/Kotlin real-model/quantization parity|BLOCKED / NOT TESTED|12shared rule fixture contracts pass; no new native-runtime score/event/device or learned tokenizer/quantization parity|Approved probes/tolerances/conditions and supported operators/devices before model promotion; Audio ML + Kotlin owner|
|Low-RAM latency, battery, thermal|BLOCKED / NOT TESTED|Single host paced smoke RSS330004KiB, not Android PSS/battery. No representative2–3/4/6–8GB phones|Approved model-load/inference/drop/PSS/energy/thermal budgets on agreed devices/durations; Kotlin/performance owner + safety lead|
|Physical wearable intentional SOS reliability|BLOCKED / NOT TESTED|No firmware, transport/button protocol or hardware. App manual path software tests pass; physical authentication/replay/ACK/debounce/reconnect/battery/order absent|Actual board/firmware/key/transport contract and reviewed button-to-app/end-to-end reliability/delay/fault criteria; hardware lead + Kotlin/incident owner|
|Permission/background/audio/connectivity/sensor faults|BLOCKED / NOT TESTED|Injected lifecycle/queue/failure tests pass; OS revoke/lock/call/process-death/OEM/physical sensor tests absent. No assumed sensors|Approved real fault matrix, coverage descriptions/manual independence and recovery; Kotlin + hardware + safety owners|
|Privacy/security/consent|BLOCKED|No runtime Internet permission/API key/cloud transcription or persistent raw audio/text logs added; private files ignored; real consent/security threat-model/withdrawal audit absent|Reviewed collection/retention/permissions/authentication/incident policy; privacy/security + product owner|
|Native/Android distribution compatibility|BLOCKED|64-bit ELF16KB alignment/offset congruence and ZIP alignment now pass; no16KB-page device runtime. Debug/instrumentation and unsigned release APK/AAB compile; release signing/R8/device/store readiness not qualified|Approved ABI/API/OEM/page-size matrix, signing/key custody, shrinker/native/resource and store tests; Kotlin/release owner|
|User-confirmed check-in and independent manual workflow|BLOCKED|Unit tests preserve manual action with unavailable models/permission/opt-out; ML cues remain uncertain check-ins only. Real user/device/wearable incident flow not validated|Approved user action/cancellation/location/contact policy and measured independent manual UX; incident/product/Kotlin + hardware owner|

## Measured engineering checks (separate from release gates)

- **PASS:** recovery merge on a new branch from current main preserves original
  Task 1–4 ancestry. Main itself is unchanged pending PR review. See
  [reconciliation](repository-reconciliation.md), merge055dc672 and exact79paths.
- **PASS:** recovered tree rerun147Python/22Kotlin; hardened tree160Python/27Kotlin,
  zero failures/skips. No previously reported count was assumed for the new tree.
- **PASS:**6actual APK integrity/ABI/fixture/file-map/ELF checks; debug APK signature
  and16KB ZIP alignment. Native64-bit static alignment changed from prior FAIL to
  PASS with verified LiteRT1.4.2/Vosk0.3.75/JNA5.18.1. Runtime compatibility remains
  a blocked product gate, not automatic PASS. Vosk32-bit ELF remains4096-aligned.
- **PASS:** debug app/instrumentation plus release APK/AAB compile; lint0errors/
  12warnings. Release APK/AAB have no configured production signing; R8/shrinking
  is not enabled/qualified by this build. The debug APK uses a development signature.
- **PASS:** actual local native host YAMNet+EN/HI Vosk coordinator smoke and cleanup,
  not microphone/phone validation.2nonempty TTS results,31YAMNet windows,0errors/
  drops,0positive heuristic results. No alert was fabricated to make the demo pass.
- **EXPECTED BLOCK:** scream fit/risk fit/evidence commands with absent reviewed
  datasets return exit2 and create no models. Missing references/unsupported metrics
  remain unavailable, not invented accuracy. No genuine learned model produced.
- **BLOCKED:** emulator startup failed disk-space preflight; no physical phone or
  wearable attached, no KVM. New Android runtimes were not executed on Android.

## Evidence, identities and reproduction

All current evidence is under `docs/verification/production/`:
reconciled-python/reconciled-kotlin, python-final/android-final, kotlin-summary,
apk-verification/build-artifacts, native-candidates/resolved-runtime-integrity/
runtime-licenses, asset preparation/idempotence, native-coordinator, emulator
attempt, environment/local-data inventory and blocked training statuses. Historical
Task 1–4 logs remain intact. A transient duplicate asset-merger failure is retained
in asset-merger-failure.txt; provisioning happened during an active build, then the
generated merger state was rebuilt with stable assets. Publication is now atomic/
idempotent. This was fixed, not hidden or misrepresented as a device result.

Dataset identities/sizes: **no independent human-labeled safety dataset supplied**.
Tracked audio:2authored TTS PCM fixtures, not a benchmark. Local Hindi fixture is
also authored TTS (32637samples), as documented in Task 2. No unknown Kaggle files
were presumed accessible. Historical30-file external test has been reused and
cannot be untouched. Source/speaker/session/root/hash/crop checks exist for new
manifests but do not prove authentic consent or independent labels.

Device identity/class: only Linux development host, Python3.12.14, actual Android
SDK35/JDK21/Gradle8.9 build tools; attempted API26software emulator never booted.
No physical phone model/OEM/RAM/battery/firmware identity exists to report. Host
runtimes are distinct from upgraded Android artifacts; do not transfer host metrics
to phones. Exact model IDs/SHA256s and commands are in [inventory](model-inventory.md),
[model evaluation](model-evaluation-report.md), [audio protocol](audio-data-protocol.md)
and [wearable boundary](wearable-android-integration.md).

Actual commands (from repository root; no inference network):

```bash
PYTHONPATH=python python scripts/prepare_android_assets.py \
 --cache /workspace/work/packaging-cache --offline
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
-Dorg.gradle.workers.max=1 -p android :safora-ai-core:testDebugUnitTest \
:app:assembleDebug :app:assembleDebugAndroidTest :app:assembleRelease :app:bundleRelease :app:lintDebug --console=plain
PYTHONPATH=python python scripts/verify_apk.py \
 android/app/build/outputs/apk/debug/app-debug.apk --build-tools /workspace/work/android-sdk/build-tools/35.0.0 \
 --require-native-16kb --output docs/verification/production/apk-verification.json
```

Current debug APK153,456,660bytes, SHA256
`3ef70b5af19578417cda0163a6b42632dd52b93faaf81709347dbb9fe209dfba`.
Unsigned release APK151,332,920bytes; release AAB111,097,768bytes. Exact hashes in
build-artifacts.json. Native compatibility upgrade increased debug APK size from
the historical Task 4 artifact; no storage-size/performance gain is claimed. No
artifact was installed/run or uploaded to a store. Source does not include these
binaries or signing keys.

## Required remaining decisions and owners

Safety/product lead convenes a documented hazard review and sets acceptance
limits/confidence assumptions/device classes, data independence rules and signoff
before anyone can mark these gates PASS. Audio ML/data reviewers acquire real
licensed/consented strong-event audio, hard negatives and multilingual ASR/risk
labels; run train-only/validation-only/untouched-once studies with no post-hoc test
tuning. Kotlin owner runs real native parity/OS lifecycle/mic/resource/notification/
release tests and integrates truthful coverage in teammate-owned UI. Hardware lead
supplies reviewed firmware/transport/key/counter/button/battery/ACK semantics and
physical devices; no guessed stack or gesture implementation substitutes for it.
Incident manager owns user-approved contact/location/cancellation/dispatch policy.

No merge, rollout or production qualification is authorized by these software
results. **NOT PRODUCTION-QUALIFIED** until every required product-specific gate
has real evidence and stakeholder signoff.
