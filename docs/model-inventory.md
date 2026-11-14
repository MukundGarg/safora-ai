# Model and runtime inventory — 14 November 2026

The restored/hardened tree separates **model weights**, **runtime libraries** and
**heuristic rules**. Three language/acoustic artifacts can be provisioned/verified
independently; they are not three newly trained SAFORA models. Source contains no
huge model binaries, private audio or learned risk checkpoint.

| Artifact ID / model version | Type/contract | Pinned file bytes / SHA256 |
|---|---|---|
|google-yamnet-classification-tflite-1 /1|Real generic AudioSet pretrained YAMNet;16kHz mono float32[15600]→float32[1,521], scream index11|4,126,810 / `10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de`|
|vosk-model-small-en-us-0.15 /0.15|Real offline English ASR;16kHz mono normalized float→signed PCM16|41,205,931 ZIP / `30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498`|
|vosk-model-small-hi-0.22 /0.22|Real offline Hindi ASR;one selected monolingual model, no automatic Hinglish LID|44,458,845 ZIP / `7c50a10866889f0ac21d912c20537a055a597ed09fc1d3e5bcd798f9f0017e48`|
|risk-v2 / risk-v2-rules|UNVALIDATED_HEURISTIC Python/Kotlin rule implementations, no learned weights|No trained-weight hash exists|
|Learned multilingual risk encoder|ABSENT; no genuine fine-tuned/exported checkpoint|BLOCKED by reviewed human-labeled data/base/dependencies/evaluation/parity|
|Historical LR/MIL scream experiments|Not live/runtime artifacts; poor reported generalization|Excluded from inference/decisions; not promoted|
|New250ms research probe|Untrained research machinery only, not live/runtime|No output weights produced; requires actual strongly annotated data|

`models/android-assets.json` schema2 is the single pinned artifact catalog. Each
real model now has artifact ID/version/provenance/input contract and archive/file
SHA256. Vosk entries additionally contain exact uncompressed size and every
extracted-file size/SHA256, derived from the already pinned verified archives.
Files/maps are reproducible from those archive bytes, not guessed hashes. Same
model versions and weights as Task 4; no accuracy improvement is implied.

## Provision, replace and verify offline

```bash
pip install -e '.[yamnet,vosk]' # Setup; network allowed only for explicit provisioning.
PYTHONPATH=python python scripts/prepare_android_assets.py --cache /local/cache
# Once cached, fail instead of making any download request:
PYTHONPATH=python python scripts/prepare_android_assets.py --cache /local/cache --offline
PYTHONPATH=python python scripts/prepare_vosk.py --language en \
  --archive /local/cache/vosk-en.zip --output /local/models/en
PYTHONPATH=python python scripts/prepare_vosk.py --language hi \
  --archive /local/cache/vosk-hi.zip --output /local/models/hi
PYTHONPATH=python python scripts/prepare_vosk.py --language hi \
  --archive /local/cache/vosk-hi.zip --output /local/models/hi --validate-only
```

Archive extraction validates SHA, size, bounded ZIP paths/entries, required files,
and pinned per-file hashes. Host smoke/validation now checks prepared directories
against the source catalog, **not self-asserted receipt hashes**. Rewriting a local
receipt cannot authenticate corrupted model bytes. Android validates archive SHA
before extraction and all pinned file hashes on cached/new directories, checks
usable storage for extraction/estimated file-block overhead and publishes a fresh
private directory only after validation. Cleanup removes partial extraction after
failure. Native load failure remains exposed as unavailable; model failure does not
disable manual SOS. Low-storage preflight is unit-tested, not a physical full-disk
phone test. Background failure retry is bounded1s→30s; unavailable input is discarded
rather than queued for later replay.

Preparation publishes changed assets atomically and leaves unchanged files alone;
run provisioning **before Gradle**, not concurrently with a build. Gradle rejects
stale metadata/model hashes/missing notices. Versioned Android private directories
are `files/models/vosk-{en|hi}-{archive_sha_prefix}`; one selected model is loaded
and released by its speech worker. Runtime has no Internet permission/API key or
cloud fallback. Input sample rate/tensor shape must match supported contracts.

Replacement means a reviewed compatible artifact, exact new source/license/size/
archive/file hashes and parity tests, followed by a rebuild/resign—not dynamically
trusting an arbitrary path/download. Independent artifact provisioning does not
remove app coordination, resource or combined safety-validation requirements.

## Native runtime packaging change

| Runtime | Previous | Candidate adopted on branch | Measured64-bit ELF PT_LOAD |
|---|---|---|---|
|Interpreter|org.tensorflow:tensorflow-lite2.16.1|com.google.ai.edge.litert:litert1.4.2, classic Interpreter API|16384bytes|
|Vosk Android|0.3.47|0.3.75|16384bytes|
|JNA AAR|5.13.0|5.18.1|16384bytes|

Official Maven artifact metadata/POMs were read; downloaded direct AAR hashes match
Gradle-resolved artifacts. See [native candidates](verification/production/native-candidates.json),
[resolved integrity](verification/production/resolved-runtime-integrity.json) and
[POM licenses](verification/production/runtime-licenses.json). Published minimum
SDKs remain compatible with this project's API26minimum. All four packaged ABIs
have the required JNI runtimes; Vosk32-bit libraries retain4096-byte ELF alignment.
The rebuilt APK passes the64-bit ELF alignment/offset congruence check as well as
separate ZIP16KB alignment. Actual16KB-page Android runtime compatibility is still
BLOCKED/NOT TESTED. Upgrading a runtime is not training or improving a model.

Runtime notices are tracked in `models/android-runtime-notices.md` and bundled as
RUNTIME-NOTICES.txt alongside model notices and Apache license text. LiteRT/Vosk
published POMs specify Apache-2.0; JNA is dual licensed and Apache-2.0 is selected.
Model license/provenance remains as recorded in existing model notices; final legal
supply-chain/NOTICE review is not replaced by a checksum or by these documents.

## Actual runtime evidence

Real local host YAMNet and EN/HI Vosk were independently exercised by regressions
and together through the coordinator, with canonical per-file verification. Host
runtimes: ai-edge-litert2.3.0, Vosk Python0.3.45; those are distinct from the new
Android libraries above. The Android candidates compiled and packaged but did not
run here: emulator disk check failed and no physical phone was attached. Optional
Whisper Tiny remains a host ASR adapter, not default Android inference or a learned
risk model. Its existing local-only smoke passed; no Whisper accuracy comparison
or replacement of Vosk is claimed.
