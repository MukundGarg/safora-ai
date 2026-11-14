# Model assets

`asset-manifest.json` pins host YAMNet and multilingual Whisper Tiny files.
`android-assets.json` pins the models bundled in the Android debug APK:

- Google YAMNet classification TFLite1: 4,126,810 bytes, input float32[15600],
  output float32[1,521], 16kHz mono. Screaming index11; selected auxiliary classes
  are documented in the source. The supplied Kaggle-local path was absent; this
  exact public model was retrieved and real CPU smoke-tested.
- Vosk small English US0.15: 41,205,931-byte ZIP, Apache-2.0 upstream table.
- Vosk small Hindi0.22: 44,458,845-byte ZIP, Apache-2.0 upstream table.

Run `python scripts/prepare_android_assets.py` to create generated app assets.
The Gradle pre-build task checks hashes. Android verifies the hashes and extracts
only the selected Vosk archive to private storage. ZIP paths and extracted total
size are checked; installation has no runtime network dependency.

There is **no trained SAFORA conversation-risk model** in this repository or APK.
The package manifest explicitly declares `UNVALIDATED_HEURISTIC` and
`trained_weights:false`. The Python training/loading/export pipeline requires a
properly licensed labeled dataset and local pretrained encoder. Do not substitute
generic pretrained encoder weights for a fine-tuned safety classifier.

Whisper Tiny is integrated and silence smoke-tested on the development host using
CPU int8 CTranslate2. It is not the Android ASR backend. English and Hindi Vosk
models are separate; no accurate multilingual/code-switching coverage is claimed.
See `android-model-notices.md` for provenance and licenses and `docs/test-report.md`
for the precise runtime evidence. Source exports exclude generated weight files;
APK exports contain the licensed YAMNet/Vosk assets plus notices.
