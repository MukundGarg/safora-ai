SAFORA source code in this repository is licensed under Apache-2.0 (LICENSE).
Source-only exports omit model weights, private recordings and third-party datasets.
The debug APK bundles the public YAMNet and English/Hindi Vosk models documented
in models/android-assets.json and models/android-model-notices.md, with license
notices and Apache-2.0 text. It includes no trained SAFORA risk encoder.
Dependency/model license inventory (verify exact downloaded revisions before shipment):
- NumPy, SciPy: BSD-3-Clause; Python: PSF; Kotlin: Apache-2.0.
- TensorFlow Lite/YAMNet code: Apache-2.0; model provenance and Kaggle terms must be checked separately.
- faster-whisper, CTranslate2, Whisper code/weights: MIT upstream; record exact model card.
- Silero VAD: MIT upstream; record exact asset revision.
- Transformers: Apache-2.0; PyTorch: BSD-style; ONNX Runtime: MIT.
- multilingual MiniLM candidate: check upstream model card (commonly Apache-2.0).
- Android SDK/tools and Gradle distributions carry their own terms.
AudioSet labels/ontology and source videos do not grant blanket redistribution rights
for recordings. Obtain consent, licenses and deletion policies per dataset; retain
provenance privately. No training dataset is supplied or represented as licensed.

- Vosk Android library: Apache-2.0; separately verify each language model's license.
- JNA: LGPL-2.1-or-later or Apache-2.0 dual license; select Apache-2.0 for packaging.
- Robolectric Android framework artifacts were used only for compile checking,
  not distributed. Subsequent APKs were built using the actual Android SDK35;
  the older surrogate check is not presented as device/runtime verification.
