# Models in the Android demonstration

YAMNet classification TFLite v1, Google: retrieved from
https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite/1
SHA256 10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de.
Google YAMNet model/code is provided under Apache-2.0. Original model information:
https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

Vosk small English US 0.15 and small Hindi 0.22, Alpha Cephei:
https://alphacephei.com/vosk/models
The upstream model table lists Apache 2.0 for both exact models (inspected
14 November 2026). Archives retain their upstream README files. Upstream README
benchmark values are upstream claims and are NOT SAFORA evaluation results.
Record SHA256/source/size from android-assets.json. Full Apache-2.0 license is
included as MODEL-LICENSE-APACHE-2.0.txt in the APK.

These are pretrained acoustic/speech models. No trained SAFORA risk encoder is
bundled. Risk analysis uses the explicitly unvalidated DevelopmentClassifier.
English and Hindi are separate monolingual ASR models; no code-switching accuracy
or safety-detection accuracy is claimed.

Native runtimes: TensorFlow Lite 2.16.1 (Apache-2.0), Vosk Android 0.3.47
(Apache-2.0), JNA 5.13.0 (Apache-2.0 dual-license option). See LICENSES.md.
