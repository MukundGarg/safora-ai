# Android native runtime notices

This source selects independently versioned runtime artifacts, not new trained models:

- `com.google.ai.edge.litert:litert:1.4.2` (classic `org.tensorflow.lite.Interpreter` API).
  Apache-2.0 per its published Google Maven POM:
  https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/1.4.2/litert-1.4.2.pom
- `com.alphacephei:vosk-android:0.3.75`, Apache-2.0 per Maven Central POM:
  https://repo.maven.apache.org/maven2/com/alphacephei/vosk-android/0.3.75/vosk-android-0.3.75.pom
- `net.java.dev.jna:jna:5.18.1@aar`, dual LGPL-2.1-or-later / Apache-2.0;
  Apache-2.0 is selected for this distribution, per the published POM:
  https://repo.maven.apache.org/maven2/net/java/dev/jna/jna/5.18.1/jna-5.18.1.pom

Apache-2.0 license text is included as MODEL-LICENSE-APACHE-2.0.txt. Preserve
upstream copyright/NOTICE obligations and review the complete resolved dependency
inventory before distribution. Direct AAR hashes and measured ELF alignment are
recorded in docs/verification/production/native-candidates.json. Static alignment
and compilation do not establish actual 16KB-page device compatibility, performance
or model accuracy. YAMNet and Vosk model artifacts retain their separate notices.
