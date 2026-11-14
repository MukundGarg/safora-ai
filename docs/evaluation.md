# Validation protocol
The supplied four recordings' peak values are observations only; recordings were
not provided. Unit tests cover negative maxima and temporal logic, not reproduce
female/male event counts from absent score sequences. No population accuracy claim.

Group source and session IDs across splits; training code rejects cross-split IDs.
Use independent validation to calibrate thresholds and freeze them before testing.
Event matching uses one prediction per annotated interval, allowing up to 1 s after
end. Report precision, recall, F1, delays, missed events and false alerts per negative
hour; inspect every miss. Test short/distant screams, traffic, crying, shouting,
women with diverse voices, and male screams without pitch-based suppression.
Report recording-level confusion counts, language and recording-condition slices.
No aggregate score is a substitute for condition-specific failure review.

For ASR, benchmark local multilingual Whisper tiny int8 and a locally available
smaller sherpa-onnx Hindi/English-capable model in separate fresh processes.
Whisper tiny is an integration candidate, not a measured winner. Do not compare
English-only Vosk to multilingual Whisper as equivalent coverage. benchmark/asr
manifest is a JSON array with id, wav, reference and language; <=12s PCM16 clips.
Include English, Hindi Devanagari and Hinglish/code-switching, quiet speech, overlap,
ASR hallucinations, TV dialogue, refusals followed by pressure, and consensual
conversations with matching vocabulary. WER whitespace tokenization is a baseline;
report CER and document Hindi normalization/transliteration policy separately.
Compare file bytes, CPU latency, real-time factor, peak RSS and quality, not size alone.

Noise helper supports deterministic SNR augmentation; it is not a measured acoustic
robustness result. Run each augmentation through the real model and evaluate per SNR.
Text training infrastructure is multi-label, uses local pretrained encoder, and
reports per-label 2x2 confusion matrices on validation/test. No training performed.
Use human reviewed, consented context with negative examples and speaker turns;
synthetic examples alone cannot qualify a safety classifier. Baseline can mishandle
negations, quotes, hypotheticals and speaker attribution; never enable high-level
escalation from it. Encoder calibration and safety validation remain required.

Android: JDK17, Android SDK35, Gradle8.9; run gradle test assembleDebug from android/.
Run on 2–3/4/6–8 GB physical devices and OEM/API matrix (26–35). Record incremental
PSS versus idle, p50/p95 latency, queue loss, temperature and battery per hour.
Use benchmarks/android.sh before/after 30–60min opt-in monitoring. Also test denied
and revoked permission, lock screen, call interruption, notification stop, background
restrictions, process death, low memory and hot device. Targets <100/200/300 MB
incremental AI memory are goals only. Android resource controller immediately disables optional speech under pressure
and releases Vosk in the serialized inference worker. No device resource target
is verified. Python Vosk alternative uses benchmarks/asr.py --backend vosk.
