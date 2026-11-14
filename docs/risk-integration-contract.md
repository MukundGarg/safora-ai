# Task 3 inference and Kotlin export contract

## Python API

```python
from safora.classifier.risk import load_classifier
from safora.classifier.text import Turn
with load_classifier() as classifier:  # Explicit unvalidated heuristic fallback.
    risk = classifier.classify(
        transcript.text,
        history=[Turn(previous_text, asr_uncertain=previous_uncertainty)],
        asr_uncertain=transcript.uncertain,
    )
```

Examples are API structure, not training examples. No model path loads the explicit
`UNVALIDATED_HEURISTIC`; an explicit invalid/corrupt model path raises rather than
silently substituting a different backend. `EncoderClassifier(local_directory)`
requires version2 research metadata, correct taxonomy order, actual bundle files/
checksums, training/data provenance and frozen per-label validation thresholds.
Integrity checks/attestations cannot establish weight provenance against forged
metadata; accept bundles only from reviewed training runs. No verified bundle is
currently present. Native transformer inference remains unexecuted in Task 3.

Input is current transcript plus oldest-to-newest bounded context and explicit ASR
uncertainty. `Turn.speaker` may carry an opaque **caller-supplied** ID; neither
backend infers identity/gender, or treats different IDs as validated role/consent
attribution. Plain context strings remain compatible and mark speakers unknown.
Context ASR uncertainty propagates; uncertain earlier turns cannot establish a
heuristic refusal. The legacy pipeline now retains that flag in memory. Context
expiry/continuity/generation belong to the caller; never pass history across a
Task 2 gap, restart, language switch, new session or opt-out. Final app wiring and
legacy scheduling/context race corrections belong to Task 4; these APIs alone do
not wire the independent controllers together.

Output `Risk` preserves original first three fields (`labels`, `uncertain`,
`source`), adds `scores` (None for heuristic), `version`, `reasons`. Labels obey
neutral exclusivity. Metadata/reasons contain fixed status codes, never transcripts.
ASR uncertainty cannot be upgraded to certainty. All heuristic and current research
encoder predictions remain uncertain/not safety validated. Encoder scores are
sigmoid label-pattern scores and are **not** probabilities of personal danger.
`neutral_or_uncertain` is derived from positive thresholds; its raw encoder score,
if reported, is diagnostic and not an independent decision.

Normalization uses NFC and whitespace collapse, preserving negation, punctuation,
Devanagari marks and ZWJ/ZWNJ. Surrogates, unexpected C0 controls and bidi override/
isolate characters produce uncertain neutral `invalid_input`. Current text is
bounded to2048 code points; context≤4turns/4096total. Truncation is explicit in
reasons. Large/incompatible histories are rejected as invalid. The encoder uses
current-first text then ordered context, max256tokens; tokenizer truncation may
still lose late current/context content, and no linguistic completeness is claimed.
The model remains uncertain regardless. The heuristic is stateless; caller owns
transient history. `close()` is idempotent, drops model/tokenizer references and
rejects later inference. Strings/native allocator memory cannot be guaranteed
securely erased. No transcript/file/network logging is added.

Give each encoder one background worker owner; do not share/close it concurrently
with inference. Run risk work downstream of the independent Task 2 speech queue,
never in Task 1 capture or its inference lock. CPU transformer memory/latency are
not measured; no low-RAM suitability is established merely by max256tokens.
Prefer a genuinely compatible compact local multilingual base after measuring it.
Use the explicit heuristic fallback or disable conversation analysis under memory
pressure, with truthful provenance/coverage status. Neither backend sends SOS,
location, contacts or alerts; manual SOS remains the host app's independent path.

## Guarded ONNX path (not executed on real weights)

```bash
PYTHONPATH=python python scripts/export_encoder.py /local/research-risk-v2 \
  /local/research-risk-v2-onnx --probes /local/authorized-parity-probes.json
```

Requires a reviewed real fine-tuned version2 bundle, local torch/transformers/onnx/
onnxruntime, and JSON list of3..100 authorized text probes≤2048 code points. Do not
use an untouched test set as export-tuning probes. Missing/untrained bundle or
missing dependencies blocks export; no arbitrary base/random weights are saved.
Exporter writes to temporary staging and copies a new directory only after parity;
failed checks do not publish the destination. This path supports standard
HF-compatible sequence-classification fast-tokenizer architectures exporting integer
token inputs/logits at opset17, with optional `token_type_ids` included when needed.
Unsupported architectures/exporters fail explicitly; compatibility is unverified
until run with the selected real checkpoint.

Output: `encoder.float.onnx`, `encoder.int8.onnx` (dynamic signed-int8 weight
quantization), tokenizer files, `parity.json`, `safora_metadata.json`. Metadata
includes schema/taxonomy/model version, label order, frozen validation thresholds,
max tokens, input names/int64 type, original training provenance and exported-file
SHA256s. The artifact is `RESEARCH_ONNX_EXPORT`, never a production safety model.
ONNX export validates float-vs-PyTorch absolute logit tolerance1e-4 and int8-vs-
PyTorch tolerance0.1, rtol0, finite matching shapes, plus identical frozen-threshold
decisions on the supplied probes. These are provisional export tolerances, not
accuracy/robustness evidence. Probe coverage and device numerical parity need
independent review. Quantization can change decisions on other inputs.

## Android work reserved for Task 4

No Android dependencies/UI/core changes are introduced in Task 3. Existing Kotlin
`ConversationClassifier.classify(Recognition, List<String>): ConversationRisk`
retains only labels/uncertainty; it cannot express scores, provenance, versions,
per-turn ASR uncertainty or generations. Existing `DevelopmentClassifier` remains
its earlier separate unvalidated implementation; Python rule improvements are
**not** silently present in Kotlin. Task 4 must port/test the chosen rules or
implement an isolated adapter for a real exported bundle, carrying the additional
metadata alongside or through a reviewed core contract change.

A native adapter must implement identical bounded NFC/whitespace/context encoding,
matching tokenizer.json behavior, special tokens/attention mask/padding/truncation,
int64 tensors named by export metadata, stable sigmoid, per-label thresholding and
neutral exclusivity. Verify model/tokenizer/metadata checksums, run numerical and
label parity against Python on authorized English/Hindi/Hinglish/hard-negative
probes, close tokenizer/session resources off capture and measure real CPU/PSS.
Do not add an Android runtime solely because an ONNX file exists: inspect current
app dependencies and device ABI/API/operator support first. Final decision engine,
queue/event wiring, microphone/device tests and privacy lifecycle are Task 4.
