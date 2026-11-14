# Local conversation-risk data and evaluation contract

There is **no training dataset in this repository**. Test fixture text, fake model
bytes and known-score metric fixtures are software tests only; never train on them
or report their results as learned accuracy. No corpus is downloaded by these tools.

## JSONL schema (one object per line)

Machine-readable row schema: `configs/risk-data.schema.json`. The Python validator
additionally enforces cross-row provenance/annotation constraints.

| Field | Required representation |
|---|---|
| text | Nonempty string ≤2048 code points; Unicode rules match inference |
| context | Ordered oldest-to-newest list of ≤4 strings, each ≤2048, total≤4096 |
| labels | Nonempty unique list from risk-v2 taxonomy; neutral exclusive |
| language | `en`, `hi`, `romanized_hi`, `hinglish`, or explicit `unknown` |
| source_id / session_id | Nonempty opaque strings; keep groups wholly in one split |
| split | `train`, `validation`, or `test`; all three nonempty |
| license | Explicit nonempty license identifier or documented consent permission reference |
| provenance | Object: `verified: true`, nonempty `origin`, `reviewed_by`, `permission_basis` |
| id / group_id | Optional nonempty opaque strings; IDs unique, groups split-isolated |
| annotations | Optional list of objects with `annotator_id`, `status: reviewed|adjudicated`, `labels` exactly agreeing with final labels |
| hard_negative | Optional boolean, true only with sole neutral label |

`verified` is the supplier's permission-review attestation, not automatic legal or
accuracy validation. Record consent scope/withdrawal procedures or actual permitted
license terms in the referenced permission record. A name or URL alone is not proof
of consent or compatible usage rights. Do not fabricate license metadata.

Strict UTF-8 JSONL parsing rejects blank/malformed lines and duplicate object keys.
Dataset limit64MiB/100000 rows, line limit65536 code points. Missing license/review,
unresolved/inconsistent annotation, invalid labels/context/Unicode, duplicates and
source/session/group split leakage reject the run **before dependency/model loading**.
Normalized full-text/context duplicates conflict even within one split. Identical
NFC/casefold/whitespace-normalized current text cannot cross splits, conservatively
including innocuous phrases; group or remove these before splitting. Near-duplicate
paraphrase/translation leakage still needs human/source review. Numeric/aggregate
metrics cannot establish independence.

## Training, thresholds and a single held-out run

```bash
# Provision dependencies separately; all model operations below are local/offline.
python -m safora.training.train --data /local/permission-reviewed.jsonl \
  --base /local/compatible-base --check-data --status-output /local/preflight.json
python -m safora.training.train --data /local/permission-reviewed.jsonl \
  --base /local/compatible-base --output /local/research-risk-v2 \
  --epochs 3 --batch 4 --threads 2
```

Requires local HF-compatible config/safetensors and a fast tokenizer; CPU PyTorch/
Transformers are optional setup dependencies (`.[training]`). Remote code is
disabled, local-files-only is enforced and HF offline flags are set. New task heads
may be initialized for supervised optimization only when legitimate reviewed data
exists. Fresh output directories prevent silent overwrites. No output model is
created on missing-data/base/dependency preflight failure. Failed later training/
evaluation may leave an incomplete research directory, rejected by the loader
without complete metadata/checksums; review failures instead of publishing it.
Trainer fixes seed42, uses deterministic CPU algorithms, bounded batch1..32,
threads1..4, AdamW2e-5, gradient clipping1, ≤20epochs, max256tokens. Runtime text is
current-first then chronological context, so older context cannot truncate away
the current utterance. A long current utterance may still be token-truncated.

All eight logits remain compatible. Positive label thresholds are selected on
validation only, fixed grid0.05..0.95 in0.05 steps, maximizing per-label F1, tie to
higher threshold. A label without both validation positives and negatives has
`null` threshold and is disabled/unavailable, never silently assigned0.5. Neutral
is derived from absence of selected positives, not separately thresholded. These
are **decision thresholds**, not calibrated probabilities or verified safety
thresholds. Re-evaluate under supported languages/conditions; no claim of Platt/
isotonic/temperature probability calibration is made.

After thresholds freeze, trainer predicts the independent test set once, writing
an exclusive `heldout-evaluation.json` receipt. Reuse of the same output directory
is rejected; even failed evaluations retain the reservation. This local guard does
not prevent a person copying data/model elsewhere or repeatedly selecting models
using test results. Maintain an access-controlled untouched test release, record
all runs and create a new independent test corpus after contamination. Do not tune
thresholds, prompts/rules or models against test feedback.

Reports include per-class precision/recall/F1 and `[[TN,FP],[FN,TP]]`, positive-only
macro/micro F1, support counts, per-language slices and hard-negative false-positive
counts/hashed row IDs. No positive reference support means recall/F1 unavailable
(`null`), not fabricated0% or100%. Empty prediction denominator means precision
unavailable; misses still yield F1=0 when positive references exist. Unsupported
classes are excluded from macro average and their count is explicit. Neutral has
its own class metrics but is excluded from positive macro/micro. Persistent reports
contain hashes/counts/status, not reference/hypothesis transcripts.

For independent offline score-file arithmetic/reproduction:

```bash
python -m safora.evaluation.risk --data /local/data.jsonl --scores /local/validation-scores.json \
  --split validation --output /local/frozen-calibration.json
python -m safora.evaluation.risk --data /local/data.jsonl --scores /local/test-scores.json \
  --split test --calibration /local/frozen-calibration.json --output /local/heldout-once
```

Score JSON: object with `model_sha256` (actual checkpoint SHA256), `dataset_sha256`
(`training.data.fingerprint(all_rows)`), and `rows` list, one per selected row:
`{"row_sha256": fingerprint(row), "scores": [eight finite 0..1 values in LABELS order]}`.
Do not place text in IDs. The tool verifies split/row/model/dataset binding, freezes
validation output and uses an exclusive test receipt. It cannot prove authenticity
of supplied scores or labels. Authored numeric fixtures are not model accuracy.

## Obtaining real data next

Define a consent/compatible-license collection protocol with withdrawal and secure
access, bilingual annotation guidelines and disagreement adjudication. Collect
English/Hindi/romanized Hindi/Hinglish with bounded conversational turns, quiet/
noisy ASR errors and hard negatives (consensual, quoted, negated, news/fiction).
Have independent reviewers annotate evidence/ambiguity using this taxonomy, not
crime/gender. Group all speakers/sessions/sources and derivatives before splitting;
hold the test release outside iterative rule/model development. Review minority
labels and language/condition support. Only then run the trainer, freeze validation
thresholds and evaluate once. Training data should remain private outside Git.
