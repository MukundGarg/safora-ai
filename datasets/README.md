No recordings or training data are included. Collect consented, properly licensed data.
JSONL risk schema: text, context[], labels[], source_id, session_id, split, license,
language, recording_condition. Assign all utterances from one original source or
session to the same split. Check both IDs with validate_splits before training.
Acoustic calibration JSON is an array of {id, split:"validation", duration,
scores:[[window_end_seconds,scream_score]], events:[[onset,end]], source_id,session_id}.
Keep final test recordings untouched by threshold choice. Annotate event boundaries,
background TV/media, traffic, distance, noise, voice diversity, consent and withdrawals.
Never infer speaker gender from pitch. Report women-centric consented cohorts separately.
