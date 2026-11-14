"""Bounded Unicode/context handling shared by training and inference; no logging."""
from dataclasses import dataclass, field
import unicodedata

MAX_TEXT = 2048
MAX_TURNS = 4
MAX_CONTEXT = 4096


@dataclass(frozen=True)
class Turn:
    text: str = field(repr=False)
    asr_uncertain: bool = False
    speaker: str | None = None  # Caller-declared opaque ID only, never inferred.


def normalize(text):
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    # Reject surrogates, bidi overrides/isolates and non-whitespace controls rather
    # than silently repairing meaning. Preserve ZWJ/ZWNJ and Devanagari marks.
    for char in text:
        if (unicodedata.category(char) == 'Cs' or
                unicodedata.category(char) == 'Cc' and char not in '\t\n\r' or
                char in '\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069'):
            raise ValueError('Unsupported Unicode control')
    return ' '.join(unicodedata.normalize('NFC', text).split())


def prepare(text, history=(), asr_uncertain=False):
    if not isinstance(asr_uncertain, bool): raise ValueError('ASR uncertainty must be boolean')
    if not isinstance(history, (list, tuple)) or len(history) > 100:
        raise ValueError('Context must be a bounded sequence')
    if not isinstance(text,str): raise ValueError('Text must be a string')
    truncated = len(text) > MAX_TEXT or len(history) > MAX_TURNS
    current = normalize(text[:MAX_TEXT])[:MAX_TEXT]
    turns = []
    for item in history[-MAX_TURNS:]:
        turn = Turn(item) if isinstance(item, str) else item
        if not isinstance(turn, Turn) or not isinstance(turn.asr_uncertain, bool):
            raise ValueError('Invalid context turn')
        if turn.speaker is not None and (not isinstance(turn.speaker, str) or len(turn.speaker) > 64):
            raise ValueError('Invalid declared speaker')
        if not isinstance(turn.text,str): raise ValueError('Invalid context text')
        truncated |= len(turn.text) > MAX_TEXT
        clean = normalize(turn.text[:MAX_TEXT])
        turns.append(Turn(clean[:MAX_TEXT], turn.asr_uncertain, turn.speaker))
    while sum(len(t.text) for t in turns) > MAX_CONTEXT:
        turns.pop(0); truncated = True
    reasons = []
    if asr_uncertain or any(t.asr_uncertain for t in turns): reasons.append('asr_uncertain')
    if truncated: reasons.append('context_or_text_truncated')
    if turns and any(t.speaker is None for t in turns): reasons.append('speaker_unknown')
    return current, turns, tuple(reasons)


def encoder_text(text, history=()):
    current, turns, _ = prepare(text, history)
    # Explicit positions, no guessed speaker roles. Keep refusal/negation verbatim.
    return '\n'.join([f'Current: {current}', *(f'Context: {t.text}' for t in turns)])
