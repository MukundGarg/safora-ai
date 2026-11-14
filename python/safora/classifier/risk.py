from dataclasses import dataclass
import re
from .text import prepare, encoder_text, Turn

LABELS = ['sexual_coercion','explicit_threat','refusal_ignored','unwanted_sexual_advance',
          'secrecy_pressure','isolation_pressure','distress_expression','neutral_or_uncertain']
POSITIVE_LABELS = LABELS[:-1]
VERSION = 'risk-v2'


def consistent_labels(labels):
    if not isinstance(labels,(list,tuple,set,frozenset)) or any(not isinstance(l,str) for l in labels):
        raise ValueError('Invalid risk labels')
    values = set(labels)
    if not values or not values <= set(LABELS): raise ValueError('Invalid risk labels')
    if 'neutral_or_uncertain' in values and len(values) != 1:
        raise ValueError('Neutral must be exclusive')
    return values


@dataclass
class Risk:
    labels: set[str]
    uncertain: bool = True
    source: str = 'UNVALIDATED_HEURISTIC'
    scores: dict[str, float] | None = None
    version: str = VERSION
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        self.labels = consistent_labels(self.labels)


# Deliberately narrow lexical evidence. None of these rules establish speakers,
# consent, crime, or safety; absent matches never mean a conversation is safe.
REFUSAL = r"\b(?:i said no|i don't want (?:this|to)|stop touching|leave me alone|mana kiya|mujhe mat chhuo)\b|(?:मना किया|छूना बंद|मुझे छोड़|मुझे मत छुओ)"
PRESSURE = r"\b(?:do it anyway|you have no choice|i don't care|phir bhi|karna padega)\b|(?:फिर भी|कोई चारा नहीं|करना पड़ेगा)"
THREAT = r"\b(?:i (?:will|gonna) (?:hurt|kill) you|jaan se maar(?: dunga)?|maar dunga)\b|(?:जान से मार दूंगा|जान से मार दूँगा|तुम्हें मार दूंगा)"
SEXUAL = r"\b(?:sex|touch you|touching|kiss|kapde utaro)\b|(?:सेक्स|चूम|कपड़े उतार|छूना)"
DISTRESS = r"\b(?:help me|i am scared|i'm scared|mujhe dar lag raha|bachao)\b|(?:बचाओ|मुझे डर लग रहा|मेरी मदद करो)"
SECRECY = r"\b(?:don't tell|do not tell|kisi ko mat batana)\b|किसी को मत बताना"
ISOLATION = r"\b(?:come alone|be alone|somewhere alone|akele aao)\b|अकेले आओ"
REPORTING = r"\b(?:news|reported|reporter|headline|movie|film|novel|fiction|screenplay|roleplay|quoted|he said|she said|they said)\b|(?:समाचार|खबर में|फ़िल्म|फिल्म|उसने कहा)"
NEGATED_THREAT = r"\b(?:i (?:will not|won't|would not|wouldn't) (?:hurt|kill) you|i (?:do not|don't) (?:want|intend|plan) to (?:hurt|kill) you|not saying|never said|don't say|i won't do it anyway|i will not do it anyway)\b|(?:नहीं मार|nahi maar|nahin maar)"


def _has(pattern, text): return re.search(pattern, text) is not None


def _direct(text):
    # Exclude explicitly quoted spans, preserving apostrophes in contractions.
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|(?<!\w)\'[^\'\n]*\'(?!\w)', ' ', text)
    clauses = re.split(r'[.!?;।]+', text)
    return ' '.join(c for c in clauses if not _has(REPORTING, c) and not _has(NEGATED_THREAT, c))


class ContextBaseline:
    """Stateless, bounded, UNVALIDATED_HEURISTIC; no persistent transcript state."""
    def __init__(self): self.closed = False

    def classify(self, text, history=(), asr_uncertain=False):
        if self.closed: raise RuntimeError('Classifier is closed')
        try: current, turns, reasons = prepare(text, history, asr_uncertain)
        except ValueError:
            return Risk({'neutral_or_uncertain'}, reasons=('invalid_input',))
        t = _direct(current.casefold().replace('’', "'"))
        # Uncertain ASR context must not establish an earlier refusal. Turns are
        # ordered but speaker identity/continuity is still unknown.
        context = ' '.join(_direct(turn.text.casefold().replace('’', "'")) for turn in turns if not turn.asr_uncertain)
        if turns and _has(r'\b(?:i will stop|i respect your no|okay i will not)\b|मैं रुक जाऊंगा', turns[-1].text.casefold()):
            context = ''
        refusal, pressure, sexual = _has(REFUSAL,t), _has(PRESSURE,t), _has(SEXUAL,t)
        threat = _has(THREAT,t)
        labels = set()
        if threat: labels.add('explicit_threat')
        if refusal or _has(DISTRESS,t): labels.add('distress_expression')
        if pressure and (refusal or _has(REFUSAL,context)): labels.add('refusal_ignored')
        if sexual and (threat or pressure): labels.add('sexual_coercion')
        if sexual and refusal: labels.add('unwanted_sexual_advance')
        if _has(SECRECY,t) and (threat or pressure): labels.add('secrecy_pressure')
        if _has(ISOLATION,t) and (threat or pressure): labels.add('isolation_pressure')
        return Risk(labels or {'neutral_or_uncertain'}, reasons=(*reasons,'unvalidated_rules','speaker_and_intent_not_established'))

    def close(self): self.closed = True
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class EncoderClassifier:
    """Verified local training bundle only; scores are not calibrated danger probabilities."""
    def __init__(self, path, threshold=None):
        from .bundle import validate_bundle
        meta = validate_bundle(path)
        if threshold is not None: raise ValueError('Use frozen per-label validation thresholds from metadata')
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.meta = meta
        self.tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_safetensors=True).cpu().eval()
        if self.model.config.num_labels != len(LABELS): raise ValueError('Encoder output shape mismatch')
        self.closed = False

    def classify(self, text, history=(), asr_uncertain=False):
        if self.closed: raise RuntimeError('Classifier is closed')
        try: current, turns, reasons = prepare(text,history,asr_uncertain)
        except ValueError:
            return Risk({'neutral_or_uncertain'},source='FINETUNED_ENCODER',reasons=('invalid_input',))
        if not current:
            return Risk({'neutral_or_uncertain'},source='FINETUNED_ENCODER',reasons=(*reasons,'empty_input'))
        inputs = self.tokenizer(encoder_text(current,turns),return_tensors='pt',truncation=True,
                                max_length=self.meta['max_tokens'],padding=False)
        with self.torch.inference_mode(): values = self.model(**inputs).logits.sigmoid()[0].tolist()
        from safora.evaluation.risk import select_labels, validate_scores
        validate_scores([values])
        labels = select_labels(values,self.meta['thresholds'])
        return Risk(labels,True,'FINETUNED_ENCODER',dict(zip(LABELS,values)),self.meta['model_version'],
                    (*reasons,'not_safety_validated','speaker_and_intent_not_established'))

    def close(self): self.model = self.tokenizer = None; self.closed = True
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def load_classifier(path=None):
    """Explicit unavailable-model fallback; does not disguise missing/corrupt weights."""
    if path is None: return ContextBaseline()
    return EncoderClassifier(path)
