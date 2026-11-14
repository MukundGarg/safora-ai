from dataclasses import dataclass
import re
LABELS=['sexual_coercion','explicit_threat','refusal_ignored','unwanted_sexual_advance','secrecy_pressure','isolation_pressure','distress_expression','neutral_or_uncertain']
@dataclass
class Risk:
    labels: set[str]
    uncertain: bool = True
    source: str = "UNVALIDATED_HEURISTIC"

class ContextBaseline:
    """Development baseline only; never licenses automatic external escalation.
    No speaker attribution: refusal_ignored requires explicit continuation text.
    """
    def classify(self,text,history=(),asr_uncertain=False):
        t=text.casefold();context=" ".join(history).casefold();labels=set()
        threat=bool(re.search(r"(i (will|gonna) (hurt|kill)|jaan se maar|maar dunga)",t))
        refusal=bool(re.search(r"(i said no|stop touching|leave me alone|mujhe mat|mana kiya)",t))
        pressure=bool(re.search(r"(do it anyway|you have no choice|i don't care|phir bhi)",t))
        sexual=bool(re.search(r"(sex|touch you|touching|kiss|kapde)",t))
        if threat: labels.add('explicit_threat')
        if refusal: labels.add('distress_expression')
        if pressure and (refusal or re.search(r"(said no|stop|mana kiya)",context)): labels.add('refusal_ignored')
        if sexual and (threat or pressure): labels.add('sexual_coercion')
        if sexual and refusal: labels.add('unwanted_sexual_advance')
        if 'alone' in t and (threat or pressure): labels.add('isolation_pressure')
        if ("don't tell" in t or 'kisi ko mat' in t) and (threat or pressure): labels.add('secrecy_pressure')
        return Risk(labels or {'neutral_or_uncertain'},True)

class EncoderClassifier:
    """Load a locally fine-tuned multi-label encoder, never generic base weights."""
    def __init__(self,path,threshold=.5):
        import json
        from pathlib import Path
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        meta=json.loads((Path(path)/'safora_metadata.json').read_text())
        if meta['labels']!=LABELS: raise ValueError('Label order mismatch')
        self.torch=torch;self.threshold=threshold;self.validated=bool(meta.get("validated_for_emergency_escalation",False))
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).eval()
    def classify(self,text,history=(),asr_uncertain=False):
        inputs=self.tokenizer(' [SEP] '.join([*history,text]),return_tensors='pt',truncation=True,max_length=256)
        with self.torch.inference_mode(): scores=self.model(**inputs).logits.sigmoid()[0].tolist()
        labels={l for l,s in zip(LABELS,scores) if s>=self.threshold}
        return Risk(labels or {'neutral_or_uncertain'},asr_uncertain or not self.validated,'FINETUNED_ENCODER')
