"""Bundle integrity and inference plumbing with fake tensors, not real model weights."""
import importlib.util,json,tempfile,unittest,sys,types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from safora.classifier.bundle import validate_bundle,file_hashes
from safora.classifier.risk import LABELS,POSITIVE_LABELS,VERSION,EncoderClassifier
from safora.training.data import dataset_metadata
from test_risk_data import rows


def fixture(root):
    # Invalid as real model/tokenizer: only integrity schema fixtures.
    for name in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json'):
        (root/name).write_text('test-only placeholder, not weights')
    metadata=dict(schema_version=2,labels=LABELS,taxonomy_version=VERSION,
        artifact_kind='RESEARCH_FINETUNED_ENCODER',training_completed=True,
        validated_for_emergency_escalation=False,model_version='unit-test-only',
        dataset=dataset_metadata(rows()),thresholds=dict.fromkeys(POSITIVE_LABELS,.5),
        threshold_source='validation',max_tokens=256,file_sha256=file_hashes(root))
    (root/'safora_metadata.json').write_text(json.dumps(metadata));return metadata


class RiskBundleTests(unittest.TestCase):
    def test_missing_generic_untrained_and_wrong_label_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(ValueError):validate_bundle(root)
            meta=fixture(root)
            for update in [dict(training_completed=False),dict(labels=list(reversed(LABELS))),
                           dict(artifact_kind='GENERIC_BASE_ENCODER'),dict(threshold_source='test'),
                           dict(validated_for_emergency_escalation=True),dict(dataset={})]:
                (root/'safora_metadata.json').write_text(json.dumps({**meta,**update}))
                with self.assertRaises(ValueError):validate_bundle(root)
    def test_hash_corruption_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);meta=fixture(root);validate_bundle(root)
            (root/'model.safetensors').write_text('changed')
            with self.assertRaises(ValueError):validate_bundle(root)
            meta=fixture(root);meta['file_sha256']['../escape']='x'
            (root/'safora_metadata.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):validate_bundle(root)
    def test_encoder_api_local_loading_uncertainty_scores_exclusivity_cleanup(self):
        class Values:
            def sigmoid(self):return self
            def __getitem__(self,index):return self
            def tolist(self):return [0,.9,0,0,0,0,0,.99]
        class Model:
            config=types.SimpleNamespace(num_labels=8)
            def cpu(self):return self
            def eval(self):return self
            def __call__(self,**kw):return types.SimpleNamespace(logits=Values())
        calls=[]
        class Loader:
            @staticmethod
            def from_pretrained(path,**kw):
                calls.append(kw);return Model()
        class Tokenizer:
            @staticmethod
            def from_pretrained(path,**kw):calls.append(kw);return lambda text,**kwargs:{}
        torch=types.SimpleNamespace(inference_mode=nullcontext)
        transformers=types.SimpleNamespace(AutoTokenizer=Tokenizer,AutoModelForSequenceClassification=Loader)
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,torch=torch,transformers=transformers):
            fixture(Path(temp));model=EncoderClassifier(temp)
            result=model.classify('authored fixture',asr_uncertain=True)
            self.assertEqual(result.labels,{'explicit_threat'});self.assertTrue(result.uncertain)
            self.assertIn('asr_uncertain',result.reasons);self.assertEqual(result.scores['explicit_threat'],.9)
            self.assertEqual(result.source,'FINETUNED_ENCODER');self.assertEqual(result.version,'unit-test-only')
            self.assertTrue(all(c['local_files_only'] and not c['trust_remote_code'] for c in calls))
            self.assertEqual(model.classify('').labels,{'neutral_or_uncertain'})
            self.assertIn('invalid_input',model.classify('\ud800').reasons)
            model.close();model.close();self.assertIsNone(model.model)
            with self.assertRaises(RuntimeError):model.classify('fixture')
    def test_export_guard_creates_no_output_for_untrained_directory(self):
        script=Path(__file__).resolve().parents[1]/'scripts/export_encoder.py'
        spec=importlib.util.spec_from_file_location('risk_export_fixture',script)
        exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'production-model'
            with self.assertRaises(ValueError):exporter.export(temp,output,Path(temp)/'absent-probes.json')
            self.assertFalse(output.exists())
