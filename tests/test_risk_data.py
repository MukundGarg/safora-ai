"""Fabricated schema/parser fixtures only, never used to train a model."""
import copy,json,tempfile,unittest
from pathlib import Path
from safora.training.data import validate_rows,load_jsonl,dataset_metadata
from safora.training.train import preflight


def rows():
    return [dict(text=f'unit test row {i}',context=[],labels=['neutral_or_uncertain'],language='en',
        source_id=f'source-{i}',session_id=f'session-{i}',split=split,license='TEST-ONLY-NOT-A-DATASET',
        provenance=dict(verified=True,origin='unit fixture only',reviewed_by='test stub',permission_basis='schema test'))
        for i,split in enumerate(('train','validation','test'))]


class RiskDataTests(unittest.TestCase):
    def test_valid_schema_and_stable_metadata_without_text(self):
        data=rows();self.assertEqual(validate_rows(data),dict(train=1,validation=1,test=1))
        self.assertEqual(dataset_metadata(data),dataset_metadata(copy.deepcopy(data)))
        self.assertNotIn('unit test row',json.dumps(dataset_metadata(data)))
    def test_source_session_group_leakage(self):
        for field in ('source_id','session_id','group_id'):
            data=rows();data[0][field]='shared';data[1][field]='shared'
            with self.assertRaisesRegex(ValueError,'leakage'):validate_rows(data)
    def test_duplicate_conflicting_or_normalized_text(self):
        for labels in [['neutral_or_uncertain'],['explicit_threat']]:
            data=rows();data[1]['text']=data[0]['text'].upper();data[1]['labels']=labels
            with self.assertRaises(ValueError):validate_rows(data)
        data=rows();data.append(copy.deepcopy(data[0]))
        with self.assertRaises(ValueError):validate_rows(data)
    def test_missing_license_permission_and_provenance(self):
        for update in [dict(license=''),dict(provenance={}),dict(provenance=dict(verified=False)),dict(language='made-up')]:
            data=rows();data[0].update(update)
            with self.assertRaises(ValueError):validate_rows(data)
    def test_invalid_labels_annotations_and_hard_negative(self):
        for update in [dict(labels=[]),dict(labels=[{}]),dict(labels=['unknown']),dict(labels=['explicit_threat','neutral_or_uncertain']),
                       dict(labels=['explicit_threat'],hard_negative=True),
                       dict(annotations=[dict(annotator_id='reviewer',status='reviewed',labels=['explicit_threat'])])]:
            data=rows();data[0].update(update)
            with self.assertRaises(ValueError):validate_rows(data)
    def test_text_context_unicode_and_split_bounds(self):
        for update in [dict(text='\ud800'),dict(text='x'*2049),dict(context=['x']*5),dict(context='x'),
                       dict(split='unassigned'),dict(text=''),dict(session_id='')]:
            data=rows();data[0].update(update)
            with self.assertRaises(ValueError):validate_rows(data)
        with self.assertRaises(ValueError):validate_rows(rows()[:2])
    def test_strict_jsonl_duplicate_keys_invalid_json_and_empty_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.jsonl'
            for text in ['{"text":"one","text":"two"}\n','not json\n','{"extra": NaN}\n','\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):load_jsonl(path)
            path.write_text('\n'.join(json.dumps(r) for r in rows())+'\n');self.assertEqual(len(load_jsonl(path)),3)
    def test_preflight_blocks_before_imports_or_model_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'missing.jsonl'
            with self.assertRaisesRegex(ValueError,'no verified dataset'):preflight(path,Path(temp)/'base')
            path.write_text('\n'.join(json.dumps(r) for r in rows())+'\n')
            with self.assertRaisesRegex(ValueError,'local base'):preflight(path,Path(temp)/'base')
