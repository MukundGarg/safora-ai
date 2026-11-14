"""Known score/label arithmetic fixtures, not measured classifier performance."""
import json,tempfile,unittest
from pathlib import Path
from safora.classifier.risk import LABELS,POSITIVE_LABELS
from safora.evaluation.risk import calibrate,evaluate,metrics,select_labels,evaluate_test_once


def score(value):return [0,value,0,0,0,0,0,0]
def examples(split='validation'):
    return [dict(labels=['explicit_threat'],split=split,language='en'),
            dict(labels=['neutral_or_uncertain'],split=split,language='hi',hard_negative=True)]


class RiskEvaluationTests(unittest.TestCase):
    def test_validation_only_thresholds_ties_and_unsupported_labels(self):
        result=calibrate(examples(),[score(.8),score(.2)])
        self.assertEqual(result['thresholds']['explicit_threat'],.8)
        self.assertIsNone(result['thresholds']['sexual_coercion']);self.assertFalse(result['probability_calibrated'])
        for split in ('train','test'):
            with self.assertRaises(ValueError):calibrate(examples(split),[score(.8),score(.2)])
    def test_consistent_neutral_decision(self):
        thresholds=dict.fromkeys(POSITIVE_LABELS,None)
        self.assertEqual(select_labels([1]*8,thresholds),{'neutral_or_uncertain'})
        thresholds['explicit_threat']=.8
        self.assertEqual(select_labels(score(.8),thresholds),{'explicit_threat'})
    def test_metrics_confusion_macro_micro_and_unavailable(self):
        result=metrics([{'explicit_threat'},{'neutral_or_uncertain'}],[{'explicit_threat'},{'explicit_threat'}])
        self.assertEqual(result['per_class']['explicit_threat']['confusion_matrix'],[[0,1],[0,1]])
        self.assertAlmostEqual(result['micro_f1'],2/3);self.assertAlmostEqual(result['macro_f1'],2/3)
        self.assertIsNone(result['per_class']['sexual_coercion']['f1'])
        self.assertIsNone(metrics([],[])['micro_f1']);self.assertIsNone(metrics([],[])['macro_f1'])
    def test_empty_or_bad_shapes_nan_and_thresholds_rejected(self):
        for values in [[],[float('nan')]*8,[1.1]*8,[True]*8]:
            with self.assertRaises(ValueError):select_labels(values,dict.fromkeys(POSITIVE_LABELS,.5))
        with self.assertRaises(ValueError):metrics([{'neutral_or_uncertain'}],[])
        with self.assertRaises(ValueError):calibrate(examples(),[score(.5)])
    def test_hard_negative_error_analysis_and_language_slices_no_text(self):
        rows=examples();rows[1]['text']='private transcript'
        thresholds=dict.fromkeys(POSITIVE_LABELS,None);thresholds['explicit_threat']=.5
        report=evaluate(rows,[score(.7),score(.6)],thresholds)
        self.assertEqual(report['hard_negatives']['false_positive_count'],1)
        self.assertNotIn('private transcript',json.dumps(report));self.assertIn('hi',report['by_language'])
    def test_heldout_exclusive_once_and_frozen_thresholds(self):
        with tempfile.TemporaryDirectory() as temp:
            thresholds=calibrate(examples(),[score(.8),score(.2)])['thresholds'];calls=[]
            def predict(rows):calls.append(True);return [score(.9),score(.1)]
            evaluate_test_once(examples('test'),predict,thresholds,temp)
            with self.assertRaises(FileExistsError):evaluate_test_once(examples('test'),predict,thresholds,temp)
            self.assertEqual(len(calls),1)
            self.assertEqual(json.loads((Path(temp)/'heldout-evaluation.json').read_text())['status'],'complete')
    def test_failed_heldout_run_is_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            def fail(rows):raise RuntimeError('fixture failure')
            thresholds=dict.fromkeys(POSITIVE_LABELS,None)
            with self.assertRaises(RuntimeError):evaluate_test_once(examples('test'),fail,thresholds,temp)
            with self.assertRaises(FileExistsError):evaluate_test_once(examples('test'),fail,thresholds,temp)

    def test_score_cli_binds_dataset_model_and_excludes_test_from_calibration(self):
        import sys
        from unittest.mock import patch
        from safora.training.data import fingerprint
        from safora.evaluation.risk import main
        from test_risk_data import rows
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=rows();data[1]['labels']=['explicit_threat']
            # One validation category has only positives: threshold unavailable.
            path=root/'data.jsonl';path.write_text('\n'.join(json.dumps(r) for r in data)+'\n')
            supplied=dict(model_sha256='a'*64,dataset_sha256=fingerprint(data),
                          rows=[dict(row_sha256=fingerprint(data[1]),scores=score(.8))])
            scores=root/'scores.json';scores.write_text(json.dumps(supplied));cal=root/'calibration.json'
            with patch.object(sys,'argv',['risk','--data',str(path),'--scores',str(scores),'--split','validation','--output',str(cal)]):
                self.assertEqual(main(),0)
            self.assertIsNone(json.loads(cal.read_text())['thresholds']['explicit_threat'])
            supplied['rows']=[dict(row_sha256=fingerprint(data[2]),scores=score(.2))];scores.write_text(json.dumps(supplied))
            argv=['risk','--data',str(path),'--scores',str(scores),'--split','test','--calibration',str(cal),'--output',str(root/'evaluation')]
            with patch.object(sys,'argv',argv):self.assertEqual(main(),0)
            with patch.object(sys,'argv',argv):self.assertEqual(main(),2)
