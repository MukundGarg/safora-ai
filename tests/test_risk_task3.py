"""Authored rule-contract fixtures only: not a licensed or representative labeled corpus."""
import unittest
from safora.classifier.risk import ContextBaseline, Risk, LABELS, load_classifier
from safora.classifier.text import Turn, normalize, prepare, encoder_text, MAX_CONTEXT


class RiskRulesTests(unittest.TestCase):
    def setUp(self): self.classifier=ContextBaseline()
    def positive(self,text,label,history=()):
        result=self.classifier.classify(text,history)
        self.assertIn(label,result.labels);self.assertNotIn('neutral_or_uncertain',result.labels)
        self.assertTrue(result.uncertain);self.assertEqual(result.source,'UNVALIDATED_HEURISTIC');self.assertIsNone(result.scores)
    def test_english_threat_and_distress(self):
        self.positive('If you leave, I will hurt you.','explicit_threat')
        self.positive('Help me, I am scared.','distress_expression')
    def test_devanagari_hindi(self):
        self.positive('मैं तुम्हें जान से मार दूंगा','explicit_threat')
        self.positive('छूना बंद करो','unwanted_sexual_advance')
        self.positive('बचाओ मेरी मदद करो','distress_expression')
    def test_romanized_hindi_and_hinglish(self):
        self.positive('jaan se maar dunga','explicit_threat')
        self.positive('phir bhi do it anyway','refusal_ignored',['maine mana kiya'])
        self.positive('kapde utaro, karna padega','sexual_coercion')
    def test_refusal_then_continuation_requires_pressure(self):
        self.positive('I do it anyway','refusal_ignored',['I said no'])
        for text in ['Would you like coffee?','Okay I will stop.','Can I help you?']:
            self.assertEqual(self.classifier.classify(text,['I said no']).labels,{'neutral_or_uncertain'})
        self.assertNotIn('refusal_ignored',self.classifier.classify('do it anyway',['I said no','Okay I will stop']).labels)
    def test_sexual_coercion_and_unwanted_advance(self):
        self.positive('You have no choice, kiss me.','sexual_coercion')
        self.positive('I said no, stop touching me.','unwanted_sexual_advance')
    def test_secrecy_isolation_require_coercion(self):
        self.positive("Don't tell anyone, you have no choice.",'secrecy_pressure')
        self.positive('अकेले आओ, कोई चारा नहीं','isolation_pressure')
        for text in ["Don't tell anyone about the surprise.",'Come alone to the appointment.','kisi ko mat batana']:
            self.assertEqual(self.classifier.classify(text).labels,{'neutral_or_uncertain'})
    def test_consensual_normal_and_ambiguous_dialogue(self):
        for text in ['Would you like to kiss me?','Yes, I want to have sex.','Can I touch you there?',
                     'Let us go for coffee.','We agreed to be alone.','मुझे चाय चाहिए','kya tum coffee loge?']:
            with self.subTest(text=text):self.assertEqual(self.classifier.classify(text).labels,{'neutral_or_uncertain'})
    def test_quotes_reporting_fiction_negation(self):
        for text in ['He said I will kill you.','The news reported: I will hurt you.',
                     'In the movie I will kill you.','“I will kill you” is a quote.',
                     'The line is "I will hurt you".',"'I will hurt you' is fictional.",
                     "I won't hurt you.",'I will not kill you.',"I don't want to hurt you.",
                     'Not saying I will hurt you.', 'उसने कहा जान से मार दूंगा',
                     'jaan se nahi maar dunga', "I won't do it anyway"]:
            with self.subTest(text=text):self.assertEqual(self.classifier.classify(text,['I said no']).labels,{'neutral_or_uncertain'})
    def test_asr_uncertainty_and_unknown_speakers(self):
        result=self.classifier.classify('do it anyway',[Turn('I said no',True)],True)
        self.assertNotIn('refusal_ignored',result.labels);self.assertIn('asr_uncertain',result.reasons)
        self.assertIn('speaker_unknown',result.reasons)
        self.assertNotIn('speaker_unknown',self.classifier.classify('help me',[Turn('hello',False,'caller-provided-A')]).reasons)
    def test_empty_invalid_unicode_and_input_types(self):
        for text in ['', '   ', '\ud800', 'hello\u202eI will hurt you', 'text\x00', None, 42]:
            result=self.classifier.classify(text);self.assertEqual(result.labels,{'neutral_or_uncertain'});self.assertTrue(result.uncertain)
        for context in ['not-a-list',[{}],['x']*101]:
            self.assertIn('invalid_input',self.classifier.classify('help me',context).reasons)
    def test_normalization_preserves_negation_script_and_marks(self):
        self.assertEqual(normalize("  I\nwill not\t hurt you "),'I will not hurt you')
        self.assertEqual(normalize('e\u0301'),'é');self.assertEqual(normalize('नहीं\u200d'),'नहीं\u200d')
    def test_context_and_text_bounds_privacy(self):
        text,turns,reasons=prepare('x'*3000,['y'*2000]*8)
        self.assertEqual(len(text),2048);self.assertLessEqual(sum(len(t.text) for t in turns),MAX_CONTEXT)
        self.assertIn('context_or_text_truncated',reasons)
        self.assertNotIn('private text',repr(Turn('private text')))
        self.assertTrue(encoder_text('current',['old']).startswith('Current: current'))
    def test_neutral_consistency_and_close(self):
        for labels in [set(),{'invalid'},{'explicit_threat','neutral_or_uncertain'}]:
            with self.assertRaises(ValueError):Risk(labels)
        c=load_classifier();c.close();c.close()
        with self.assertRaises(RuntimeError):c.classify('help me')
        with self.assertRaises(ValueError):load_classifier('/absent/not-trained')
