# Conversation language-pattern taxonomy — risk-v2

Multi-label output preserves the eight existing labels/order. Positive categories
can coexist; `neutral_or_uncertain` is exclusive and used when no positive category
is selected. It means insufficient selected evidence, **not** established safety.
Neither a category nor a score establishes consent, a speaker's gender/identity,
criminal intent, imminent danger or a need for automatic emergency dispatch.
Women-centric safety scenarios motivate the taxonomy; voices/text do not establish
that a speaker is a woman, an aggressor or a victim.

| Label | Evidence category | Exclusions / interpretation boundary |
|---|---|---|
| sexual_coercion | Sexual contact/activity language coupled with explicit pressure or a threat | Sexual vocabulary, questions and consensual statements alone are insufficient; consent is not inferred |
| explicit_threat | Direct first-person threat to hurt/kill an addressee in the baseline's narrow patterns | Quoted/reported/fictional or explicitly negated threat statements are suppressed; target and sincerity remain unknown |
| refusal_ignored | Explicit refusal in current/bounded earlier context followed by explicit continued pressure | A refusal followed by an ordinary question or an acknowledgement to stop is insufficient; speaker continuity is not established |
| unwanted_sexual_advance | Contact/sexual language with an explicit refusal/stop expression | Indicates refusal language, not a legal finding or proof of whose consent is involved |
| secrecy_pressure | Secrecy request with explicit coercive pressure/threat | Surprise parties, ordinary privacy and secrecy requests alone are insufficient |
| isolation_pressure | Request to come/be alone with coercive pressure/threat | Being alone, appointments and ordinary invitations alone are insufficient |
| distress_expression | Explicit help/fear/refusal/stop language | Does not prove physical danger; contextual or metaphorical usages can still be misread |
| neutral_or_uncertain | No positive category selected / inadequate usable input | Not a negative safety guarantee; unknown language, missing model, truncation and missed phrasing remain possible |

Heuristic rules are deliberately narrow and always `UNVALIDATED_HEURISTIC` and
uncertain; they provide no numerical confidence. Quote/news/fiction/negation clause
suppression can miss an actual threat appearing in the same clause. Unmarked
sarcasm, indirect threats, complex negation, spelling variation, reported abuse,
consensual roleplay and ambiguous speaker turns are not resolved. English, Hindi,
romanized Hindi and Hinglish **authored unit fixtures** cover selected behavior;
there is no measured language/population coverage. Do not substitute a vocabulary
list or a generic multilingual encoder for consented human annotation/evaluation.

For human labeling, retain only consented/licensed bounded context, allow multiple
positive categories, explicitly adjudicate disagreements and use neutral alone
where evidence is absent/uncertain. Do not force ambiguous utterances into positive
labels. Record language, source/session, permission provenance and optional reviewed
annotations. Independent bilingual reviewers and people experienced with relevant
safety scenarios should define examples/exclusions before data collection. Never
label criminal intent or gender based on transcript content.
