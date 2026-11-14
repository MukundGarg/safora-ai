from dataclasses import dataclass
from enum import IntEnum
class Level(IntEnum): NORMAL=0; CONCERN=1; HIGH=2
@dataclass
class Incident:
    id: int
    level: Level
    reason: str
    manual: bool = False
    action: str = 'local_check_in'
class DecisionEngine:
    """Produces local events only. Incident manager owns verification and contacts."""
    def __init__(self,cooldown=10,context_ttl=30):
        self.cooldown=cooldown;self.ttl=context_ttl;self.last=-float('inf');self.clock=-float('inf');self.active=None;self.count=0;self.evidence=[]
    def update(self,time,acoustic=None,risk=None,manual=False):
        if time<self.clock: raise ValueError('Nonmonotonic time')
        self.clock=time;self.evidence=[(t,l) for t,l in self.evidence if time-t<=self.ttl]
        level=Level.NORMAL;reason='normal'
        if acoustic: level=Level.HIGH;reason=acoustic.reason
        if risk and risk.labels!={'neutral_or_uncertain'}:
            self.evidence.append((time,risk.labels))
            recent=set().union(*(labels for _,labels in self.evidence))
            strong='explicit_threat' in risk.labels or {'refusal_ignored','sexual_coercion'}<=recent
            candidate=Level.HIGH if strong and not risk.uncertain else Level.CONCERN
            if candidate>level: level=candidate;reason=','.join(sorted(risk.labels))
        if manual: level=Level.HIGH;reason='manual_sos'
        if level==Level.NORMAL: return None
        if not manual:
            if self.active and level<=self.active.level: return None
            if not self.active and time-self.last<self.cooldown: return None
        self.count+=1;self.last=time
        self.active=Incident(self.count,level,reason,manual,'incident_verification' if level==Level.HIGH else 'local_check_in')
        return self.active
    def resolve(self): self.active=None;self.evidence.clear()
