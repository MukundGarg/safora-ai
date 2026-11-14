from dataclasses import dataclass
@dataclass
class Pressure:
    available_mb: float
    latency_ms: float=0
    queue: int=0
    thermal: int=0
    battery: int=100
    low_memory: bool=False
    foreground: bool=True
class Controller:
    def __init__(self,requested='standard'): self.requested=requested;self.mode='lite';self.good=0
    def update(self,p):
        severe=p.low_memory or p.available_mb<180 or p.thermal>=3 or p.queue>=2 or p.latency_ms>1500 or p.battery<10
        if severe: self.mode='lite';self.good=0
        else:
            self.good+=1
            if self.good>=3:
                self.mode='enhanced' if self.requested=='enhanced' and p.available_mb>700 and p.foreground else ('standard' if self.requested!='lite' and p.available_mb>350 else 'lite')
        return {'mode':self.mode,'conversation_enabled':self.mode!='lite','coverage':'acoustic_only' if self.mode=='lite' else 'audio_and_conversation','target_mb':{'lite':100,'standard':200,'enhanced':300}[self.mode]}
