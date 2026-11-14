package org.safora.ai
import org.junit.Assert.*
import org.junit.Test
class EngineTest {
    @Test fun temporalAndCooldown() {
        val d=TemporalDetector()
        assertFalse(d.update(0,.4f));assertFalse(d.update(500,.1f));assertTrue(d.update(1000,.6f))
        assertFalse(d.update(1500,.9f));assertFalse(d.update(2000,.9f))
        assertTrue(d.update(11000,.9f))
    }
    @Test fun manualBypassesSuppression() {
        val events=mutableListOf<RiskEvent>()
        val e=DecisionEngine(object:IncidentSink {
            override fun onRisk(event:RiskEvent) { events.add(event) }
            override fun onCoverage(status:String) {}
        })
        e.acoustic(1000);e.manual(1001);assertEquals(2,events.size);assertTrue(events.last().manual)
    }
    @Test fun higherPriorityEscalatesDuringCooldown() {
        val events=mutableListOf<RiskEvent>()
        val e=DecisionEngine(object:IncidentSink {
            override fun onRisk(event:RiskEvent) { events.add(event) }
            override fun onCoverage(status:String) {}
        })
        e.conversation(1000,ConversationRisk(setOf("distress_expression"),true))
        e.acoustic(1100);assertEquals(listOf(1,2),events.map { it.level })
    }
    @Test fun pressureDegradesImmediately() {
        val c=ResourceController(Mode.ENHANCED)
        repeat(3) { c.update(ResourceSample(900,false,0,100,0,80)) }
        assertEquals(Mode.ENHANCED,c.mode)
        assertEquals(Mode.LITE,c.update(ResourceSample(900,false,3,100,0,80)))
    }
}
