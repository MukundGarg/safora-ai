package org.safora.ai
import org.junit.Assert.*
import org.junit.Test
class PipelineTest {
    @Test fun baselineStaysUncertainAndEmitsOnlyLocalConcern() {
        val events=mutableListOf<RiskEvent>()
        val engine=DecisionEngine(object:IncidentSink {
            override fun onRisk(event:RiskEvent) { events.add(event) }
            override fun onCoverage(status:String) {}
        })
        val classifier=DevelopmentClassifier()
        val risk=classifier.classify(Recognition("i said no stop touching me",false),emptyList())
        assertTrue(risk.uncertain);engine.conversation(1000,risk)
        assertEquals(1,events.single().level)
        engine.conversation(1100,risk);assertEquals(1,events.size)
        engine.manual(1101);assertTrue(events.last().manual)
    }
    @Test fun ordinaryInvitationAndAmbiguousPrivacyDoNotEscalate() {
        val c=DevelopmentClassifier()
        for(text in listOf("let's go somewhere for coffee","come somewhere alone with me","don't tell anyone what happened")) {
            assertEquals(setOf("neutral_or_uncertain"),c.classify(Recognition(text,false),emptyList()).labels)
        }
    }
    @Test fun refusalRequiresExplicitContinuedPressure() {
        val c=DevelopmentClassifier()
        assertTrue("refusal_ignored" in c.classify(Recognition("do it anyway",false),listOf("i said no")).labels)
        assertFalse("refusal_ignored" in c.classify(Recognition("would you like coffee",false),listOf("i said no")).labels)
    }
    @Test fun gapCannotConfirmAcrossMissingAudio() {
        val d=TemporalDetector();assertFalse(d.update(1000,.7f))
        d.resetContinuity();assertFalse(d.update(2000,.7f));assertTrue(d.update(2500,.7f))
    }
    @Test fun metricsAccumulateWithoutAudioOrText() {
        val m=PipelineMetrics()
        m.inference("yamnet",10000000,4000000);m.inference("yamnet",20000000,6000000)
        m.inference("asr",100000000,50000000);m.drop(8000);m.skip(2);m.queue(2)
        m.memory(100000,150000);m.memory(90000,140000)
        val s=m.snapshot()
        assertEquals(15.0,s.yamnetWallMeanMs,0.0);assertEquals(20.0,s.yamnetWallMaxMs,0.0)
        assertEquals(5.0,s.yamnetCpuMeanMs,0.0);assertEquals(100000L,s.peakSampledPssKb)
        assertEquals(150000L,s.peakRssKb);assertEquals(8000L,s.droppedSamples);assertEquals(2L,s.skippedWindows)
    }
    @Test fun discontinuousRingUsesAbsoluteSampleClock() {
        val ring=AudioRing();ring.add(ShortArray(8000),8000) { _,_->fail("Too short") }
        ring.clear(32000)
        val times=mutableListOf<Long>()
        ring.add(ShortArray(15600),15600) { _,time->times.add(time) }
        assertEquals(listOf(2975L),times)
    }
}
