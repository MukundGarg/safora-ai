package org.safora.ai

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.CopyOnWriteArrayList

class MlControllerTest {
    private class Acoustic(val score:Float=0f):AcousticBackend {
        @Volatile var calls=0;@Volatile var closed=false
        override fun scream(samples:FloatArray):Float { calls++;return score }
        override fun close() { closed=true }
    }
    private class Speech(val entered:CountDownLatch?=null,val release:CountDownLatch?=null):SpeechRecognizer {
        @Volatile var closed=false
        override fun transcribe(pcm:FloatArray):Recognition { entered?.countDown();release?.await(5,TimeUnit.SECONDS);return Recognition("I will hurt you",false) }
        override fun close() { closed=true }
    }
    @Test fun permissionOptOutSilenceAndManualStayIndependent() {
        val events=mutableListOf<RiskEvent>();val engine=DecisionEngine(object:IncidentSink {
            override fun onCoverage(status:String) {}
            override fun onRisk(event:RiskEvent) { events.add(event) }
        })
        val cues=CopyOnWriteArrayList<MlCue>();val c=MlController({Acoustic()},{Speech()},cues::add)
        assertFalse(c.start(permission=false));engine.manual(0);assertTrue(events.single().manual)
        assertTrue(c.start());repeat(12) { c.submit(ShortArray(1600));assertTrue(c.awaitIdle(1000)) }
        assertTrue(cues.isEmpty());assertTrue(c.optOut());assertFalse(c.submit(ShortArray(1600)))
        engine.monitoringCue(1,"yamnet");assertEquals(1,events.size) // Active manual incident suppresses lower cue.
    }
    @Test fun slowSpeechCannotBlockAcousticsAndObsoleteResultsAreSuppressed() {
        val entered=CountDownLatch(1);val release=CountDownLatch(1);val acoustic=Acoustic();val speech=Speech(entered,release)
        val cues=CopyOnWriteArrayList<MlCue>();val c=MlController({acoustic},{speech},cues::add,segmentSamples=1600)
        c.configureSpeech(true,"en");c.start()
        try {
            c.submit(ShortArray(1600) { 1000 });assertTrue(entered.await(2,TimeUnit.SECONDS))
            repeat(15) { c.submit(ShortArray(1600));Thread.sleep(10) }
            val deadline=System.nanoTime()+2000000000
            while(acoustic.calls==0 && System.nanoTime()<deadline) Thread.sleep(5)
            assertTrue(acoustic.calls>0);assertEquals(1L,release.count)
            c.configureSpeech(false,null);release.countDown();assertTrue(c.awaitIdle(2000));assertTrue(cues.isEmpty())
        } finally { release.countDown();assertTrue(c.stop()) }
        assertTrue(speech.closed);assertTrue(acoustic.closed)
    }
    @Test fun boundedQueueOverflowAndNativeStopOwnership() {
        val entered=CountDownLatch(1);val release=CountDownLatch(1);val speech=Speech(entered,release)
        val c=MlController({Acoustic()},{speech},{},segmentSamples=1600,capacity=1)
        c.configureSpeech(true,"hi");c.start()
        try {
            c.submit(ShortArray(1600) { 1000 });assertTrue(entered.await(2,TimeUnit.SECONDS))
            repeat(10) { c.submit(ShortArray(1600) { 1000 }) }
            assertTrue(c.snapshot().speechDrops>0);assertTrue(c.snapshot().speechDepth<=1)
            assertFalse(c.stop(10));assertFalse(speech.closed)
            try { c.start();fail("Native work still owned") } catch(_:IllegalStateException) {}
        } finally { release.countDown();assertTrue(c.stop()) }
        assertTrue(speech.closed);assertTrue(c.start());assertTrue(c.stop())
    }
    @Test fun checkinIsAlwaysUncertainAndRequiresConfirmation() {
        val cues=CopyOnWriteArrayList<MlCue>();val c=MlController({Acoustic(.9f)},{Speech()},cues::add)
        c.start()
        try { repeat(25) { c.submit(ShortArray(1600));assertTrue(c.awaitIdle(1000)) } }
        finally { assertTrue(c.stop()) }
        assertEquals(1,cues.size);assertTrue(cues.single().uncertain);assertTrue(cues.single().monitoringOnly);assertTrue(cues.single().requiresUserConfirmation)
        val events=mutableListOf<RiskEvent>();val engine=DecisionEngine(object:IncidentSink {
            override fun onRisk(event:RiskEvent) { events.add(event) }
            override fun onCoverage(status:String) {}
        })
        engine.monitoringCue(100,"yamnet");assertEquals(1,events.single().level)
        engine.manual(101);assertTrue(events.last().manual);assertEquals(2,events.last().level)
    }
    @Test fun missingModelsDoNotKillControllerAndManualHasNoModelDependency() {
        val statuses=CopyOnWriteArrayList<String>();val c=MlController({error("fixture")},{error("fixture")},{},statuses::add)
        c.start();c.submit(ShortArray(1600));assertTrue(c.awaitIdle(2000));assertTrue(c.snapshot().errors>0)
        assertTrue(c.permissionLost());assertTrue(statuses.contains("permission_lost"))
    }
    @Test fun rulesPropagateAsrAndSuppressNegationQuotes() {
        val c=MonitoringRules()
        for(text in listOf("I won't hurt you.","He said I will kill you.","Would you like coffee?","“I will hurt you”"))
            assertEquals(setOf("neutral_or_uncertain"),c.classify(Recognition(text,false),emptyList(),"en").labels)
        assertTrue("explicit_threat" in c.classify(Recognition("जान से मार दूंगा",true),emptyList(),"hi").labels)
        assertFalse("refusal_ignored" in c.classify(Recognition("do it anyway",false),listOf(RiskTurn("I said no",true)),"en").labels)
        val risk=c.classify(Recognition("do it anyway",false),listOf(RiskTurn("I said no",false)),"en")
        assertTrue("refusal_ignored" in risk.labels);assertTrue(risk.uncertain);assertNull(risk.scores)
    }
    @Test fun immediateLanguageSwitchLoadsNewModelAndClearsOldWork() {
        val languages=CopyOnWriteArrayList<String>();val cues=CopyOnWriteArrayList<MlCue>()
        val c=MlController({Acoustic()},{ lang ->languages.add(lang);Speech() },cues::add,segmentSamples=1600,cooldownMs=0)
        c.configureSpeech(true,"en");c.start()
        try {
            c.submit(ShortArray(1600) { 1000 });assertTrue(c.awaitIdle(2000))
            c.configureSpeech(true,"hi");c.submit(ShortArray(1600) { 1000 });assertTrue(c.awaitIdle(2000))
            assertEquals(listOf("en","hi"),languages.toList());assertEquals("hi",cues.last().language)
            c.configureSpeech(false,null);Thread.sleep(100);assertFalse(c.snapshot().speechReady)
        } finally { assertTrue(c.stop()) }
    }
    @Test fun sharedRuleContractFixturesMatchKotlin() {
        val rules=MonitoringRules()
        val file=java.io.File("../../configs/risk-rule-contract.tsv")
        for(line in file.readLines().drop(1)) {
            val fields=line.split('\t');val context=if(fields[2]=="-") emptyList() else listOf(RiskTurn(fields[2],fields[3]=="true"))
            assertEquals(fields[4].split(',').toSet(),rules.classify(Recognition(fields[1],false),context,fields[0]).labels)
        }
    }
}
