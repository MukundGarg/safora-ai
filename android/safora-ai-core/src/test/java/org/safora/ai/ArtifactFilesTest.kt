package org.safora.ai

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files
import java.io.File

class ArtifactFilesTest {
    @Test fun detectsEqualLengthCorruptionAndMissingFiles() {
        val root=Files.createTempDirectory("model-fixture").toFile()
        try {
            val file=File(root,"am/final.mdl");file.parentFile!!.mkdirs();file.writeText("unit-placeholder")
            val manifest=mapOf("am/final.mdl" to ArtifactFile(file.length(),ArtifactFiles.hash(file)))
            assertTrue(ArtifactFiles.validate(root,manifest));file.writeText("UNIT-placeholder")
            assertFalse(ArtifactFiles.validate(root,manifest));file.delete();assertFalse(ArtifactFiles.validate(root,manifest))
        } finally { root.deleteRecursively() }
    }
    @Test fun rejectsTraversalAndLowStorage() {
        val root=Files.createTempDirectory("model-fixture").toFile()
        try {
            assertFalse(ArtifactFiles.validate(root,mapOf("../escape" to ArtifactFile(1,"bad"))))
            val needed=ArtifactFiles.requiredSpace(100,3);assertEquals(16484L,needed)
            try { ArtifactFiles.requireStorage(needed-1,needed);fail("Must reject low storage") } catch(e:IllegalStateException) { assertEquals("model_storage_unavailable",e.message) }
        } finally { root.deleteRecursively() }
    }
    @Test fun coverageIsNotSafetyQualificationAndManualRemainsIndependent() {
        val state=MlState(1,1,true,true,"en",0,1,0,0,0,true,true)
        assertEquals(Availability.AVAILABLE,state.coverage()[0].state)
        assertEquals(Availability.DEGRADED,state.coverage()[1].state)
        assertTrue(state.coverage().none { it.safetyValidated })
        assertEquals(Availability.UNAVAILABLE,state.copy(language=null).coverage()[1].state)
        assertEquals(Availability.UNAVAILABLE,state.copy(running=false).coverage()[0].state)
    }
    @Test fun failedModelLoadUsesBoundedBackoffAndDoesNotReplayAudio() {
        val clock=java.util.concurrent.atomic.AtomicLong(1000000000L)
        val loads=java.util.concurrent.atomic.AtomicInteger()
        val core=MlController({ loads.incrementAndGet();error("fixture") },{ error("unused") },{},nanoClock=clock::get)
        core.start()
        try {
            core.submit(ShortArray(1600));assertTrue(core.awaitIdle(2000));assertEquals(1,loads.get())
            core.submit(ShortArray(1600));assertTrue(core.awaitIdle(2000));assertEquals(1,loads.get())
            clock.addAndGet(2000000000L)
            core.submit(ShortArray(1600));assertTrue(core.awaitIdle(2000));assertEquals(2,loads.get())
            assertFalse(core.snapshot().acousticReady)
        } finally { assertTrue(core.stop()) }
    }
    @Test fun speechQueuePressureCannotResetFailedModelBackoff() {
        val entered=java.util.concurrent.CountDownLatch(1);val release=java.util.concurrent.CountDownLatch(1)
        val attempts=java.util.concurrent.atomic.AtomicInteger();val clock=java.util.concurrent.atomic.AtomicLong(1000000000)
        val core=MlController({ object:AcousticBackend {
            override fun scream(samples:FloatArray)=0f
            override fun close() {}
        } },{ attempts.incrementAndGet();entered.countDown();release.await(3,java.util.concurrent.TimeUnit.SECONDS);error("fixture") },{},capacity=1,nanoClock=clock::get)
        core.configureSpeech(true,"en");core.start()
        try {
            core.submit(ShortArray(1600));assertTrue(entered.await(2,java.util.concurrent.TimeUnit.SECONDS))
            repeat(5) { core.submit(ShortArray(1600)) }
            assertTrue(core.snapshot().speechDrops>0);release.countDown();assertTrue(core.awaitIdle(3000))
            core.submit(ShortArray(1600));assertTrue(core.awaitIdle(2000));assertEquals(1,attempts.get())
        } finally { release.countDown();assertTrue(core.stop()) }
    }
}
