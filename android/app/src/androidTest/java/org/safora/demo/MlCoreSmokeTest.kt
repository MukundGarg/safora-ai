package org.safora.demo

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.Assert.*
import org.junit.runner.RunWith
import org.safora.ai.*
import android.os.Debug
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CopyOnWriteArrayList

/** Runnable native-core smoke; compilation does not mean these tests ran on Android. */
@RunWith(AndroidJUnit4::class)
class MlCoreSmokeTest {
    private val context get()=InstrumentationRegistry.getInstrumentation().targetContext
    @Test fun nativeModelsThroughIndependentController() {
        val cues=CopyOnWriteArrayList<MlCue>()
        val controller=MlController(
            { val model=Yamnet(context);object:AcousticBackend {
                override fun scream(samples:FloatArray)=model.scream(samples)
                override fun close()=model.close()
            } },
            { language ->VoskSpeech(runBlocking { ModelAssets.prepareVosk(context,language) }.absolutePath) },
            cues::add,cpuClock={ Debug.threadCpuTimeNanos() })
        val bytes=InstrumentationRegistry.getInstrumentation().context.assets.open("synthetic-refusal.pcm16").use { it.readBytes() }
        val data=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val pcm=ShortArray(bytes.size/2) { data.short }
        controller.configureSpeech(true,"en")
        assertTrue(controller.start())
        try {
            for(start in pcm.indices step 1600) {
                controller.submit(pcm.copyOfRange(start,minOf(start+1600,pcm.size)))
                assertTrue(controller.awaitIdle(120000))
            }
            assertTrue(controller.metrics.snapshot().yamnetRuns>0)
            assertEquals(1L,controller.metrics.snapshot().asrRuns)
            assertTrue(cues.all { it.uncertain && it.monitoringOnly && it.requiresUserConfirmation })
            // Recognition/rule outcomes are not guaranteed even on authored TTS.
            controller.configureSpeech(true,"hi")
            repeat(80) { controller.submit(ShortArray(1600));assertTrue(controller.awaitIdle(120000)) }
            assertTrue(controller.snapshot().speechReady)
            assertEquals("hi",controller.snapshot().language)
        } finally { pcm.fill(0);bytes.fill(0);assertTrue(controller.stop(120000)) }
    }
    @Test fun explicitPermissionAndOptInPolicyDoNotLoadNativeModels() {
        var calls=0
        val controller=MlController({ calls++;error("Must not load") },{ calls++;error("Must not load") },{})
        assertFalse(controller.start(permission=false));assertFalse(controller.start(optedIn=false))
        assertEquals(0,calls);assertTrue(controller.stop())
    }
}
