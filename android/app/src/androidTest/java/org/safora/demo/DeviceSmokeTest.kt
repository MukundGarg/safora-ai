package org.safora.demo

import android.Manifest
import android.app.NotificationManager
import android.content.Intent
import android.os.Build
import android.os.SystemClock
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.*
import org.junit.Assert.*
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import org.safora.ai.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Runs only on a device/emulator. Synthetic fixture is authored TTS, not validation data. */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class DeviceSmokeTest {
    private val context get()=InstrumentationRegistry.getInstrumentation().targetContext
    @get:Rule val permissions: GrantPermissionRule = GrantPermissionRule.grant(
        *if(Build.VERSION.SDK_INT>=33) arrayOf(Manifest.permission.RECORD_AUDIO,Manifest.permission.POST_NOTIFICATIONS)
         else arrayOf(Manifest.permission.RECORD_AUDIO))
    @Test fun a_bundledModelChecksumsAndNativeHindiSilence()=runBlocking {
        ModelAssets.validateYamnet(context)
        for(language in listOf("en","hi")) {
            val directory=ModelAssets.prepareVosk(context,language)
            assertTrue(directory.resolve("am/final.mdl").isFile)
            VoskSpeech(directory.absolutePath).use { assertTrue(it.transcribe(FloatArray(16000*8)).uncertain) }
        }
    }
    @Test fun b_realOfflineModelsSyntheticRefusalToCheckInNotification()=runBlocking {
        val fixture=InstrumentationRegistry.getInstrumentation().context.assets.open("synthetic-refusal.pcm16").use { it.readBytes() }
        val data=ByteBuffer.wrap(fixture).order(ByteOrder.LITTLE_ENDIAN)
        val shorts=ShortArray(fixture.size/2) { data.short }
        val pcm=FloatArray(shorts.size) { shorts[it]/32768f }
        val events=mutableListOf<RiskEvent>();val alerts=LocalAlerts(context.applicationContext as android.app.Application)
        val engine=DecisionEngine(object:IncidentSink {
            override fun onCoverage(status:String) {}
            override fun onRisk(event:RiskEvent) { events.add(event);alerts.show(event) }
        })
        try {
            val ring=AudioRing();val detector=TemporalDetector()
            Yamnet(context).use { model ->
                ring.add(shorts,shorts.size) { window,time ->
                    if(detector.update(time,model.scream(window))) engine.acoustic(time)
                }
            }
            VoskSpeech(ModelAssets.prepareVosk(context,"en").absolutePath).use { asr ->
                val recognized=asr.transcribe(pcm)
                assertTrue("Synthetic fixture must produce nonempty ASR text",recognized.text.isNotBlank())
                val risk=DevelopmentClassifier().classify(recognized,emptyList())
                assertTrue("Synthetic refusal fixture must reach development concern path",risk.labels!in listOf(emptySet<String>(),setOf("neutral_or_uncertain")))
                engine.conversation(9000,risk)
            }
            assertTrue(events.any { it.level==1 })
            assertTrue(context.getSystemService(NotificationManager::class.java)!!.activeNotifications.any { it.id==2 })
        } finally { fixture.fill(0);shorts.fill(0);pcm.fill(0f);alerts.clear() }
    }
    @Test fun c_manualSosWorksWithoutMonitoring() {
        context.stopService(Intent(context,MonitoringService::class.java));DemoState.engine.resolve()
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                find(activity.window.decorView) { it is Button && it.text=="MANUAL SOS" }.performClick()
            }
            assertTrue(DemoState.event.contains("manual_sos"))
            assertTrue(context.getSystemService(NotificationManager::class.java)!!.activeNotifications.any { it.id==2 })
            DemoState.engine.resolve();DemoState.alerts?.clear()
        }
    }
    @Test fun d_foregroundMicrophoneServiceProcessesAndStops() {
        DemoState.requestedMode=Mode.LITE // Native ASR is checked separately; no claim of real speech quality.
        DemoState.metricsJson="{}"
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            try {
                scenario.onActivity { activity ->
                    (find(activity.window.decorView) { it is CheckBox } as CheckBox).isChecked=true
                    find(activity.window.decorView) { it is Button && it.text=="Start monitoring" }.performClick()
                }
                val deadline=SystemClock.elapsedRealtime()+60000
                while(SystemClock.elapsedRealtime()<deadline && JSONObject(DemoState.metricsJson).optLong("yamnet_runs")==0L) Thread.sleep(500)
                assertTrue("Real microphone service must complete YAMNet inference",JSONObject(DemoState.metricsJson).optLong("yamnet_runs")>0)
            } finally { context.stopService(Intent(context,MonitoringService::class.java)) }
            val deadline=SystemClock.elapsedRealtime()+20000
            while(SystemClock.elapsedRealtime()<deadline && DemoState.status!="Monitoring stopped") Thread.sleep(200)
            assertEquals("Monitoring stopped",DemoState.status)
        }
    }
    private fun find(root:View,predicate:(View)->Boolean):View {
        fun search(view:View):View? {
            if(predicate(view)) return view
            if(view is ViewGroup) for(i in 0 until view.childCount) search(view.getChildAt(i))?.let { return it }
            return null
        }
        return requireNotNull(search(root)) { "Missing test control" }
    }
}
