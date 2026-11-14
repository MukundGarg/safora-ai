package org.safora.demo

import android.app.*
import android.content.*
import android.media.AudioManager
import android.media.AudioFocusRequest
import android.media.AudioAttributes
import android.os.*
import kotlinx.coroutines.*
import org.safora.ai.*

object DemoState {
    @Volatile var status = "Stopped — microphone monitoring is opt-in"
    @Volatile var event = "No incident"
    @Volatile var mode = "LITE"
    @Volatile var requestedMode = Mode.STANDARD
    @Volatile var speechLanguage = "en"
    var alerts: LocalAlerts? = null
    @Volatile var metricsJson = "{}"
    @Volatile var appVisible = true
    @Volatile var performance = "No performance sample yet"
    val sink = object : IncidentSink {
        override fun onRisk(event: RiskEvent) {
            DemoState.event="Level ${event.level}: ${event.reason}; local verification required"
            val shown=alerts?.show(event) ?: false
            if(!shown) DemoState.event+="; notifications unavailable—view this screen"
        }
        override fun onCoverage(status: String) { DemoState.status = status }
    }
    val engine = DecisionEngine(sink)
}
class MonitoringService : Service() {
    private val scope = CoroutineScope(SupervisorJob()+Dispatchers.Main.immediate)
    private var pipeline: AudioPipeline? = null
    @Volatile private var destroyed=false
    private val controller = ResourceController()
    private val serviceSink=object: IncidentSink {
        override fun onRisk(event:RiskEvent) { DemoState.sink.onRisk(event) }
        override fun onCoverage(status:String) {
            DemoState.sink.onCoverage(status)
            if(!destroyed) requireNotNull(getSystemService(NotificationManager::class.java)).notify(1,notification(status))
        }
    }
    private fun notification(status:String): Notification {
        val open=PendingIntent.getActivity(this,0,Intent(this,MainActivity::class.java),PendingIntent.FLAG_IMMUTABLE)
        val stop=PendingIntent.getService(this,1,Intent(this,MonitoringService::class.java).setAction("STOP"),PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this,"monitor").setContentTitle("SAFORA monitoring status")
            .setContentText(status).setStyle(Notification.BigTextStyle().bigText(status))
            .setSmallIcon(android.R.drawable.ic_btn_speak_now).setContentIntent(open).setOngoing(true)
            .addAction(Notification.Action.Builder(null,"Stop",stop).build()).build()
    }
    private lateinit var audio: AudioManager
    private lateinit var focus: AudioFocusRequest
    private var thermalListener: Any? = null
    override fun onCreate() {
        super.onCreate()
        val nm = requireNotNull(getSystemService(NotificationManager::class.java))
        nm.createNotificationChannel(NotificationChannel("monitor", "Safety monitoring", NotificationManager.IMPORTANCE_LOW))
        audio = requireNotNull(getSystemService(AudioManager::class.java))
        focus = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY).build())
            .setOnAudioFocusChangeListener { change ->
                if (change < 0) { DemoState.sink.onCoverage("Paused: audio interruption; restart from app"); stopSelf() }
            }.build()
        if (Build.VERSION.SDK_INT >= 29) {
            val listener=PowerManager.OnThermalStatusChangedListener { status ->
                if(status >= PowerManager.THERMAL_STATUS_SEVERE) {
                    pipeline?.setConversationEnabled(false)
                    serviceSink.onCoverage("Thermal pressure: acoustic only")
                }
            }
            thermalListener=listener
            requireNotNull(getSystemService(PowerManager::class.java)).addThermalStatusListener(listener)
        }
    }
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "STOP") { stopSelf();return START_NOT_STICKY }
        startForeground(1, notification("Starting acoustic monitoring…"))
        if (pipeline == null) {
            if (audio.requestAudioFocus(focus) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
                DemoState.sink.onCoverage("Monitoring unavailable: audio focus denied");stopSelf();return START_NOT_STICKY
            }
            pipeline=AudioPipeline(this,DemoState.engine,serviceSink,scope,DemoState.speechLanguage).also { it.start() }
            scope.launch(Dispatchers.Default) {
                while (isActive) {
                    val mem=ActivityManager.MemoryInfo();requireNotNull(getSystemService(ActivityManager::class.java)).getMemoryInfo(mem)
                    val power=requireNotNull(getSystemService(PowerManager::class.java))
                    val battery=requireNotNull(getSystemService(BatteryManager::class.java)).getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                    pipeline?.metrics?.memory(Debug.getPss().toLong(),processPeakRssKb())
                    val metric=pipeline?.metrics?.snapshot()
                    val sample=ResourceSample(mem.availMem/1048576,mem.lowMemory,
                        if(Build.VERSION.SDK_INT>=29) power.currentThermalStatus else 0,pipeline?.latencyMs?:0,pipeline?.backlog?:0,battery,Debug.getPss().toLong()/1024,DemoState.appVisible)
                    DemoState.performance=metric?.let {
                        "YAMNet mean/max: %.1f/%.1f ms; ASR mean/max: %.1f/%.1f ms\nPeak sampled PSS: %.1f MB; peak RSS: %.1f MB\nDropped blocks: %d; skipped windows (estimate): %d".format(
                            it.yamnetWallMeanMs,it.yamnetWallMaxMs,it.asrWallMeanMs,it.asrWallMaxMs,
                            it.peakSampledPssKb/1024.0,it.peakRssKb/1024.0,it.droppedBlocks,it.skippedWindows)
                    } ?: "No metrics yet"
                    if(metric!=null) DemoState.metricsJson=metric.toJson()
                    controller.requested=DemoState.requestedMode
                    val mode=controller.update(sample)
                    pipeline?.setConversationEnabled(mode != Mode.LITE)
                    DemoState.mode=mode.name+" (ASR requires local model)"
                    delay(5000)
                }
            }
        }
        return START_NOT_STICKY
    }
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        controller.update(ResourceSample(0,true,0,0,0,100));pipeline?.setConversationEnabled(false);DemoState.mode="LITE (memory pressure)"
    }
    override fun onDestroy() {
        destroyed=true
        DemoState.status="Stopping monitoring; pending inference cleanup"
        val retiring=pipeline;pipeline=null
        scope.cancel()
        CoroutineScope(Dispatchers.IO).launch {
            retiring?.stop()
            retiring?.metrics?.snapshot()?.let { DemoState.metricsJson=it.toJson() }
        }
        stopForeground(STOP_FOREGROUND_REMOVE)
        audio.abandonAudioFocusRequest(focus)
        if (Build.VERSION.SDK_INT>=29) {
            (thermalListener as? PowerManager.OnThermalStatusChangedListener)?.let {
                requireNotNull(getSystemService(PowerManager::class.java)).removeThermalStatusListener(it)
            }
        }
        super.onDestroy()
    }
    private fun processPeakRssKb(): Long = try {
        java.io.File("/proc/self/status").useLines { lines ->
            lines.firstOrNull { it.startsWith("VmHWM:") }?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull() ?: 0
        }
    } catch(_:Exception) { 0 }
    override fun onBind(intent: Intent?): IBinder? = null
}
