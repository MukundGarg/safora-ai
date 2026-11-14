package org.safora.ai

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import android.os.Debug
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Yamnet(context: Context) : AutoCloseable {
    private val interpreter: Interpreter
    private val input = ByteBuffer.allocateDirect(15600 * 4).order(ByteOrder.nativeOrder())
    private val output = arrayOf(FloatArray(521))
    init {
        ModelAssets.validateYamnet(context)
        val bytes = context.assets.open("yamnet.tflite").use { it.readBytes() }
        val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).put(bytes)
        model.rewind()
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(2))
        require(interpreter.getInputTensor(0).dataType()==org.tensorflow.lite.DataType.FLOAT32)
        require(interpreter.getOutputTensor(0).dataType()==org.tensorflow.lite.DataType.FLOAT32)
        require(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(15600)))
        require(interpreter.getOutputTensor(0).shape().contentEquals(intArrayOf(1,521)))
    }
    fun scream(samples: FloatArray): Float {
        require(samples.size == 15600)
        input.rewind()
        samples.forEach { require(it.isFinite() && it in -1f..1f); input.putFloat(it) }
        input.rewind(); interpreter.run(input, output)
        return output[0][11]
    }
    override fun close() = interpreter.close()
}

/** Fixed-size ring. Borrowed window is consumed synchronously, never retained. */
class AudioRing(private val window: Int = 15600, private val hop: Int = 8000) {
    private val ring = FloatArray(window)
    private val view = FloatArray(window)
    private var cursor = 0
    private var count = 0L
    private var origin = 0L
    fun add(samples: ShortArray, size: Int, consume: (FloatArray, Long) -> Unit) {
        for (i in 0 until size) {
            ring[cursor] = samples[i] / 32768f
            cursor = (cursor+1) % window; count++
            if (count-origin >= window && (count-origin-window) % hop == 0L) {
                for (j in view.indices) view[j] = ring[(cursor+j) % window]
                consume(view, count*1000/16000)
            }
        }
    }
    fun clear(startSample: Long = 0) { ring.fill(0f);view.fill(0f);cursor=0;count=startSample;origin=startSample }
}

/** Single microphone owner. ML controller has independent bounded acoustic/speech workers. */
class AudioPipeline(private val context:Context,private val engine:DecisionEngine,
                    private val sink:IncidentSink,private val scope:CoroutineScope,
                    private val language:String="en") {
    @Volatile private var record:AudioRecord?=null
    private var job:Job?=null
    @Volatile private var stopping=false
    @Volatile private var conversationEnabled=false
    @Volatile private var selectedLanguage:String?=language
    @Volatile private var controller:MlController?=null
    val metrics=PipelineMetrics()
    val latencyMs:Long get()=controller?.latencyMs ?: 0
    val backlog:Int get()=controller?.backlog ?: 0
    fun setConversationEnabled(value:Boolean) { conversationEnabled=value;controller?.configureSpeech(value,selectedLanguage) }
    fun setSpeechLanguage(value:String?) {
        require(value==null || value in setOf("en","hi"))
        selectedLanguage=value;controller?.configureSpeech(conversationEnabled,value)
    }
    @SuppressLint("MissingPermission")
    fun start() {
        check(job==null);stopping=false
        job=scope.launch(Dispatchers.IO) {
            val core=MlController(
                acousticFactory={ val yam=Yamnet(context);object:AcousticBackend {
                    override fun scream(samples:FloatArray)=yam.scream(samples)
                    override fun close()=yam.close()
                } },
                speechFactory={ lang ->
                    val dir=runBlocking { ModelAssets.prepareVosk(context,lang) }
                    VoskSpeech(dir.absolutePath)
                },
                onCue={ cue -> val state=controller?.snapshot()
                    if(!stopping && state?.running==true && state.session==cue.session && state.generation==cue.generation)
                        engine.monitoringCue(SystemClock.elapsedRealtime(),cue.source) },
                onStatus={ code -> if(!stopping) sink.onCoverage(code) },metrics=metrics,
                cpuClock={ Debug.threadCpuTimeNanos() })
            controller=core;core.configureSpeech(conversationEnabled,selectedLanguage)
            try {
                val allowed=context.checkSelfPermission(android.Manifest.permission.RECORD_AUDIO)==android.content.pm.PackageManager.PERMISSION_GRANTED
                if(!core.start(permission=allowed)) { sink.onCoverage("permission_denied; manual SOS remains available");return@launch }
                val min=AudioRecord.getMinBufferSize(16000,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT)
                check(min>0) { "16kHz capture unavailable" }
                val mic=AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,16000,
                    AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT,maxOf(min,32000))
                record=mic;check(mic.state==AudioRecord.STATE_INITIALIZED);mic.startRecording()
                sink.onCoverage("Monitoring only: uncertain ML cues require user confirmation")
                val pcm=ShortArray(1600)
                try {
                    while(isActive && !stopping) {
                        val n=mic.read(pcm,0,pcm.size,AudioRecord.READ_BLOCKING)
                        check(n>0) { "Microphone interrupted" }
                        core.submit(pcm,n);pcm.fill(0)
                    }
                } finally { pcm.fill(0) }
            } catch(e:CancellationException) { throw e }
            catch(_:Exception) { if(!stopping) sink.onCoverage("capture_unavailable; manual SOS remains available") }
            finally {
                try { record?.stop() } catch(_:IllegalStateException) {}
                record?.release();record=null
                // Native calls cannot be interrupted. Own their resources until worker return.
                withContext(NonCancellable+Dispatchers.IO) { while(!core.stop(1000)) { delay(10) } }
                controller=null
            }
        }
    }
    suspend fun stop() {
        stopping=true
        try { record?.stop() } catch(_:IllegalStateException) {}
        job?.cancelAndJoin();job=null;sink.onCoverage("Monitoring stopped")
    }
}
