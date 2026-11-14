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

/** One microphone, bounded nonblocking capture, serial inference, explicit coverage gaps. */
class AudioPipeline(private val context: Context, private val engine: DecisionEngine,
                    private val sink: IncidentSink, private val scope: CoroutineScope,
                    private val language: String = "en") {
    private var record: AudioRecord? = null
    private var job: Job? = null
    @Volatile private var stopping = false
    @Volatile private var conversationEnabled = false
    val metrics=PipelineMetrics()
    @Volatile var latencyMs = 0L; private set
    private val depth=AtomicInteger()
    val backlog: Int get()=depth.get().coerceAtLeast(0)
    fun setConversationEnabled(value: Boolean) { conversationEnabled=value }
    private data class Block(val pcm: ShortArray,val size: Int,val start: Long)
    @SuppressLint("MissingPermission")
    fun start() {
        check(job == null);stopping=false
        job=scope.launch(Dispatchers.IO) {
            var model: Yamnet?=null;var speech: VoskSpeech?=null
            val classifier=DevelopmentClassifier();val history=ArrayDeque<Pair<Long,String>>()
            var speechFailure=false;var gapStatusAt=0L
            val ring=AudioRing();val detector=TemporalDetector()
            val speechPcm=FloatArray(16000*8);var speechSize=0
            val queue=Channel<Block>(2);val pool=Channel<ShortArray>(4)
            repeat(4) { pool.trySend(ShortArray(8000)) }
            val discard=ShortArray(8000)
            var expectedStart=0L;var previousWindowEnd=0L
            try {
                sink.onCoverage("Preparing bundled YAMNet; speech model: $language (unvalidated)")
                model=Yamnet(context)
                val min=AudioRecord.getMinBufferSize(16000,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT)
                check(min>0) { "16 kHz capture unavailable" }
                val mic=AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,16000,
                    AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT,maxOf(min,32000))
                record=mic;check(mic.state==AudioRecord.STATE_INITIALIZED);mic.startRecording()
                sink.onCoverage("Monitoring: acoustic only; speech starts in Standard/Enhanced mode")
                coroutineScope {
                    val capture=launch(Dispatchers.IO) {
                        var captured=0L
                        try {
                            while(isActive && !stopping) {
                                val reusable=pool.tryReceive().getOrNull()
                                val pcm=reusable ?: discard
                                val n=mic.read(pcm,0,pcm.size,AudioRecord.READ_BLOCKING)
                                if(n<=0) { reusable?.let { pool.trySend(it) };error("Microphone interrupted: $n") }
                                val block=Block(pcm,n,captured);captured+=n
                                if(reusable==null) { metrics.drop(n);pcm.fill(0);continue }
                                depth.incrementAndGet()
                                if(queue.trySend(block).isSuccess) metrics.queue(backlog)
                                else { depth.decrementAndGet();metrics.drop(n);pcm.fill(0);pool.trySend(pcm) }
                            }
                        } finally { queue.close() }
                    }
                    try {
                        for(block in queue) {
                            depth.decrementAndGet()
                            val (pcm,n,startSample)=block
                            try {
                                val now=SystemClock.elapsedRealtime()
                                while(history.isNotEmpty() && now-history.first().first>30000) history.removeFirst()
                                if(startSample!=expectedStart) {
                                    ring.clear(startSample);detector.resetContinuity();speechPcm.fill(0f);speechSize=0;history.clear()
                                    if(now-gapStatusAt>=5000) {
                                        sink.onCoverage("Audio gap: dropped capture blocks; partial speech discarded");gapStatusAt=now
                                    }
                                }
                                expectedStart=startSample+n
                                ring.add(pcm,n) { audio,time ->
                                    val end=time*16
                                    if(previousWindowEnd>0) metrics.skip(((end-previousWindowEnd)/8000-1).coerceAtLeast(0))
                                    else metrics.skip(((end-15600)/8000).coerceAtLeast(0))
                                    previousWindowEnd=end
                                    val wall=SystemClock.elapsedRealtimeNanos();val cpu=Debug.threadCpuTimeNanos()
                                    val score=requireNotNull(model).scream(audio)
                                    val duration=SystemClock.elapsedRealtimeNanos()-wall
                                    latencyMs=duration/1000000
                                    metrics.inference("yamnet",duration,Debug.threadCpuTimeNanos()-cpu)
                                    if(detector.update(time,score)) engine.acoustic(SystemClock.elapsedRealtime())
                                }
                                if(!conversationEnabled) {
                                    if(speech!=null) { speech?.close();speech=null;sink.onCoverage("Monitoring: acoustic only; speech suspended by selected/resource mode") }
                                    speechPcm.fill(0f);speechSize=0;history.clear()
                                } else {
                                    if(speech==null && !speechFailure) {
                                        try {
                                            sink.onCoverage("Acoustic monitoring active; preparing offline $language speech model")
                                            val dir=ModelAssets.prepareVosk(context,language)
                                            currentCoroutineContext().ensureActive()
                                            if(conversationEnabled) {
                                                speech=VoskSpeech(dir.absolutePath)
                                                sink.onCoverage("Monitoring: YAMNet + offline Vosk $language + UNVALIDATED text baseline")
                                            }
                                        } catch(e:CancellationException) { throw e }
                                        catch(e:Exception) { speechFailure=true;sink.onCoverage("Acoustic only: ASR asset/runtime failed (${e.javaClass.simpleName})") }
                                    }
                                    if(speech!=null) {
                                        for(i in 0 until n) {
                                            speechPcm[speechSize++]=pcm[i]/32768f
                                            if(speechSize==speechPcm.size) {
                                                var energy=0.0;speechPcm.forEach { energy+=it*it }
                                                // ponytail: energy VAD is unvalidated; quiet-speech coverage needs calibrated VAD.
                                                if(kotlin.math.sqrt(energy/speechPcm.size)>.008) {
                                                    try {
                                                        val wall=SystemClock.elapsedRealtimeNanos();val cpu=Debug.threadCpuTimeNanos()
                                                        val recognition=requireNotNull(speech).transcribe(speechPcm)
                                                        metrics.inference("asr",SystemClock.elapsedRealtimeNanos()-wall,Debug.threadCpuTimeNanos()-cpu)
                                                        engine.conversation(SystemClock.elapsedRealtime(),classifier.classify(recognition,history.map { it.second }))
                                                        history.addLast(SystemClock.elapsedRealtime() to recognition.text)
                                                        if(history.size>3) history.removeFirst()
                                                    } catch(e:Exception) {
                                                        speech?.close();speech=null;speechFailure=true
                                                        sink.onCoverage("Acoustic only: speech inference failed")
                                                    }
                                                }
                                                speechSize=0;speechPcm.fill(0f)
                                            }
                                        }
                                    }
                                }
                            } finally { pcm.fill(0);pool.trySend(pcm) }
                        }
                    } finally {
                        capture.cancel()
                        try { mic.stop() } catch(_:IllegalStateException) {}
                        capture.join()
                    }
                }
            } catch(e:CancellationException) { throw e }
            catch(e:Exception) { if(!stopping) sink.onCoverage("Monitoring failed: ${e.javaClass.simpleName}; manual SOS remains available") }
            finally {
                try { record?.stop() } catch(_:IllegalStateException) {}
                record?.release();record=null
                model?.close();speech?.close();classifier.close();ring.clear();speechPcm.fill(0f);history.clear();discard.fill(0)
                while(true) { val pcm=queue.tryReceive().getOrNull()?.pcm ?: break;pcm.fill(0) }
                while(true) { val pcm=pool.tryReceive().getOrNull() ?: break;pcm.fill(0) }
                depth.set(0)
            }
        }
    }
    suspend fun stop() {
        stopping=true
        try { record?.stop() } catch(_:IllegalStateException) {}
        job?.cancelAndJoin();job=null;sink.onCoverage("Monitoring stopped")
    }
}
