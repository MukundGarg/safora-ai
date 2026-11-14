package org.safora.ai

import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** No AudioRecord/UI ownership. Exactly one host capture stream, mono PCM16 at16kHz. */
interface AcousticBackend:AutoCloseable { fun scream(samples:FloatArray):Float }
data class MlCue(val session:Long,val generation:Long,val endSample:Long,val source:String,
                 val language:String?,val uncertain:Boolean=true,val monitoringOnly:Boolean=true,
                 val requiresUserConfirmation:Boolean=true,val provenance:String="UNVALIDATED_HEURISTIC",
                 val asrUncertain:Boolean?=null,val riskVersion:String="risk-v2-rules")
data class MlState(val session:Long,val generation:Long,val running:Boolean,val speechEnabled:Boolean,
                   val language:String?,val acousticDrops:Long,val speechDrops:Long,val errors:Long,
                   val acousticDepth:Int,val speechDepth:Int,val acousticReady:Boolean,val speechReady:Boolean)

/** Two bounded queues and native-owner workers. Capture never loads/executes models.
 * Every ML cue is an uncertain, user-confirmed check-in, never dispatch or SOS.
 * The owner calls manual SOS independently through its existing DecisionEngine.
 */
class MlController(private val acousticFactory:()->AcousticBackend,
                   private val speechFactory:(String)->SpeechRecognizer,
                   private val onCue:(MlCue)->Unit,
                   private val onStatus:(String)->Unit={},
                   val metrics:PipelineMetrics=PipelineMetrics(),
                   private val segmentSamples:Int=16000*8,
                   private val capacity:Int=2,
                   private val cooldownMs:Long=10000,
                   private val nanoClock:()->Long=System::nanoTime,
                   private val cpuClock:()->Long={ 0L }) {
    init { require(segmentSamples in 1600..128000 && capacity in 1..8 && cooldownMs>=0) }
    private data class Block(val pcm:ShortArray,val start:Long,val epoch:Long,val captureEpoch:Long)
    private val gate=Any()
    private val audio=ArrayBlockingQueue<Block>(capacity)
    private val speech=ArrayBlockingQueue<Block>(capacity)
    private val audioPending=AtomicInteger()
    private val speechPending=AtomicInteger()
    private val running=AtomicBoolean(false)
    private var workers:List<Thread> = emptyList()
    private var session=0L;private var epoch=0L;private var audioEpoch=0L;private var next=0L;private var lastCue=Long.MIN_VALUE/2
    private var enabled=false;private var language:String?="en"
    private var acousticDrops=0L;private var speechDrops=0L;private var errors=0L
    @Volatile private var acousticReady=false
    @Volatile private var speechReady=false
    @Volatile private var controlGeneration=0L
    @Volatile var latencyMs=0L;private set
    @Volatile private var acousticBusy=false
    @Volatile private var speechBusy=false
    val backlog:Int get()=audio.size
    private fun status(code:String) { try { onStatus(code) } catch(_:Exception) {} }
    private fun clear(queue:ArrayBlockingQueue<Block>) {
        while(true) { val block=queue.poll() ?: break;block.pcm.fill(0)
            (if(queue===audio) audioPending else speechPending).decrementAndGet() }
    }
    private fun invalidate() { epoch++;controlGeneration=epoch;clear(speech) }
    fun configureSpeech(value:Boolean,selected:String?) {
        require(selected==null || selected in setOf("en","hi")) { "Explicit language required; no automatic LID" }
        synchronized(gate) { if(enabled!=value || language!=selected) { enabled=value;language=selected;invalidate() } }
    }
    fun start(permission:Boolean=true,optedIn:Boolean=true):Boolean = synchronized(gate) {
        check(!running.get() && workers.none { it.isAlive }) { "Stop/join previous native work before resume" }
        if(!permission || !optedIn) { status(if(!permission) "permission_denied" else "opted_out");return@synchronized false }
        session++;epoch++;audioEpoch++;controlGeneration=epoch;next=0;acousticDrops=0;speechDrops=0;errors=0;lastCue=Long.MIN_VALUE/2;running.set(true)
        clear(audio);clear(speech)
        workers=listOf(Thread({ acousticLoop() },"safora-acoustic"),Thread({ speechLoop() },"safora-speech"))
        workers.forEach { it.start() };true
    }
    /** False means acoustic block not accepted. Speech has independent acceptance/drop counters. */
    fun submit(pcm:ShortArray,size:Int=pcm.size,startSample:Long?=null,discontinuity:Boolean=false):Boolean {
        require(size in 1..1600 && size<=pcm.size) { "16kHz mono PCM16, max100ms" }
        synchronized(gate) {
            if(!running.get()) return false
            val start=startSample ?: next;require(start>=next)
            if(discontinuity || start!=next) { audioEpoch++;invalidate();clear(audio) }
            next=start+size
            val a=Block(pcm.copyOf(size),start,epoch,audioEpoch)
            audioPending.incrementAndGet();val ok=audio.offer(a)
            if(!ok) { audioPending.decrementAndGet();a.pcm.fill(0);acousticDrops++;metrics.drop(size) } else metrics.queue(audio.size)
            if(enabled && language!=null) {
                val b=Block(pcm.copyOf(size),start,epoch,audioEpoch)
                speechPending.incrementAndGet()
                if(!speech.offer(b)) { speechPending.decrementAndGet();b.pcm.fill(0);speechDrops++;invalidate() }
            }
            return ok
        }
    }
    private fun cue(source:String,end:Long,token:Long,lang:String?,sessionId:Long,asrUncertain:Boolean?=null) {
        val event=synchronized(gate) {
            if(!running.get() || session!=sessionId || (if(source=="yamnet") audioEpoch!=token else epoch!=token)) return
            val now=nanoClock()/1000000
            if(now-lastCue<cooldownMs) return
            lastCue=now
            MlCue(session,epoch,end,source,lang,provenance=if(source=="yamnet") "YAMNET_PRETRAINED_UNVALIDATED_SAFETY" else "UNVALIDATED_HEURISTIC",asrUncertain=asrUncertain)
        }
        try { onCue(event) } catch(_:Exception) { synchronized(gate) { errors++ } }
    }
    private fun acousticLoop() {
        var model:AcousticBackend?=null
        val ring=AudioRing();val detector=TemporalDetector();var expected=-1L;var token=-1L;var previousEnd=0L
        val id=synchronized(gate) { session };var retryAt=Long.MIN_VALUE;var retryMs=1000L
        try {
            while(running.get()) {
                val b=audio.poll(50,TimeUnit.MILLISECONDS) ?: continue
                acousticBusy=true
                try {
                    if(model==null && nanoClock()<retryAt) continue
                    if(model==null) { model=acousticFactory();retryMs=1000L;acousticReady=true;status("acoustic_ready_monitoring_only") }
                    if(b.start!=expected || b.captureEpoch!=token) { ring.clear(b.start);detector.resetContinuity() }
                    expected=b.start+b.pcm.size;token=b.captureEpoch
                    ring.add(b.pcm,b.pcm.size) { samples,time ->
                        val end=time*16
                        if(previousEnd>0) metrics.skip(((end-previousEnd)/8000-1).coerceAtLeast(0))
                        previousEnd=end
                        val wall=nanoClock();val cpu=cpuClock()
                        val score=requireNotNull(model).scream(samples)
                        require(score.isFinite() && score in 0f..1f)
                        val duration=(nanoClock()-wall).coerceAtLeast(0);latencyMs=duration/1000000
                        metrics.inference("yamnet",duration,(cpuClock()-cpu).coerceAtLeast(0))
                        if(detector.update(time,score)) cue("yamnet",end,token,null,id)
                    }
                } catch(_:Exception) {
                    synchronized(gate) { errors++ };status("acoustic_unavailable_manual_sos_independent")
                    ring.clear(expected.coerceAtLeast(0));detector.resetContinuity()
                    try { model?.close() } catch(_:Exception) {};model=null;acousticReady=false
                    retryAt=nanoClock()+retryMs*1000000;retryMs=(retryMs*2).coerceAtMost(30000)
                } finally { b.pcm.fill(0);acousticBusy=false;audioPending.decrementAndGet() }
            }
        } finally { try { model?.close() } catch(_:Exception) {};ring.clear();acousticReady=false }
    }
    private fun speechLoop() {
        var model:SpeechRecognizer?=null;var selected:String?=null;var token=-1L;var expected=-1L
        val buffer=FloatArray(segmentSamples);var size=0
        val history=ArrayDeque<Pair<Long,RiskTurn>>()
        val classifier=MonitoringRules();var retryAt=Long.MIN_VALUE;var retryMs=1000L
        var retrySelection:Pair<Boolean,String?>?=null
        val id=synchronized(gate) { session }
        try {
            while(running.get()) {
                var state=synchronized(gate) { Triple(epoch,enabled,language) }
                if(state.first!=token) {
                    history.clear();buffer.fill(0f);size=0;expected=-1
                    val selection=state.second to state.third
                    if(selection!=retrySelection) { retryAt=Long.MIN_VALUE;retryMs=1000L;retrySelection=selection }
                    if(!state.second || state.third!=selected) {
                        try { model?.close() } catch(_:Exception) {};model=null;selected=null;speechReady=false
                    }
                    token=state.first
                }
                val b=speech.poll(50,TimeUnit.MILLISECONDS) ?: continue
                speechBusy=true
                try {
                    // A control change can arrive while poll() waits. Adopt it
                    // before consuming its first block, not after discarding it.
                    state=synchronized(gate) { Triple(epoch,enabled,language) }
                    if(state.first!=token) {
                        history.clear();buffer.fill(0f);size=0;expected=-1
                    val selection=state.second to state.third
                    if(selection!=retrySelection) { retryAt=Long.MIN_VALUE;retryMs=1000L;retrySelection=selection }
                        if(!state.second || state.third!=selected) {
                            try { model?.close() } catch(_:Exception) {};model=null;selected=null;speechReady=false
                        }
                        token=state.first
                    }
                    if(!state.second || state.third==null || b.epoch!=token || token!=controlGeneration) continue
                    if(b.start!=expected) { history.clear();buffer.fill(0f);size=0 }
                    expected=b.start+b.pcm.size
                    if(model==null && nanoClock()<retryAt) continue
                    if(model==null) { model=speechFactory(state.third!!);retryMs=1000L;selected=state.third;speechReady=true;status("speech_ready_unvalidated_"+selected) }
                    if(token!=controlGeneration) continue
                    for(i in b.pcm.indices) {
                        buffer[size++]=b.pcm[i]/32768f
                        if(size==buffer.size) {
                            val end=b.start+i+1
                            var energy=0.0;buffer.forEach { energy+=it*it }
                            if(kotlin.math.sqrt(energy/buffer.size)>=.008) {
                                val wall=nanoClock();val cpu=cpuClock()
                                val recognition=requireNotNull(model).transcribe(buffer)
                                metrics.inference("asr",(nanoClock()-wall).coerceAtLeast(0),(cpuClock()-cpu).coerceAtLeast(0))
                                if(token!=controlGeneration || !running.get()) break
                                while(history.isNotEmpty() && end-history.first().first>30*16000) history.removeFirst()
                                val risk=classifier.classify(recognition,history.map { it.second },selected)
                                history.addLast(end to RiskTurn(recognition.text,recognition.uncertain))
                                while(history.size>4) history.removeFirst()
                                if(risk.labels!=setOf("neutral_or_uncertain")) cue("conversation",end,token,selected,id,risk.asrUncertain)
                            }
                            buffer.fill(0f);size=0
                        }
                    }
                } catch(error:Exception) {
                    synchronized(gate) { errors++ }
                    status(if(error.message=="model_storage_unavailable") "speech_model_storage_unavailable" else "speech_unavailable_acoustic_independent")
                    try { model?.close() } catch(_:Exception) {};model=null;selected=null;speechReady=false
                    history.clear();buffer.fill(0f);size=0
                    retryAt=nanoClock()+retryMs*1000000;retryMs=(retryMs*2).coerceAtMost(30000)
                } finally { b.pcm.fill(0);speechBusy=false;speechPending.decrementAndGet() }
            }
        } finally { try { model?.close() } catch(_:Exception) {};buffer.fill(0f);history.clear();speechReady=false }
    }
    fun snapshot():MlState=synchronized(gate) { MlState(session,epoch,running.get(),enabled,language,acousticDrops,speechDrops,errors,audio.size,speech.size,acousticReady,speechReady) }
    fun awaitIdle(timeoutMs:Long=30000):Boolean {
        val deadline=System.nanoTime()+timeoutMs*1000000
        while(System.nanoTime()<deadline) {
            if(audioPending.get()==0 && speechPending.get()==0) return true
            Thread.sleep(5)
        };return false
    }
    fun stop(timeoutMs:Long=5000):Boolean {
        synchronized(gate) { running.set(false);invalidate();clear(audio) }
        val deadline=System.nanoTime()+timeoutMs*1000000
        for(worker in workers) if(worker!==Thread.currentThread()) worker.join(((deadline-System.nanoTime())/1000000).coerceAtLeast(1))
        return workers.none { it.isAlive }
    }
    fun permissionLost():Boolean { status("permission_lost");return stop() }
    fun optOut():Boolean { status("opted_out");return stop() }
}
