package org.safora.ai

/** Contains counts/timings only. Never audio, text, speaker information or risk labels. */
data class MetricsSnapshot(
    val yamnetRuns: Long, val yamnetWallMeanMs: Double, val yamnetWallMaxMs: Double,
    val yamnetCpuMeanMs: Double, val asrRuns: Long, val asrWallMeanMs: Double,
    val asrWallMaxMs: Double, val asrCpuMeanMs: Double,
    val droppedSamples: Long, val droppedBlocks: Long, val skippedWindows: Long,
    val queueHighWater: Int, val currentPssKb: Long, val peakSampledPssKb: Long, val peakRssKb: Long
)
class PipelineMetrics {
    private var yamnetRuns=0L;private var yamnetWall=0.0;private var yamnetWallMax=0.0;private var yamnetCpu=0.0
    private var asrRuns=0L;private var asrWall=0.0;private var asrWallMax=0.0;private var asrCpu=0.0
    private var droppedSamples=0L;private var droppedBlocks=0L;private var skippedWindows=0L
    private var queueHighWater=0;private var pss=0L;private var peakPss=0L;private var peakRss=0L
    @Synchronized fun inference(kind: String, wallNanos: Long, cpuNanos: Long) {
        require(wallNanos>=0 && cpuNanos>=0)
        val wall=wallNanos/1e6;val cpu=cpuNanos/1e6
        if(kind=="yamnet") { yamnetRuns++;yamnetWall+=wall;yamnetWallMax=maxOf(yamnetWallMax,wall);yamnetCpu+=cpu }
        else { require(kind=="asr");asrRuns++;asrWall+=wall;asrWallMax=maxOf(asrWallMax,wall);asrCpu+=cpu }
    }
    @Synchronized fun drop(samples: Int) { require(samples>0);droppedSamples+=samples;droppedBlocks++ }
    @Synchronized fun skip(windows: Long) { require(windows>=0);skippedWindows+=windows }
    @Synchronized fun queue(depth: Int) { queueHighWater=maxOf(queueHighWater,depth) }
    @Synchronized fun memory(pssKb: Long, rssHighWaterKb: Long = 0) { require(pssKb>=0 && rssHighWaterKb>=0);pss=pssKb;peakPss=maxOf(peakPss,pss);peakRss=maxOf(peakRss,rssHighWaterKb) }
    @Synchronized fun snapshot()=MetricsSnapshot(yamnetRuns,yamnetWall/maxOf(1,yamnetRuns),yamnetWallMax,
        yamnetCpu/maxOf(1,yamnetRuns),asrRuns,asrWall/maxOf(1,asrRuns),asrWallMax,
        asrCpu/maxOf(1,asrRuns),droppedSamples,droppedBlocks,skippedWindows,queueHighWater,pss,peakPss,peakRss)
}
