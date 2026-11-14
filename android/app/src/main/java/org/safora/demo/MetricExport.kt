package org.safora.demo
import org.json.JSONObject
import org.safora.ai.MetricsSnapshot
fun MetricsSnapshot.toJson(): String=JSONObject().apply {
    put("version",1);put("yamnet_runs",yamnetRuns);put("yamnet_wall_mean_ms",yamnetWallMeanMs)
    put("yamnet_wall_max_ms",yamnetWallMaxMs);put("yamnet_worker_cpu_mean_ms",yamnetCpuMeanMs)
    put("asr_runs",asrRuns);put("asr_wall_mean_ms",asrWallMeanMs);put("asr_wall_max_ms",asrWallMaxMs)
    put("asr_worker_cpu_mean_ms",asrCpuMeanMs);put("app_dropped_samples",droppedSamples)
    put("app_dropped_blocks",droppedBlocks);put("skipped_analysis_windows_estimate",skippedWindows)
    put("queue_high_water",queueHighWater);put("current_pss_kb",currentPssKb)
    put("peak_sampled_pss_kb",peakSampledPssKb);put("process_peak_rss_kb",peakRssKb)
}.toString(2)
