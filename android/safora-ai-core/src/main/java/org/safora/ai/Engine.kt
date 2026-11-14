package org.safora.ai

import java.util.ArrayDeque

data class RiskEvent(val id: Long, val level: Int, val reason: String, val manual: Boolean = false,
                     val sessionId: String = "", val elapsedRealtimeMs: Long = 0)
interface IncidentSink { fun onRisk(event: RiskEvent); fun onCoverage(status: String) }
interface SpeechRecognizer : AutoCloseable { fun transcribe(pcm: FloatArray): Recognition }
data class Recognition(val text: String, val uncertain: Boolean)
interface ConversationClassifier : AutoCloseable { fun classify(current: Recognition, context: List<String>): ConversationRisk }
data class ConversationRisk(val labels: Set<String>, val uncertain: Boolean)

class TemporalDetector(private val threshold: Float = .30f, private val hits: Int = 2,
                       private val windows: Int = 3, private val cooldownMs: Long = 10000) {
    init { require(threshold in 0f..1f && hits in 1..windows && cooldownMs >= 0) }
    private val history = ArrayDeque<Boolean>()
    private var last = -cooldownMs
    private var clock = -1L
    fun resetContinuity() { history.clear() }
    fun update(now: Long, score: Float): Boolean {
        require(now >= clock && score.isFinite() && score in 0f..1f)
        clock = now
        history.addLast(score >= threshold)
        if (history.size > windows) history.removeFirst()
        if (history.count { it } >= hits && now-last >= cooldownMs) {
            last = now; history.clear(); return true
        }
        return false
    }
}

class DecisionEngine(private val sink: IncidentSink, private val cooldownMs: Long = 10000) {
    private var last = -cooldownMs
    private var activeLevel = 0
    private var sequence = 0L
    private val sessionId=java.util.UUID.randomUUID().toString()
    @Synchronized fun emit(now: Long, level: Int, reason: String, manual: Boolean = false) {
        if (!manual && (level <= activeLevel || (activeLevel == 0 && now-last < cooldownMs))) return
        last = now; activeLevel = level
        sink.onRisk(RiskEvent(++sequence, level, reason, manual, sessionId, now))
    }
    fun acoustic(now: Long) = emit(now, 2, "temporally_confirmed_scream")
    fun conversation(now: Long, risk: ConversationRisk) {
        if (risk.labels.isEmpty() || risk.labels == setOf("neutral_or_uncertain")) return
        val strong = "explicit_threat" in risk.labels || risk.labels.containsAll(setOf("refusal_ignored", "sexual_coercion"))
        emit(now, if (strong && !risk.uncertain) 2 else 1, risk.labels.sorted().joinToString(","))
    }
    fun manual(now: Long) = emit(now, 2, "manual_sos", true)
    @Synchronized fun resolve() { activeLevel = 0 }
}

enum class Mode { LITE, STANDARD, ENHANCED }
data class ResourceSample(val availableMb: Long, val lowMemory: Boolean, val thermal: Int,
                          val latencyMs: Long, val backlog: Int, val batteryPercent: Int,
                          val processPssMb: Long = 0, val appVisible: Boolean = true)
class ResourceController(var requested: Mode = Mode.STANDARD) {
    var mode = Mode.LITE; private set
    private var healthy = 0
    @Synchronized fun update(s: ResourceSample): Mode {
        if (s.lowMemory || s.availableMb < 180 || s.thermal >= 3 || s.latencyMs > 1500 || s.backlog >= 2 || s.batteryPercent < 10) {
            healthy = 0; mode = Mode.LITE
        } else if (++healthy >= 3) {
            mode = when {
                requested == Mode.ENHANCED && s.availableMb > 700 && s.appVisible -> Mode.ENHANCED
                requested != Mode.LITE && s.availableMb > 350 -> Mode.STANDARD
                else -> Mode.LITE
            }
        }
        return mode
    }
}
