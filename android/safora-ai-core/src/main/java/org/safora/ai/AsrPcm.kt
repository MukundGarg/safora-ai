package org.safora.ai

/** Shared codec contract: mono normalized floats at 16kHz; never silently clip invalid input. */
fun asrPcm16(samples:FloatArray):ShortArray {
    require(samples.size<=16000*12) { "ASR segment exceeds 12 seconds" }
    require(samples.all { it.isFinite() && it in -1f..1f }) { "ASR input must be finite normalized mono" }
    return ShortArray(samples.size) { kotlin.math.round(samples[it].toDouble()*32768).toInt().coerceIn(-32768,32767).toShort() }
}
fun asrPcm16LeBytes(samples:FloatArray):ByteArray {
    val shorts=asrPcm16(samples)
    return try {
        ByteArray(shorts.size*2).also { bytes ->
            for(i in shorts.indices) {
                bytes[i*2]=(shorts[i].toInt() and 255).toByte()
                bytes[i*2+1]=((shorts[i].toInt() ushr 8) and 255).toByte()
            }
        }
    } finally { shorts.fill(0) }
}
