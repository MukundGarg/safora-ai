package org.safora.ai

enum class Availability { AVAILABLE, DEGRADED, UNAVAILABLE }
data class FeatureCoverage(val feature:String,val state:Availability,val reason:String,
                           val safetyValidated:Boolean=false)

/** Availability is execution coverage, never accuracy or safety certification. */
fun MlState.coverage():List<FeatureCoverage> {
    val acoustic=when {
        !running -> FeatureCoverage("acoustic",Availability.UNAVAILABLE,"stopped_or_permission_unavailable")
        !acousticReady -> FeatureCoverage("acoustic",Availability.UNAVAILABLE,"model_not_ready_or_failed")
        acousticDrops>0 -> FeatureCoverage("acoustic",Availability.DEGRADED,"audio_dropped_this_session")
        else -> FeatureCoverage("acoustic",Availability.AVAILABLE,"monitoring_only")
    }
    val speech=when {
        !running || !speechEnabled -> FeatureCoverage("speech",Availability.UNAVAILABLE,"stopped_or_opted_out")
        language==null -> FeatureCoverage("speech",Availability.UNAVAILABLE,"unknown_language_no_lid")
        !speechReady -> FeatureCoverage("speech",Availability.UNAVAILABLE,"model_not_ready_or_failed")
        speechDrops>0 -> FeatureCoverage("speech",Availability.DEGRADED,"speech_dropped_this_session")
        else -> FeatureCoverage("speech",Availability.AVAILABLE,"selected_monolingual_model_accuracy_unvalidated")
    }
    return listOf(acoustic,speech,FeatureCoverage("conversation_risk",if(speech.state==Availability.UNAVAILABLE) Availability.UNAVAILABLE else Availability.DEGRADED,
        "UNVALIDATED_HEURISTIC_no_trained_weights"))
}
