package org.safora.ai

import org.vosk.Model
import org.vosk.Recognizer
import org.json.JSONObject

/** Local Vosk alternative. Model language must be declared to the user.
 * A monolingual Hindi/English model does not imply code-switching support. */
class VoskSpeech(modelDirectory: String) : SpeechRecognizer {
    private val model=Model(modelDirectory)
    private var closed=false
    @Synchronized override fun transcribe(pcm:FloatArray):Recognition {
        check(!closed) { "ASR is closed" }
        val shorts=asrPcm16(pcm)
        if(shorts.isEmpty()) return Recognition("",true)
        return try {
            Recognizer(model,16000f).use { recognizer ->
                recognizer.setWords(true)
                val parts=mutableListOf<String>();var confidence=0.0;var count=0;var unavailable=false;var malformed=false
                fun collect(raw:String) {
                    try {
                        require(raw.length<=65536)
                        val result=JSONObject(raw)
                        val text=result.get("text") as? String ?: error("Malformed ASR text")
                        require(text.length<=4096 && parts.sumOf { it.length }+text.length<=4096)
                        if(text.isNotBlank()) {
                            parts.add(text.trim())
                            val words=result.optJSONArray("result")
                            if(words==null || words.length()==0) unavailable=true
                            else {
                                require(words.length()<=2048)
                                for(i in 0 until words.length()) {
                                    val value=words.getJSONObject(i).get("conf")
                                    require(value is Number)
                                    val score=value.toDouble();require(score.isFinite() && score in 0.0..1.0)
                                    confidence+=score;count++
                                }
                            }
                        }
                    } catch(_:Exception) { malformed=true }
                }
                for(start in shorts.indices step 1600) {
                    val chunk=shorts.copyOfRange(start,minOf(shorts.size,start+1600))
                    try { if(recognizer.acceptWaveForm(chunk,chunk.size)) collect(recognizer.result) }
                    finally { chunk.fill(0) }
                }
                collect(recognizer.finalResult)
                val text=if(malformed) "" else parts.joinToString(" ")
                Recognition(text,malformed || text.isBlank() || unavailable || count==0 || confidence/count<0.75)
            }
        } finally { shorts.fill(0) }
    }
    @Synchronized override fun close() { if(!closed) { model.close();closed=true } }
}

/** Explicitly unvalidated baseline; every result remains uncertain. */
class DevelopmentClassifier : ConversationClassifier {
    override fun classify(current: Recognition, context: List<String>): ConversationRisk {
        val t=current.text.lowercase()
        val labels=mutableSetOf<String>()
        val threat=Regex("i (will|gonna) (hurt|kill)|jaan se maar|maar dunga|जान से मार|मार दूंगा|मार दूँगा").containsMatchIn(t)
        val refusal=Regex("i said no|stop touching|leave me alone|mana kiya|छूना बंद|मना किया|मुझे छोड़").containsMatchIn(t)
        val pressure=Regex("do it anyway|you have no choice|i don't care|phir bhi|फिर भी|कोई चारा नहीं").containsMatchIn(t)
        if(threat) labels.add("explicit_threat")
        if(refusal) labels.add("distress_expression")
        if(pressure && (refusal || context.any { "said no" in it.lowercase() || "mana kiya" in it.lowercase() })) labels.add("refusal_ignored")
        if(Regex("sex|touching|touch you|kiss").containsMatchIn(t) && (pressure || threat)) labels.add("sexual_coercion")
        val sexual=Regex("sex|touching|touch you|kiss|सेक्स|चूम").containsMatchIn(t)
        if(sexual && refusal) labels.add("unwanted_sexual_advance")
        if(("alone" in t || "अकेले" in t) && (pressure || threat)) labels.add("isolation_pressure")
        if(("don't tell" in t || "मत बताना" in t) && (pressure || threat)) labels.add("secrecy_pressure")
        if(labels.isEmpty()) labels.add("neutral_or_uncertain")
        return ConversationRisk(labels,true)
    }
    override fun close() {}
}
