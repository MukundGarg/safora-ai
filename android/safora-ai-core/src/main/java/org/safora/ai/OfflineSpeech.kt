package org.safora.ai

import org.vosk.Model
import org.vosk.Recognizer
import org.json.JSONObject

/** Local Vosk alternative. Model language must be declared to the user.
 * A monolingual Hindi/English model does not imply code-switching support. */
class VoskSpeech(modelDirectory: String) : SpeechRecognizer {
    private val model = Model(modelDirectory)
    override fun transcribe(pcm: FloatArray): Recognition {
        require(pcm.size <= 16000*12)
        val shorts=ShortArray(pcm.size) { (pcm[it].coerceIn(-1f,1f)*32767).toInt().toShort() }
        return try {
            Recognizer(model,16000f).use { recognizer ->
                recognizer.setWords(true)
                recognizer.acceptWaveForm(shorts,shorts.size)
                val result=JSONObject(recognizer.finalResult)
                val text=result.optString("text","")
                val words=result.optJSONArray("result")
                val confidence=if(words!=null && words.length()>0)
                    (0 until words.length()).map { words.getJSONObject(it).optDouble("conf",0.0) }.average() else 0.0
                Recognition(text,text.isBlank() || confidence<0.75)
            }
        } finally { shorts.fill(0) }
    }
    override fun close() = model.close()
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
