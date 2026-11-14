package org.safora.ai

import java.text.Normalizer

data class RiskTurn(val text:String,val asrUncertain:Boolean)
data class MonitoringRisk(val labels:Set<String>,val uncertain:Boolean=true,
                          val source:String="UNVALIDATED_HEURISTIC",val version:String="risk-v2-rules",
                          val language:String?,val asrUncertain:Boolean,val scores:Map<String,Float>?=null)

/** Task 3 bounded pattern evidence port. No learned weights, LID, consent or identity inference. */
class MonitoringRules {
    private val refusal=Regex("""\b(?:i said no|i don't want (?:this|to)|stop touching|leave me alone|mana kiya|mujhe mat chhuo)\b|(?:मना किया|छूना बंद|मुझे छोड़|मुझे मत छुओ)""")
    private val pressure=Regex("""\b(?:do it anyway|you have no choice|i don't care|phir bhi|karna padega)\b|(?:फिर भी|कोई चारा नहीं|करना पड़ेगा)""")
    private val threat=Regex("""\b(?:i (?:will|gonna) (?:hurt|kill) you|jaan se maar(?: dunga)?|maar dunga)\b|(?:जान से मार दूंगा|जान से मार दूँगा|तुम्हें मार दूंगा)""")
    private val sexual=Regex("""\b(?:sex|touch you|touching|kiss|kapde utaro)\b|(?:सेक्स|चूम|कपड़े उतार|छूना)""")
    private val distress=Regex("""\b(?:help me|i am scared|i'm scared|mujhe dar lag raha|bachao)\b|(?:बचाओ|मुझे डर लग रहा|मेरी मदद करो)""")
    private val secrecy=Regex("""\b(?:don't tell|do not tell|kisi ko mat batana)\b|किसी को मत बताना""")
    private val isolation=Regex("""\b(?:come alone|be alone|somewhere alone|akele aao)\b|अकेले आओ""")
    private val reporting=Regex("""\b(?:news|reported|reporter|headline|movie|film|novel|fiction|screenplay|roleplay|quoted|he said|she said|they said)\b|(?:समाचार|खबर में|फ़िल्म|फिल्म|उसने कहा)""")
    private val negated=Regex("""\b(?:i (?:will not|won't|would not|wouldn't) (?:hurt|kill) you|i (?:do not|don't) (?:want|intend|plan) to (?:hurt|kill) you|not saying|never said|don't say|i won't do it anyway|i will not do it anyway)\b|(?:नहीं मार|nahi maar|nahin maar)""")
    private val quotes=Regex("\"[^\"\\n]*\"|“[^”\\n]*”|‘[^’\\n]*’|(?<!\\w)'[^'\\n]*'(?!\\w)")
    private fun clean(text:String):String {
        require(text.codePoints().noneMatch { it in 0xd800..0xdfff || it in listOf(0x202a,0x202b,0x202c,0x202d,0x202e,0x2066,0x2067,0x2068,0x2069) || Character.getType(it)==Character.CONTROL.toInt() && it !in listOf(9,10,13) })
        return Normalizer.normalize(text,Normalizer.Form.NFC).trim().replace(Regex("\\s+")," ").lowercase(java.util.Locale.ROOT).replace('’','\'')
    }
    private fun direct(text:String)=quotes.replace(text," ").split(Regex("[.!?;।]+"))
        .filter { !reporting.containsMatchIn(it) && !negated.containsMatchIn(it) }.joinToString(" ")
    fun classify(current:Recognition,context:List<RiskTurn>,language:String?):MonitoringRisk {
        val uncertain=current.uncertain || context.any { it.asrUncertain }
        fun neutral()=MonitoringRisk(setOf("neutral_or_uncertain"),language=language,asrUncertain=uncertain)
        if(context.size>100) return neutral()
        return try {
            val t=direct(clean(current.text.take(2048)))
            val selected=context.takeLast(4).map { RiskTurn(clean(it.text.take(2048)),it.asrUncertain) }.toMutableList()
            while(selected.sumOf { it.text.length }>4096) selected.removeAt(0)
            var history=selected.filter { !it.asrUncertain }.joinToString(" ") { direct(it.text) }
            if(selected.lastOrNull()?.text?.let { Regex("\\b(?:i will stop|i respect your no|okay i will not)\\b|मैं रुक जाऊंगा").containsMatchIn(it) }==true) history=""
            val no=refusal.containsMatchIn(t);val press=pressure.containsMatchIn(t);val sex=sexual.containsMatchIn(t);val danger=threat.containsMatchIn(t)
            val labels=mutableSetOf<String>()
            if(danger) labels.add("explicit_threat")
            if(no || distress.containsMatchIn(t)) labels.add("distress_expression")
            if(press && (no || refusal.containsMatchIn(history))) labels.add("refusal_ignored")
            if(sex && (danger || press)) labels.add("sexual_coercion")
            if(sex && no) labels.add("unwanted_sexual_advance")
            if(secrecy.containsMatchIn(t) && (danger || press)) labels.add("secrecy_pressure")
            if(isolation.containsMatchIn(t) && (danger || press)) labels.add("isolation_pressure")
            MonitoringRisk(labels.ifEmpty { setOf("neutral_or_uncertain") },language=language,asrUncertain=uncertain)
        } catch(_:IllegalArgumentException) { neutral() }
    }
}
