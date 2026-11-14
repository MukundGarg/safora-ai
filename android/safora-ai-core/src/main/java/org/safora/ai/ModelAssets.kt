package org.safora.ai

import android.content.Context
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.zip.ZipInputStream

object ModelAssets {
    fun metadata(context: Context)=JSONObject(context.assets.open("model-assets.json").bufferedReader().use { it.readText() })
    fun validateYamnet(context: Context) {
        val expected=metadata(context).getJSONObject("yamnet").getString("sha256")
        val digest=MessageDigest.getInstance("SHA-256")
        context.assets.open("yamnet.tflite").use { input ->
            val block=ByteArray(65536)
            while(true) { val n=input.read(block);if(n<0) break;digest.update(block,0,n) }
        }
        check(digest.digest().joinToString("") { "%02x".format(it) }==expected) { "YAMNet asset checksum mismatch" }
    }
    /** Extract only bundled, checksum-verified archives to app-private storage.
     * Atomic directory publication; no network, external storage or ZIP traversal. */
    suspend fun prepareVosk(context: Context, language: String): File {
        require(language in setOf("en","hi"))
        val metadata=metadata(context).getJSONObject("vosk").getJSONObject(language)
        val expected=metadata.getString("sha256")
        val parent=File(context.filesDir,"models").apply { mkdirs() }
        val target=File(parent,"vosk-$language-${expected.take(12)}")
        if(File(target,".ready").readTextOrNull()==expected) return target
        val temp=File(parent,target.name+".partial").apply { deleteRecursively();mkdirs() }
        try {
            val digest=MessageDigest.getInstance("SHA-256")
            context.assets.open(metadata.getString("asset")).use { input ->
                val block=ByteArray(65536)
                while(true) { currentCoroutineContext().ensureActive();val n=input.read(block);if(n<0) break;digest.update(block,0,n) }
            }
            check(digest.digest().joinToString("") { "%02x".format(it) }==expected) { "ASR asset checksum mismatch" }
            var total=0L;var entries=0
            val prefix=metadata.getString("root")+"/"
            ZipInputStream(context.assets.open(metadata.getString("asset"))).use { zip ->
                val buffer=ByteArray(65536)
                while(true) {
                    currentCoroutineContext().ensureActive()
                    val entry=zip.nextEntry ?: break
                    check(++entries<10000 && entry.name.startsWith(prefix)) { "Invalid model archive" }
                    val relative=entry.name.removePrefix(prefix)
                    if(relative.isEmpty()) continue
                    val dest=File(temp,relative)
                    check(dest.canonicalPath.startsWith(temp.canonicalPath+File.separator)) { "Unsafe model archive path" }
                    if(entry.isDirectory) dest.mkdirs() else {
                        dest.parentFile?.mkdirs()
                        dest.outputStream().use { output ->
                            while(true) {
                                currentCoroutineContext().ensureActive()
                                val n=zip.read(buffer);if(n<0) break
                                total+=n;check(total<300L*1024*1024) { "Oversized model archive" };output.write(buffer,0,n)
                            }
                        }
                    }
                    zip.closeEntry()
                }
            }
            check(File(temp,"am/final.mdl").isFile && File(temp,"conf/model.conf").isFile) { "Incomplete Vosk model" }
            File(temp,".ready").writeText(expected)
            target.deleteRecursively();check(temp.renameTo(target)) { "Model publication failed" }
            return target
        } finally { temp.deleteRecursively() }
    }
    private fun File.readTextOrNull(): String?=if(isFile) readText() else null
}
