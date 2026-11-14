package org.safora.ai

import java.io.File
import java.security.MessageDigest

data class ArtifactFile(val bytes:Long,val sha256:String)

/** Pure JVM integrity/storage checks. Metadata is pinned in the source/verified APK. */
object ArtifactFiles {
    fun hash(file:File):String {
        val digest=MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream -> val buffer=ByteArray(65536)
            while(true) { val n=stream.read(buffer);if(n<0) break;digest.update(buffer,0,n) }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
    fun validate(root:File,files:Map<String,ArtifactFile>):Boolean {
        if(files.isEmpty() || !root.isDirectory) return false
        return files.all { (name,spec) ->
            val file=File(root,name)
            !name.contains('\\') && !name.split('/').contains("..") && !File(name).isAbsolute &&
                file.canonicalPath.startsWith(root.canonicalPath+File.separator) && file.isFile &&
                file.length()==spec.bytes && hash(file)==spec.sha256
        }
    }
    fun requiredSpace(uncompressed:Long,entries:Int):Long {
        require(uncompressed in 1..300L*1024*1024 && entries in 1..10000)
        // Upper bound for one 4KiB final-block allocation per file, plus receipt.
        return uncompressed+4096L*(entries+1)
    }
    fun requireStorage(available:Long,needed:Long) {
        check(available>=needed) { "model_storage_unavailable" }
    }
}
