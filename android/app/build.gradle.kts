import java.security.MessageDigest
plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }
android {
    namespace = "org.safora.demo"
    compileSdk = 35
    buildToolsVersion = "35.0.0"
    defaultConfig { applicationId = "org.safora.demo"; minSdk = 26; targetSdk = 35; versionCode = 2; versionName = "0.2"; testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner" }
    defaultConfig { ndk { abiFilters += listOf("arm64-v8a","armeabi-v7a","x86","x86_64") } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    androidResources { noCompress += listOf("tflite", "zip") }
    kotlinOptions { jvmTarget = "17" }
}
dependencies {
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:core:1.6.1")
    androidTestImplementation("androidx.test:rules:1.6.1")
    implementation(project(":safora-ai-core"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}

val verifyOfflineAssets by tasks.registering {
    doLast {
        val source=rootProject.projectDir.parentFile.resolve("models/android-assets.json")
        val bundled=file("src/main/assets/model-assets.json")
        check(bundled.isFile && bundled.readText()==source.readText()) {
            "Missing/stale models. Run python scripts/prepare_android_assets.py from the repository root."
        }
        val manifest=groovy.json.JsonSlurper().parse(source) as Map<*,*>
        val models=listOf(manifest["yamnet"]) + (manifest["vosk"] as Map<*,*>).values
        for(entry in models) {
            val spec=entry as Map<*,*>
            val asset=file("src/main/assets/${spec["asset"]}")
            check(asset.isFile) { "Missing model ${spec["asset"]}; provision assets before building." }
            val digest=MessageDigest.getInstance("SHA-256")
            asset.inputStream().use { input ->
                val block=ByteArray(65536)
                while(true) { val n=input.read(block);if(n<0) break;digest.update(block,0,n) }
            }
            check(digest.digest().joinToString("") { "%02x".format(it) }==spec["sha256"]) { "Model checksum mismatch: $asset" }
        }
        for(name in listOf("MODEL-NOTICES.txt","MODEL-LICENSE-APACHE-2.0.txt")) check(file("src/main/assets/$name").isFile)
    }
}
tasks.named("preBuild").configure { dependsOn(verifyOfflineAssets) }
