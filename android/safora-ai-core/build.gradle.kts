plugins { id("com.android.library"); id("org.jetbrains.kotlin.android") }
android {
    namespace = "org.safora.ai"
    compileSdk = 35
    buildToolsVersion = "35.0.0"
    defaultConfig { minSdk = 26; testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner" }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}
dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("com.alphacephei:vosk-android:0.3.47@aar")
    implementation("net.java.dev.jna:jna:5.13.0@aar")
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    testImplementation("junit:junit:4.13.2")
}
