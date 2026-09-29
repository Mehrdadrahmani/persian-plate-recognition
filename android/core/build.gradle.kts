import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// Pure Kotlin/JVM module: image processing + ONNX Runtime inference, shared by the Android app and unit-tested on
// the desktop JVM against the Python pipeline (ParityTest).
plugins { id("org.jetbrains.kotlin.jvm") }


java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }

val ortVersion = "1.20.0"
dependencies {
    compileOnly("com.microsoft.onnxruntime:onnxruntime:$ortVersion")   // Android app supplies onnxruntime-android
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}

tasks.test {
    val repo = rootProject.projectDir.parentFile
    systemProperty("plate.models.dir", File(repo, "models").absolutePath)
    systemProperty("plate.data.dir", File(repo, "data").absolutePath)
    testLogging { showStandardStreams = true; events("passed", "failed", "skipped") }
}
