import groovy.json.JsonSlurper

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val repoRoot = rootProject.projectDir.parentFile
val models = repoRoot.resolve("models")
@Suppress("UNCHECKED_CAST")
val mobileCfg = JsonSlurper().parse(models.resolve("mobile/mobile_config.json")) as Map<String, Any>

android {
    namespace = "ir.platereader.app"
    compileSdk = 35
    buildToolsVersion = "35.0.1"
    defaultConfig {
        applicationId = "ir.platereader.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 4
        versionName = "2.2.0"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }  // phones (and Apple-silicon emulators)
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")  // replace with your own key before publishing
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { viewBinding = true; buildConfig = true }
    androidResources { noCompress += "onnx" }
    sourceSets["main"].assets.srcDirs(layout.buildDirectory.dir("generated/modelAssets"))
}

// Mobile models (INT8, chosen on the validation split by scripts/07_optimize_mobile.py) + settings as simple assets.
val prepareModelAssets by tasks.registering {
    val out = layout.buildDirectory.dir("generated/modelAssets")
    inputs.files(models.resolve("mobile/lpd_mobile.onnx"), models.resolve("mobile/lpr_mobile.onnx"),
                 models.resolve("mobile/mobile_config.json"), models.resolve("lpr_config.json"))
    outputs.dir(out)
    doLast {
        val dir = out.get().asFile.apply { deleteRecursively(); mkdirs() }  // no stale models from older builds
        models.resolve("mobile/lpd_mobile.onnx").copyTo(dir.resolve("lpd_mobile.onnx"), overwrite = true)
        models.resolve("mobile/lpr_mobile.onnx").copyTo(dir.resolve("lpr_mobile.onnx"), overwrite = true)
        @Suppress("UNCHECKED_CAST")
        val cfg = JsonSlurper().parse(models.resolve("lpr_config.json"), "UTF-8") as Map<String, Any>
        @Suppress("UNCHECKED_CAST") val letters = cfg["letter_vocab"] as List<String>
        @Suppress("UNCHECKED_CAST") val latin = cfg["letter_vocab_latin"] as List<String>
        dir.resolve("letters.tsv").writeText(letters.indices.joinToString("\n") { "${letters[it]}\t${latin[it]}" }, Charsets.UTF_8)
        dir.resolve("pipeline.properties").writeText(
            "yolo_imgsz=${mobileCfg["imgsz"]}\nyolo_conf_threshold=${mobileCfg["conf"]}\ncrop_padding=${cfg["crop_padding"]}\n")
    }
}
tasks.named("preBuild") { dependsOn(prepareModelAssets) }

dependencies {
    implementation(project(":core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.core:core-splashscreen:1.0.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.fragment:fragment-ktx:1.8.5")
    implementation("androidx.recyclerview:recyclerview:1.3.2")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    val camerax = "1.4.0"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
}
