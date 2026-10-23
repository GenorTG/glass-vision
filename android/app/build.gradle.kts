plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("app.cash.paparazzi")
}

// Single source of truth: repo-root VERSION = MAJOR.MINOR.PATCH.BUILD, BUILD bumped on every commit by
// scripts/git-hooks/pre-commit. versionCode must only ever grow (the in-app updater compares it), and BUILD
// resets to 0 on a MINOR bump, so encode all four parts: MAJOR < 21, MINOR/PATCH < 100, BUILD < 10000.
val appVersion: String = rootProject.file("../VERSION").readText().trim()
val appVersionCode: Int = appVersion.split('.').map(String::toInt).let { (major, minor, patch, build) ->
    require(major < 21 && minor < 100 && patch < 100 && build < 10_000) { "VERSION $appVersion out of versionCode range" }
    major * 100_000_000 + minor * 1_000_000 + patch * 10_000 + build
}

// Release signing (GitHub releases): Gradle property (-P / ~/.gradle/gradle.properties) or environment variable.
// Without all four, release builds fail instead of falling back to the debug key; debug builds don't need them.
val releaseSigningKeys = listOf("GV_KEYSTORE_FILE", "GV_KEYSTORE_PASSWORD", "GV_KEY_ALIAS", "GV_KEY_PASSWORD")
val releaseSigning: Map<String, String?> = releaseSigningKeys.associateWith { key ->
    providers.gradleProperty(key).orElse(providers.environmentVariable(key)).orNull?.takeIf { it.isNotBlank() }
}
val releaseSigningProblem: String? = run {
    val missing = releaseSigning.filterValues { it == null }.keys
    val keystore = releaseSigning["GV_KEYSTORE_FILE"]?.let(::file)
    when {
        missing.isNotEmpty() -> "missing ${missing.joinToString()}"
        keystore?.isFile != true -> "GV_KEYSTORE_FILE=${releaseSigning["GV_KEYSTORE_FILE"]} is not a file"
        else -> null
    }
}

android {
    namespace = "pl.genor.glassvision"
    compileSdk = 34

    defaultConfig {
        applicationId = "pl.genor.glassvision"
        minSdk = 26
        targetSdk = 34
        versionCode = appVersionCode
        versionName = appVersion
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            // MediaPipe + ML Kit ship native code for 4 ABIs; 32-bit x86 (old emulators only) is ~15 MB of it.
            // x86_64 stays for the emulator tests.
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    signingConfigs {
        if (releaseSigningProblem == null) {
            create("release") {
                storeFile = file(releaseSigning.getValue("GV_KEYSTORE_FILE")!!)
                storePassword = releaseSigning.getValue("GV_KEYSTORE_PASSWORD")
                keyAlias = releaseSigning.getValue("GV_KEY_ALIAS")
                keyPassword = releaseSigning.getValue("GV_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            // null without the GV_* values: preReleaseBuild below then stops the build (never debug-signed).
            signingConfig = signingConfigs.findByName("release")
            // Off on purpose: a small, inspectable APK for an open-source release.
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        buildConfig = true
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
    packaging {
        // Compress the native libraries (MediaPipe + ML Kit OCR, ~60 MB for 3 ABIs) inside the APK: the APK is
        // downloaded by the in-app updater and from GitHub, so download size beats the install-time extraction
        // AGP's default (stored uncompressed) avoids.
        jniLibs.useLegacyPackaging = true
    }
}

// First task of every release build (assembleRelease, bundleRelease…); debug/test/lintDebug never run it.
tasks.configureEach {
    if (name == "preReleaseBuild") {
        doFirst {
            releaseSigningProblem?.let {
                throw GradleException(
                    "Release signing not configured ($it). Set ${releaseSigningKeys.joinToString()} as environment " +
                        "variables or Gradle properties (-PGV_KEYSTORE_FILE=…). Release APKs are never signed with " +
                        "the debug key."
                )
            }
        }
    }
}

dependencies {
    val cameraxVersion = "1.3.4"

    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("com.google.android.material:material:1.11.0")

    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // On-device recognition ("On this phone"). MediaPipe Tasks Vision: the 0.10.x line; 1.x has the same vision
    // API but adds LiteRT-LM (~21 MB, LLM tasks this app doesn't use).
    // https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/android
    implementation("com.google.mediapipe:tasks-vision:0.10.35")
    // ML Kit Text Recognition v2, bundled model (works offline, ~4 MB per ABI; the unbundled one needs Play
    // services to download it). https://developers.google.com/ml-kit/vision/text-recognition/v2/android
    implementation("com.google.mlkit:text-recognition:16.0.1")
    implementation("com.google.mlkit:text-recognition-chinese:16.0.1")
    implementation("com.google.mlkit:text-recognition-devanagari:16.0.1")
    implementation("com.google.mlkit:text-recognition-japanese:16.0.1")
    implementation("com.google.mlkit:text-recognition-korean:16.0.1")
    // On-device language identification + offline translation; ML Kit downloads language models on first use.
    // https://developers.google.com/ml-kit/language/translation/android
    implementation("com.google.mlkit:language-id:17.0.6")
    implementation("com.google.mlkit:translate:17.0.3")

    // Paparazzi 1.3.x: layoutlib needs Guava's -jre variant, Gradle picks -android without this
    // (https://github.com/cashapp/paparazzi/issues/906).
    constraints {
        testImplementation("com.google.guava:guava") {
            attributes {
                attribute(
                    TargetJvmEnvironment.TARGET_JVM_ENVIRONMENT_ATTRIBUTE,
                    objects.named(TargetJvmEnvironment::class.java, TargetJvmEnvironment.STANDARD_JVM),
                )
            }
            because("LayoutLib and sdk-common depend on Guava's -jre published variant.")
        }
    }

    testImplementation("junit:junit:4.13.2")
    // Real org.json on the JVM test classpath (android.jar only ships stubs).
    testImplementation("org.json:json:20231013")

    // Instrumented tests (emulator/device): the on-device pipelines on real images.
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:core:1.6.1")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("junit:junit:4.13.2")
}
