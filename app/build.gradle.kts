import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

// QNN Hexagon-NPU depth backend (recon/QnnDlcDepthModel.kt + cpp/qnn_depth.cpp) is opt-in: it
// requires the Qualcomm QAIRT/QNN SDK (for headers) plus the NDK/CMake. Enable by setting
// `qnn.sdkRoot=/path/to/qairt/2.45.x` in local.properties (machine-specific, gitignored), or via
// `-Pqnn.sdkRoot=...` / the QNN_SDK_ROOT env var. When unset, the native build is skipped entirely
// so the stock (headless) build needs no NDK, and the app degrades to ExecuTorch/ARCore at runtime.
val qnnSdkRoot: String? = run {
    val fromLocal = rootProject.file("local.properties").takeIf { it.exists() }?.let { f ->
        Properties().apply { f.inputStream().use { load(it) } }.getProperty("qnn.sdkRoot")
    }
    (fromLocal
        ?: project.findProperty("qnn.sdkRoot") as String?
        ?: System.getenv("QNN_SDK_ROOT"))?.takeIf { it.isNotBlank() }
}
val qnnEnabled = qnnSdkRoot != null

android {
    namespace = "com.lantern.recorder"
    compileSdk = 35

    // Pinned to the NDK installed for the QNN native build (no-op when QNN is disabled).
    if (qnnEnabled) ndkVersion = "27.0.12077973"

    defaultConfig {
        applicationId = "com.lantern.recorder"
        // ARCore requires API 24+.
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        if (qnnEnabled) {
            // QNN/HTP is arm64 only; the Hexagon-v79 skel + stubs ship in src/main/jniLibs.
            ndk { abiFilters += "arm64-v8a" }
            externalNativeBuild {
                cmake { arguments += "-DQNN_SDK_ROOT=$qnnSdkRoot" }
            }
        }
    }

    if (qnnEnabled) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "3.22.1"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
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
        compose = true
    }

    packaging {
        jniLibs {
            // Both OpenCV and ExecuTorch's fbjni ship libc++_shared.so; keep one copy.
            pickFirsts += "lib/**/libc++_shared.so"
            // QNN's Hexagon DSP skel (libQnnHtpV79Skel.so) must be a real file on disk for
            // fastRPC to load it onto the NPU; with the modern uncompressed-in-APK default it
            // isn't, and HTP device bring-up fails. Extract native libs so the skel lands in
            // the app's nativeLibraryDir (which we add to ADSP_LIBRARY_PATH at runtime).
            if (qnnEnabled) useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.recyclerview)
    // ARCore SDK (camera, pose, raw depth).
    implementation(libs.arcore)

    // OpenCV (Turntable mode only): ChArUco board detection + solvePnP for the
    // per-frame object pose. Gated behind the capture-mode switch in the UI.
    implementation(libs.opencv)

    // ExecuTorch on-device runtime: runs the Depth-Anything-3 Small .pte (depth) on
    // the Snapdragon NPU/CPU for the live-mesh capture mode. See recon/DepthSource.kt.
    implementation(libs.executorch)

    // Jetpack Compose UI overlay + sessions screen. The BOM keeps every Compose
    // artifact on a single, mutually-compatible version set.
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    debugImplementation(libs.androidx.compose.ui.tooling)
}
