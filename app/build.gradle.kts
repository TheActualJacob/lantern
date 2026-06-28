plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.lantern.recorder"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.lantern.recorder"
        // ARCore requires API 24+.
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
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
