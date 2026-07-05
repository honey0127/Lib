import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.library)
    // Kotlin 플러그인은 별도 적용하지 않는다: AGP 9+ built-in Kotlin support 가 기본 활성화
    // (android.builtInKotlin=true, 기본값)되어 있어 org.jetbrains.kotlin.android 를 함께 적용하면
    // "Cannot add extension with name 'kotlin'" 충돌이 난다. kotlin{} 블록은 그대로 사용 가능.
}

android {
    namespace = "com.example.rag_library"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        minSdk = 26

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        // ⭐ 추가 1: C++ 빌드 시 사용할 플래그 설정 (C++17 표준 강제)
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
            }
        }

        // 빌드할 ABI 제한: 실기기(arm64-v8a) + 에뮬레이터(x86_64). AAR 크기도 축소됨.
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    // ⭐ 추가 2: CMakeLists.txt 파일 위치와 버전을 Gradle에 알려줌
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

// AGP 9 / Kotlin 2.x 권장 DSL (구 android.kotlinOptions 대체)
kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_11)
    }
}

dependencies {
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}
