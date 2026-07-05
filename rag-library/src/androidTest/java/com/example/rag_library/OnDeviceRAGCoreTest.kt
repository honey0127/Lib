package com.example.rag_library

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Phase 0 핵심 게이트: Kotlin → C++ → Kotlin JNI 왕복이 실기기/에뮬레이터에서 성공하는지 검증한다.
 *
 * 주의: 로컬 유닛테스트(host JVM, `./gradlew test`)로는 .so 를 로드할 수 없어 JNI 검증이 되지 않는다.
 * 반드시 계측 테스트로 실행할 것:
 *   ./gradlew :rag-library:connectedDebugAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class OnDeviceRAGCoreTest {
    @Test
    fun jniRoundTrip() {
        val result = OnDeviceRAGCore().stringFromJNI()
        assertTrue("JNI 왕복 실패, 반환값=$result", result.contains("C++"))
    }
}
