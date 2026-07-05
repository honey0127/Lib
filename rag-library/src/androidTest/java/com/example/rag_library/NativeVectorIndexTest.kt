package com.example.rag_library

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * FloatArray 가 JNI 경계를 넘어 C++ 인덱스까지 왕복하는지 검증한다.
 * (순수 C++ 로직은 cpp/tests 호스트 테스트가 커버 — 여기서는 JNI 경계가 초점)
 */
@RunWith(AndroidJUnit4::class)
class NativeVectorIndexTest {

    @Test
    fun cosineTopK_roundTripsThroughJni() {
        NativeVectorIndex(3).use { idx ->
            idx.add(10, floatArrayOf(2f, 0f, 0f))
            idx.add(11, floatArrayOf(0f, 5f, 0f))
            idx.add(12, floatArrayOf(3f, 3f, 0f))
            assertEquals(3, idx.size())

            val hits = idx.search(floatArrayOf(7f, 0f, 0f), 2)
            assertEquals(2, hits.size)
            assertEquals(10, hits[0].id)
            assertEquals(1f, hits[0].score, 1e-5f)
            assertEquals(12, hits[1].id)
            assertEquals(0.7071f, hits[1].score, 1e-3f)
        }
    }

    @Test
    fun kLargerThanSize_clampsToSize() {
        NativeVectorIndex(2).use { idx ->
            idx.add(1, floatArrayOf(1f, 0f))
            assertEquals(1, idx.search(floatArrayOf(1f, 0f), 10).size)
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun dimMismatch_throws() {
        NativeVectorIndex(3).use { idx ->
            idx.add(1, floatArrayOf(1f, 0f)) // 길이 2 ≠ dim 3
        }
    }

    @Test(expected = IllegalStateException::class)
    fun useAfterClose_throws() {
        val idx = NativeVectorIndex(2)
        idx.close()
        idx.size()
    }
}
