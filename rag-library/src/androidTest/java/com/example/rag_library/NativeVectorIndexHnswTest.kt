package com.example.rag_library

import androidx.test.ext.junit.runners.AndroidJUnit4
import java.util.Random
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * HNSW 백엔드가 JNI 경계를 넘어 동작하는지 검증한다.
 * (그래프 품질/재현율은 cpp/tests/hnsw_index_test.cpp 호스트 테스트가 커버)
 */
@RunWith(AndroidJUnit4::class)
class NativeVectorIndexHnswTest {

    @Test
    fun selfSearch_findsItself() {
        NativeVectorIndex.hnsw(8).use { idx ->
            val rnd = Random(7)
            val vecs = List(50) { FloatArray(8) { rnd.nextFloat() - 0.5f } }
            vecs.forEachIndexed { i, v -> idx.add(i, v) }
            assertEquals(50, idx.size())

            for (i in 0 until 10) {
                val hits = idx.search(vecs[i], 1)
                assertEquals(i, hits[0].id)
                assertEquals(1f, hits[0].score, 1e-4f)
            }
        }
    }

    @Test
    fun kLargerThanSize_clamps() {
        NativeVectorIndex.hnsw(4).use { idx ->
            idx.add(1, floatArrayOf(1f, 0f, 0f, 0f))
            assertEquals(1, idx.search(floatArrayOf(1f, 0f, 0f, 0f), 10).size)
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun invalidParams_throw() {
        NativeVectorIndex.hnsw(4, m = 1)
    }
}
