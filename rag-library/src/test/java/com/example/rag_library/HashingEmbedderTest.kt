package com.example.rag_library

import kotlin.math.sqrt
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HashingEmbedderTest {

    private val embedder = HashingEmbedder(dim = 256)

    // 입력이 이미 L2 정규화돼 있으므로 내적 = 코사인
    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }

    @Test
    fun deterministic() {
        assertArrayEquals(
            embedder.embed("김치는 발효 음식"),
            embedder.embed("김치는 발효 음식"),
            0f,
        )
    }

    @Test
    fun dimAndUnitNorm() {
        val v = embedder.embed("온디바이스 RAG 라이브러리")
        assertEquals(256, v.size)
        var sumSq = 0.0
        for (x in v) sumSq += x.toDouble() * x
        assertEquals(1.0, sqrt(sumSq), 1e-4)
    }

    @Test
    fun blankText_zeroVector() {
        assertTrue(embedder.embed("   ").all { it == 0f })
    }

    @Test
    fun surfaceSimilarity_ranksRelatedTextHigher() {
        val query = embedder.embed("김치는 맛있는 발효 음식이다")
        val related = embedder.embed("김치는 배추로 만든 발효 음식")
        val unrelated = embedder.embed("로켓은 연료를 태워 우주로 간다")
        assertTrue(cosine(query, related) > cosine(query, unrelated))
    }
}
