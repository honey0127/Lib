package com.example.rag_library

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PoolingTest {

    @Test
    fun meanPool_averagesRows_thenNormalizes() {
        // 평균 (2, 0) → L2 정규화 (1, 0)
        val hidden = arrayOf(floatArrayOf(1f, 0f), floatArrayOf(3f, 0f))
        assertArrayEquals(floatArrayOf(1f, 0f), Pooling.meanPoolNormalize(hidden), 1e-6f)
    }

    @Test
    fun outputIsUnitNorm() {
        // (3,4) → (0.6, 0.8)
        val v = Pooling.meanPoolNormalize(arrayOf(floatArrayOf(3f, 4f)))
        assertEquals(0.6f, v[0], 1e-6f)
        assertEquals(0.8f, v[1], 1e-6f)
    }

    @Test
    fun zeroHidden_returnsZeroVector_noNaN() {
        val v = Pooling.meanPoolNormalize(arrayOf(floatArrayOf(0f, 0f), floatArrayOf(0f, 0f)))
        assertTrue(v.all { it == 0f })
    }
}
