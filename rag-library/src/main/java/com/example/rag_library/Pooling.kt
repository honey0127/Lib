package com.example.rag_library

import kotlin.math.sqrt

/** 트랜스포머 출력 → 문장 임베딩 풀링 유틸. */
internal object Pooling {

    /**
     * mean pooling + L2 정규화.
     *
     * [hidden]은 `last_hidden_state`의 배치 1개 — `[seqLen][dim]`.
     * 배치=1로 패딩 없이 추론하므로 전 토큰 평균 = attention-mask 평균과 동일하다.
     */
    fun meanPoolNormalize(hidden: Array<FloatArray>): FloatArray {
        require(hidden.isNotEmpty()) { "hidden must not be empty" }
        val dim = hidden[0].size
        val out = FloatArray(dim)
        for (row in hidden) {
            require(row.size == dim) { "ragged hidden state" }
            for (j in 0 until dim) out[j] += row[j]
        }
        val inv = 1.0f / hidden.size
        for (j in 0 until dim) out[j] *= inv

        var sumSq = 0.0
        for (x in out) sumSq += x.toDouble() * x
        val norm = sqrt(sumSq)
        if (norm > 1e-12) {
            val invNorm = (1.0 / norm).toFloat()
            for (j in 0 until dim) out[j] *= invNorm
        }
        return out
    }
}
