package com.example.rag_library

import kotlin.math.sqrt

/**
 * 모델 파일 없이 동작하는 문자 n-gram 해싱 임베더 (Phase 0 기본 구현).
 *
 * 문자 1~3-gram 을 부호 있는 해싱 트릭(signed hashing trick)으로 [dim]차원에 누적한 뒤
 * L2 정규화한다. 표면 형태(문자열)가 겹치는 텍스트끼리 높은 코사인 유사도를 갖는다.
 *
 * 의미(시맨틱) 임베딩이 아니므로 데모·테스트·파이프라인 검증용이다.
 * 시맨틱 검색은 ONNX e5 구현으로 교체 예정([Embedder] 참고).
 */
class HashingEmbedder(override val dim: Int = 256) : Embedder {

    init {
        require(dim > 0) { "dim must be > 0" }
    }

    override fun embed(text: String): FloatArray {
        val vec = FloatArray(dim)
        val t = text.trim().lowercase()
        if (t.isEmpty()) return vec // 빈 텍스트는 영벡터 (검색 점수 0)

        for (n in 1..3) {
            if (t.length < n) break
            for (i in 0..t.length - n) {
                val h = mix(t.substring(i, i + n).hashCode())
                val idx = Math.floorMod(h, dim)
                val sign = if (((h ushr 20) and 1) == 0) 1f else -1f
                vec[idx] += sign
            }
        }

        // L2 정규화 — Embedder 계약. (C++ 인덱스도 방어적으로 재정규화한다)
        var sumSq = 0.0
        for (x in vec) sumSq += x.toDouble() * x
        val norm = sqrt(sumSq)
        if (norm > 1e-12) {
            val inv = (1.0 / norm).toFloat()
            for (i in vec.indices) vec[i] *= inv
        }
        return vec
    }

    // 해시 비트 분산용 정수 믹서 — idx(하위 비트)와 sign(20번 비트)이 다른 비트를 쓰게 한다
    private fun mix(seed: Int): Int {
        var h = seed
        h = h xor (h ushr 16)
        h *= 0x45d9f3b
        h = h xor (h ushr 16)
        return h
    }
}
