package com.example.rag_library

/**
 * 텍스트 → 토큰 id 시퀀스 인터페이스.
 *
 * 반환 시퀀스는 모델이 요구하는 특수 토큰(bos/eos 등)을 포함한 "모델 입력 그대로"여야 하며,
 * 길이는 [maxTokens] 이하여야 한다.
 */
interface Tokenizer {
    fun encode(text: String, maxTokens: Int): IntArray
}
