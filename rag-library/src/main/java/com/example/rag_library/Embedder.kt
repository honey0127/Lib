package com.example.rag_library

/**
 * 텍스트 → 고정 차원 벡터 임베딩 인터페이스.
 *
 * Phase 0 기본 구현은 [HashingEmbedder].
 * 다음 슬라이스에서 ONNX Runtime + multilingual-e5-small 구현으로 교체 예정이며,
 * 그때도 이 인터페이스와 [RagEngine] 은 그대로 유지된다.
 */
interface Embedder {
    /** 임베딩 차원. [RagEngine] 이 벡터 인덱스 생성 시 사용한다. */
    val dim: Int

    /** [text]를 길이 [dim]의 벡터로 변환한다. 반환 벡터는 L2 정규화되어 있어야 한다(영벡터 제외). */
    fun embed(text: String): FloatArray
}
