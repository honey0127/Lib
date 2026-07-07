package com.example.rag_library

/**
 * RAG 파이프라인 완성체 — 검색([RagEngine]) + 답변 생성([LlmEngine]).
 *
 * ```
 * val answer = RagChat(rag, llm).ask("김치는 어떤 음식이야?") { piece ->
 *     print(piece)  // 스트리밍 UI 갱신
 *     true
 * }
 * answer.sources  // 근거 청크(출처 표시용)
 * ```
 *
 * - [ask]는 블로킹 — 백그라운드 스레드에서 부를 것.
 * - 이 클래스는 [rag]/[llm]을 소유하지 않는다 — 수명 관리(close)는 호출측 책임.
 */
class RagChat(
    private val rag: RagEngine,
    private val llm: LlmEngine,
    private val topK: Int = 3,
    private val maxAnswerTokens: Int = 256,
) {

    init {
        require(topK > 0) { "topK must be > 0" }
        require(maxAnswerTokens > 0) { "maxAnswerTokens must be > 0" }
    }

    /** 답변 텍스트와 근거 청크(유사도 내림차순). */
    data class Answer(
        val text: String,
        val sources: List<SearchResult>,
    )

    /**
     * [question]에 대해 상위 [topK] 청크를 근거로 답변을 생성한다.
     * 인덱스가 비어 있으면 근거 없이 시스템 프롬프트 규칙에 따라 답한다(sources 는 빈 목록).
     * [onToken]은 완전한 문자 단위 스트리밍 콜백 — false 반환 시 중단.
     */
    fun ask(question: String, onToken: ((String) -> Boolean)? = null): Answer {
        val sources = rag.search(question, topK)
        val prompt = QwenPromptBuilder.build(question, sources.map { it.chunk })
        val text = llm.generate(prompt, maxAnswerTokens, onToken)
        return Answer(text.trim(), sources)
    }
}
