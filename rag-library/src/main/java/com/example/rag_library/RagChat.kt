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
     *
     * 이벤트 순서가 핵심이다: **[onSources]가 토큰 생성 시작 전에 먼저 호출**되므로 UI 는
     * 근거 문서를 즉시 표시할 수 있다 — 생성이 느린 기기에서도 체감 대기가 사라진다.
     * [onToken]은 완전한 문자 단위 스트리밍 콜백 — false 반환 시 중단.
     *
     * Flow 를 쓰는 앱이라면 이렇게 감싸면 된다(라이브러리는 무의존 콜백만 제공):
     * ```
     * callbackFlow {
     *     val answer = chat.ask(q,
     *         onSources = { trySend(Event.Sources(it)) },
     *         onToken = { trySend(Event.Token(it)); true })
     *     send(Event.Done(answer)); close()
     * }.flowOn(Dispatchers.IO)
     * ```
     */
    fun ask(
        question: String,
        onSources: ((List<SearchResult>) -> Unit)? = null,
        onToken: ((String) -> Boolean)? = null,
    ): Answer {
        val sources = rag.search(question, topK)
        onSources?.invoke(sources) // 토큰보다 먼저 — 근거를 즉시 UI 에
        val prompt = QwenPromptBuilder.build(question, sources.map { it.chunk })
        val text = llm.generate(prompt, maxAnswerTokens, onToken)
        return Answer(text.trim(), sources)
    }
}
