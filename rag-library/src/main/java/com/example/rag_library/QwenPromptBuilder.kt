package com.example.rag_library

/**
 * Qwen2.5 계열(ChatML 템플릿)용 RAG 프롬프트 빌더.
 *
 * 검색된 청크를 [문서] 섹션으로 넣고, 문서 근거로만 답하도록 시스템 프롬프트로 고정한다.
 * 특수 토큰(<|im_start|> 등)은 네이티브 토크나이저가 parse_special=true 로 통짜 파싱한다.
 */
internal object QwenPromptBuilder {

    private const val SYSTEM_PROMPT =
        "너는 문서 검색 비서다. 아래 [문서] 발췌에 있는 내용만 근거로 정확하고 간결하게 한국어로 답한다. " +
            "문서에 답이 없으면 \"문서에서 답을 찾을 수 없습니다\"라고 답한다."

    fun build(question: String, contextChunks: List<Chunk>): String = buildString {
        append("<|im_start|>system\n")
        append(SYSTEM_PROMPT)
        append("<|im_end|>\n")
        append("<|im_start|>user\n")
        if (contextChunks.isNotEmpty()) {
            append("[문서]\n")
            contextChunks.forEachIndexed { i, chunk ->
                append(i + 1).append(". ").append(chunk.text.trim()).append('\n')
            }
            append('\n')
        }
        append("질문: ").append(question.trim())
        append("<|im_end|>\n")
        append("<|im_start|>assistant\n")
    }
}
