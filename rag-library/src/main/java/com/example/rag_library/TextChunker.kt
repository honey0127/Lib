package com.example.rag_library

/**
 * 문서를 검색 단위 청크로 분할한다.
 *
 * 문장 경계(`.!?…。` 또는 줄바꿈 뒤)를 우선 지키고, 한 문장이 [chunkSize]를 넘으면 강제 분할한다.
 * 청크 사이에는 최대 [overlap] 글자의 겹침을 둬서 경계에 걸린 문맥이 잘리지 않게 한다.
 *
 * 길이는 소프트 한도다: 청크 최대 길이는 `chunkSize + overlap` 을 넘지 않는다.
 * (겹침 꼬리는 문자 단위로 자르므로 서로게이트 쌍(이모지 등)이 드물게 갈라질 수 있다 — Phase 0 한계)
 */
class TextChunker(
    val chunkSize: Int = 500,
    val overlap: Int = 100,
) {
    init {
        require(chunkSize > 0) { "chunkSize must be > 0" }
        require(overlap in 0 until chunkSize) { "overlap must be in [0, chunkSize)" }
    }

    fun chunk(docId: String, text: String): List<Chunk> {
        if (text.isBlank()) return emptyList()

        // 1) 문장 조각으로 분할 — chunkSize 를 넘는 문장은 강제 분할해 조각 길이 ≤ chunkSize 보장
        val pieces = mutableListOf<String>()
        for (sentence in SENTENCE_BOUNDARY.split(text)) {
            if (sentence.isBlank()) continue
            if (sentence.length <= chunkSize) pieces += sentence
            else pieces += sentence.chunked(chunkSize)
        }

        // 2) 조각을 chunkSize 까지 패킹, 청크 경계에서 overlap 만큼 꼬리를 다음 청크로 이월
        val chunks = mutableListOf<Chunk>()
        val current = StringBuilder()
        for (piece in pieces) {
            if (current.length + piece.length > chunkSize && current.length > overlap) {
                flush(docId, chunks, current)
            }
            current.append(piece)
        }
        if (current.isNotBlank()) {
            chunks += Chunk(docId, chunks.size, current.toString())
        }
        return chunks
    }

    private fun flush(docId: String, chunks: MutableList<Chunk>, current: StringBuilder) {
        val done = current.toString()
        chunks += Chunk(docId, chunks.size, done)
        current.setLength(0)
        if (overlap > 0) {
            current.append(done.takeLast(overlap))
        }
    }

    companion object {
        // 문장 종결부호/줄바꿈 '뒤' 위치에서 분할하는 고정폭 lookbehind (Java 정규식 제약 준수)
        private val SENTENCE_BOUNDARY = Regex("(?<=[.!?…。\n])")
    }
}
