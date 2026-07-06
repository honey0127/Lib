package com.example.rag_library

import java.io.Closeable

/**
 * 온디바이스 RAG 파사드 — 문서 추가와 유사도 검색.
 *
 * ```
 * RagEngine().use { rag ->
 *     rag.addDocument("guide", "김치는 발효 음식이다. 배추와 고춧가루로 만든다.")
 *     val top = rag.search("발효 음식", topK = 3)
 * }
 * ```
 *
 * - 완전 오프라인: 청킹 → 임베딩 → 벡터 검색이 전부 기기 안에서 끝난다.
 * - 네이티브(C++) 인덱스를 쥐고 있으므로 사용 후 반드시 [close] (또는 `use {}`) 할 것.
 * - 모든 공개 메서드는 스레드 안전(@Synchronized).
 */
class RagEngine(
    private val embedder: Embedder = HashingEmbedder(),
    private val chunker: TextChunker = TextChunker(),
) : Closeable {

    private val index = NativeVectorIndex(embedder.dim)
    private val chunks = ArrayList<Chunk>()
    private var closed = false

    /** [text]를 청크로 분할·임베딩해 인덱스에 추가하고, 추가된 청크 수를 반환한다. */
    @Synchronized
    fun addDocument(docId: String, text: String): Int {
        checkOpen()
        val newChunks = chunker.chunk(docId, text)
        for (c in newChunks) {
            index.add(chunks.size, embedder.embed(c.text))
            chunks.add(c)
        }
        return newChunks.size
    }

    /** [query]와 가장 유사한 청크 [topK]개를 유사도 내림차순으로 반환한다. */
    @Synchronized
    fun search(query: String, topK: Int = 5): List<SearchResult> {
        checkOpen()
        require(topK > 0) { "topK must be > 0" }
        if (chunks.isEmpty() || query.isBlank()) return emptyList()
        return index.search(embedder.embedQuery(query), topK)
            .map { SearchResult(chunks[it.id], it.score) }
    }

    /** 지금까지 인덱싱된 청크 수. */
    @Synchronized
    fun chunkCount(): Int = chunks.size

    @Synchronized
    override fun close() {
        if (!closed) {
            closed = true
            index.close()
        }
    }

    private fun checkOpen() = check(!closed) { "RagEngine is closed" }
}
